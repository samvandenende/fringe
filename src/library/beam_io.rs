use std::{f64::consts::TAU, path::Path, sync::Arc};

use num_complex::{Complex32, Complex64};
use pyo3::prelude::*;

use super::{
    beam::{Beam, BeamModel, C, Normalization, TabulatedBeam},
    error::Error,
    utils::{read_cf32, write_cf32},
};

/// Header of a beam file in fringe's native format.
///
/// The field is stored next to it in a raw interleaved little-endian 32-bit float IQ file
/// (`cf32_le`), holding `E_θ` and `E_φ` of every grid point, in the order
/// `[frequency][θ][φ][E_θ, E_φ]`.
#[derive(serde::Serialize, serde::Deserialize)]
struct BeamFileJson {
    normalization: String,
    /// Frequencies (Hz).
    frequencies: Vec<f64>,
    theta: ThetaGridJson,
    phi: PhiGridJson,
    /// Effective area (m²) per unit `|E|²` at each frequency, for relative beams.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    effective_area: Option<Vec<f64>>,
    /// Path of the field file, relative to the beam file.
    data: String,
}

/// Regular grid of `θ` (radians), measured from the beam's +z axis.
#[derive(serde::Serialize, serde::Deserialize)]
struct ThetaGridJson {
    start: f64,
    step: f64,
    count: usize,
}

/// Regular grid of `φ` (radians) around the full circle, with step `2π / count`.
#[derive(serde::Serialize, serde::Deserialize)]
struct PhiGridJson {
    start: f64,
    count: usize,
}

/// Saves a tabulated beam in fringe's native format.
///
/// The beam is written as a JSON header at `filepath`, with its field in a raw interleaved
/// little-endian 32-bit float IQ file (`cf32_le`) next to it, named `<file stem>.cf32`.
///
/// # Errors
/// Returns an `Error` (an `IOError` in Python) if the beam is analytic, or if a file cannot
/// be written.
#[pyfunction]
pub fn save_beam(beam: &Beam, filepath: &str) -> Result<(), Error> {
    let Some(t) = beam.as_tabulated() else {
        return Err(Error::new(
            "analytic beams are not saved to files, they are stored in the array or calibrator file itself",
        ));
    };
    write_tabulated(t, Path::new(filepath))
}

/// Loads a tabulated beam saved in fringe's native format.
///
/// # Errors
/// Returns an `Error` (an `IOError` in Python) if a file cannot be read or does not hold a
/// valid beam.
#[pyfunction]
pub fn load_beam(filepath: &str) -> Result<Beam, Error> {
    read_tabulated(Path::new(filepath)).map(Beam::tabulated)
}

fn write_tabulated(t: &TabulatedBeam, path: &Path) -> Result<(), Error> {
    let stem = path
        .file_stem()
        .ok_or_else(|| Error::new(format!("Invalid file path '{}'", path.display())))?
        .to_string_lossy();
    let data_name = format!("{stem}.cf32");
    let directory = path.parent().unwrap_or(Path::new(""));

    let mut samples = Vec::with_capacity(t.field.len() * 2);
    for f in 0..t.frequencies.len() {
        for i in 0..t.theta_count {
            for j in 0..t.phi_count {
                let (e_theta, e_phi) = t.spherical_at(f, i, j);
                for e in [e_theta, e_phi] {
                    samples.push(Complex32::new(e.re as f32, e.im as f32));
                }
            }
        }
    }
    write_cf32(&directory.join(&data_name), &samples)?;

    let header = BeamFileJson {
        normalization: t.normalization.name().into(),
        frequencies: t.frequencies.clone(),
        theta: ThetaGridJson {
            start: t.theta_start,
            step: t.theta_step,
            count: t.theta_count,
        },
        phi: PhiGridJson {
            start: t.phi_start,
            count: t.phi_count,
        },
        effective_area: t.effective_area.clone(),
        data: data_name,
    };
    let file = std::fs::File::create(path)
        .map_err(|e| Error::new(format!("Failed to create file '{}': {}", path.display(), e)))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &header)
        .map_err(|e| Error::new(format!("Failed to serialize beam: {}", e)))
}

fn read_tabulated(path: &Path) -> Result<TabulatedBeam, Error> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::new(format!("Failed to open file '{}': {}", path.display(), e)))?;
    let header: BeamFileJson = serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| Error::new(format!("Failed to deserialize beam: {}", e)))?;
    let invalid = |e: String| Error::new(format!("Invalid beam '{}': {}", path.display(), e));

    let normalization = Normalization::parse(&header.normalization).map_err(invalid)?;
    let (nf, nt, np) = (
        header.frequencies.len(),
        header.theta.count,
        header.phi.count,
    );
    let directory = path.parent().unwrap_or(Path::new(""));
    let samples = read_cf32(&directory.join(&header.data))?;
    if samples.len() != nf * nt * np * 2 {
        return Err(invalid(format!(
            "the field file holds {} values, expected {}",
            samples.len(),
            nf * nt * np * 2
        )));
    }

    let angles = (0..nt)
        .flat_map(|i| {
            let theta = header.theta.start + i as f64 * header.theta.step;
            (0..np).map(move |j| (theta, header.phi.start + j as f64 * TAU / np as f64))
        })
        .collect::<Vec<_>>();
    let to64 = |c: Complex32| Complex64::new(c.re as f64, c.im as f64);
    let fields = samples
        .chunks(nt * np * 2)
        .map(|chunk| {
            chunk
                .chunks(2)
                .map(|pair| (to64(pair[0]), to64(pair[1])))
                .collect()
        })
        .collect();
    TabulatedBeam::from_samples(
        header.frequencies,
        &angles,
        fields,
        normalization,
        header.effective_area,
    )
    .map_err(invalid)
}

/// Reference to a beam in an array or calibrator file.
///
/// Analytic beams are stored inline. Tabulated beams refer to a beam file in fringe's
/// native format, or directly to a far field computed by openEMS. Paths are relative to the
/// referring file.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BeamSpec {
    CosSquared,
    Isotropic {
        /// `E_θ` as `[re, im]`.
        e_theta: [f64; 2],
        /// `E_φ` as `[re, im]`.
        e_phi: [f64; 2],
    },
    ShortDipole,
    Tabulated {
        file: String,
    },
    Openems {
        file: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        incident_power: Option<Vec<f64>>,
        /// Excitation per frequency as `[re, im]`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        excitation: Option<Vec<[f64; 2]>>,
    },
}

impl BeamSpec {
    /// Loads the beam, resolving paths relative to `directory`.
    pub(crate) fn resolve(&self, directory: &Path) -> Result<Beam, Error> {
        let path = |file: &str| directory.join(file);
        let io_error = |file: &str, e: String| {
            Error::new(format!(
                "Failed to load beam '{}': {}",
                path(file).display(),
                e
            ))
        };
        Ok(match self {
            BeamSpec::CosSquared => Beam::cos_squared(),
            BeamSpec::Isotropic { e_theta, e_phi } => Beam::isotropic(
                Complex64::new(e_theta[0], e_theta[1]),
                Complex64::new(e_phi[0], e_phi[1]),
            ),
            BeamSpec::ShortDipole => Beam::short_dipole(),
            BeamSpec::Tabulated { file } => Beam::tabulated(read_tabulated(&path(file))?),
            BeamSpec::Openems {
                file,
                incident_power,
                excitation,
            } => {
                let excitation = excitation.as_ref().map(|e| {
                    e.iter()
                        .map(|&[re, im]| Complex64::new(re, im))
                        .collect::<Vec<_>>()
                });
                Beam::tabulated(
                    load_openems(
                        &path(file),
                        incident_power.as_deref(),
                        excitation.as_deref(),
                    )
                    .map_err(|e| io_error(file, e))?,
                )
            }
        })
    }
}

/// Writes the beams of a file, with tabulated beams next to it as
/// `<file stem>.beam<index>.json`. Beams that are the same are written once.
pub(crate) struct BeamWriter<'a> {
    directory: &'a Path,
    stem: String,
    written: Vec<(Arc<TabulatedBeam>, String)>,
}

impl<'a> BeamWriter<'a> {
    pub(crate) fn new(path: &'a Path) -> Result<Self, Error> {
        let stem = path
            .file_stem()
            .ok_or_else(|| Error::new(format!("Invalid file path '{}'", path.display())))?
            .to_string_lossy()
            .into_owned();
        Ok(BeamWriter {
            directory: path.parent().unwrap_or(Path::new("")),
            stem,
            written: Vec::new(),
        })
    }

    pub(crate) fn spec(&mut self, beam: &Beam) -> Result<BeamSpec, Error> {
        Ok(match &beam.model {
            BeamModel::CosSquared => BeamSpec::CosSquared,
            BeamModel::Isotropic { e_theta, e_phi } => BeamSpec::Isotropic {
                e_theta: [e_theta.re, e_theta.im],
                e_phi: [e_phi.re, e_phi.im],
            },
            BeamModel::ShortDipole => BeamSpec::ShortDipole,
            BeamModel::Tabulated(t) => {
                if let Some((_, name)) = self.written.iter().find(|(w, _)| Arc::ptr_eq(w, t)) {
                    return Ok(BeamSpec::Tabulated { file: name.clone() });
                }
                let name = format!("{}.beam{}.json", self.stem, self.written.len());
                write_tabulated(t, &self.directory.join(&name))?;
                self.written.push((t.clone(), name.clone()));
                BeamSpec::Tabulated { file: name }
            }
        })
    }
}

/// Loads a far field computed by openEMS, see [`Beam::load_openems`].
///
/// The file holds the grid in `/Mesh/theta` and `/Mesh/phi`, the frequencies, radiated
/// power and maximum directivity as the attributes `Frequency`, `Prad` and `Dmax` of
/// `/nf2ff`, and the field of frequency `n` in `/nf2ff/E_theta/FD/f<n>_real`, `..._imag`
/// and likewise for `E_phi`, stored with `φ` as the first dimension. The field is that at
/// the radius `/Mesh/r`, including the propagation phase `exp(-jkr)`, and the phase of the
/// excitation signal of the simulation.
pub(crate) fn load_openems(
    path: &Path,
    incident_power: Option<&[f64]>,
    excitation: Option<&[Complex64]>,
) -> Result<TabulatedBeam, String> {
    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let read = |name: &str| -> Result<Vec<f64>, String> {
        file.dataset(name)
            .and_then(|d| d.read_raw::<f64>())
            .map_err(|e| format!("cannot read '{name}': {e}"))
    };
    let nf2ff = file
        .group("nf2ff")
        .map_err(|e| format!("cannot open group 'nf2ff': {e}"))?;
    let attribute = |name: &str| -> Result<Vec<f64>, String> {
        nf2ff
            .attr(name)
            .and_then(|a| a.read_raw::<f64>())
            .map_err(|e| format!("cannot read attribute '{name}' of 'nf2ff': {e}"))
    };

    let mut theta = read("Mesh/theta")?;
    let mut phi = read("Mesh/phi")?;
    // angles are in radians, unless they exceed 2π
    for angles in [&mut theta, &mut phi] {
        if angles.iter().any(|a| a.abs() > TAU + 1e-6) {
            angles.iter_mut().for_each(|a| *a = a.to_radians());
        }
    }
    let radius = *read("Mesh/r")?.first().ok_or("'Mesh/r' is empty")?;
    let frequencies = attribute("Frequency")?;
    let dmax = attribute("Dmax")?;
    let prad = attribute("Prad")?;
    let nf = frequencies.len();
    if dmax.len() != nf || prad.len() != nf {
        return Err(format!(
            "{nf} frequencies, but {} values of Dmax and {} of Prad",
            dmax.len(),
            prad.len()
        ));
    }
    if let Some(p) = incident_power
        && p.len() != nf
    {
        return Err(format!(
            "the file holds {nf} frequencies, but {} incident powers were given",
            p.len()
        ));
    }
    if let Some(e) = excitation {
        if e.len() != nf {
            return Err(format!(
                "the file holds {nf} frequencies, but {} excitation values were given",
                e.len()
            ));
        }
        if e.iter().any(|e| !e.is_finite() || e.norm() == 0.0) {
            return Err("excitation values must be finite and not zero".into());
        }
    }

    let (nt, np) = (theta.len(), phi.len());
    let angles = theta
        .iter()
        .flat_map(|&t| phi.iter().map(move |&p| (t, p)))
        .collect::<Vec<_>>();
    let mut fields = Vec::with_capacity(nf);
    for n in 0..nf {
        let mut components = Vec::with_capacity(2);
        for component in ["E_theta", "E_phi"] {
            let name = format!("nf2ff/{component}/FD/f{n}_real");
            let shape = file
                .dataset(&name)
                .map_err(|e| format!("cannot read '{name}': {e}"))?
                .shape();
            let re = read(&name)?;
            let im = read(&format!("nf2ff/{component}/FD/f{n}_imag"))?;
            // openEMS stores φ as the first dimension
            let phi_first = match shape.as_slice() {
                [a, b] if *a == np && *b == nt => true,
                [a, b] if *a == nt && *b == np => false,
                _ => {
                    return Err(format!(
                        "'{name}' has shape {shape:?}, expected {np} x {nt} (φ x θ)"
                    ));
                }
            };
            let value = |i: usize, j: usize| {
                let k = if phi_first { j * nt + i } else { i * np + j };
                Complex64::new(re[k], im[k])
            };
            components.push(
                (0..nt)
                    .flat_map(|i| (0..np).map(move |j| (i, j)))
                    .map(|(i, j)| value(i, j))
                    .collect::<Vec<_>>(),
            );
        }
        let peak = components[0]
            .iter()
            .zip(&components[1])
            .map(|(a, b)| a.norm_sqr() + b.norm_sqr())
            .fold(0.0, f64::max);
        if peak.is_nan() || peak <= 0.0 || dmax[n].is_nan() || dmax[n] <= 0.0 {
            return Err(format!(
                "the field or Dmax is zero at {:.6} MHz",
                frequencies[n] / 1e6
            ));
        }
        // scale the peak to Dmax, and from directivity to realized gain if requested
        let mut scale = dmax[n] / peak;
        if let Some(p) = incident_power {
            if p[n].is_nan() || p[n] <= 0.0 {
                return Err("incident powers must be positive".into());
            }
            scale *= prad[n] / p[n];
        }
        // the phase relative to the excitation, without the propagation to the radius
        let k = TAU * frequencies[n] / C;
        let reference = match excitation {
            Some(e) => e[n] / e[n].norm() * Complex64::from_polar(1.0, -k * radius),
            None => peak_phase(&angles, &components[0], &components[1]),
        };
        let scale = reference.conj() * scale.sqrt();
        fields.push(
            components[0]
                .iter()
                .zip(&components[1])
                .map(|(a, b)| (a * scale, b * scale))
                .collect(),
        );
    }

    let normalization = if incident_power.is_some() {
        Normalization::RealizedGain
    } else {
        Normalization::Directivity
    };
    TabulatedBeam::from_samples(frequencies, &angles, fields, normalization, None)
}

/// The phase (unit phasor) of the largest Cartesian component of the field in the direction
/// of its peak, which does not depend on the `θ`/`φ` convention of the grid.
fn peak_phase(angles: &[(f64, f64)], e_theta: &[Complex64], e_phi: &[Complex64]) -> Complex64 {
    let peak = (0..angles.len())
        .max_by(|&a, &b| {
            let power = |i: usize| e_theta[i].norm_sqr() + e_phi[i].norm_sqr();
            power(a).total_cmp(&power(b))
        })
        .unwrap_or(0);
    let (t, p) = angles[peak];
    let theta_hat = [t.cos() * p.cos(), t.cos() * p.sin(), -t.sin()];
    let phi_hat = [-p.sin(), p.cos(), 0.0];
    let component = (0..3)
        .map(|i| e_theta[peak] * theta_hat[i] + e_phi[peak] * phi_hat[i])
        .max_by(|a, b| a.norm_sqr().total_cmp(&b.norm_sqr()))
        .unwrap();
    if component.norm() > 0.0 {
        component / component.norm()
    } else {
        Complex64::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::Vec3;
    use crate::library::beam::direction_from_angles;
    use crate::library::beam::tests::{sample_field, tabulate};
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("fringe-test-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn assert_same_field(a: &Beam, b: &Beam, frequency: f64, tolerance: f64) {
        for i in 0..50 {
            let direction = direction_from_angles(i as f64 * 0.061, i as f64 * 0.37);
            let (x, y) = (a.field(direction, frequency), b.field(direction, frequency));
            let error = (x.0 - y.0).norm() + (x.1 - y.1).norm();
            assert!(error < tolerance, "{direction:?}: {x:?} vs {y:?}");
        }
    }

    #[test]
    fn native_round_trip() {
        let directory = temp_dir("beam");
        let path = directory.join("dipole.json");
        let beam = tabulate(&Beam::short_dipole(), &[40e6, 60e6], 5.0, 10.0);
        save_beam(&beam, path.to_str().unwrap()).unwrap();
        assert!(directory.join("dipole.cf32").exists());
        let loaded = load_beam(path.to_str().unwrap()).unwrap();
        assert_same_field(&beam, &loaded, 47e6, 1e-6);
        std::fs::remove_dir_all(directory).unwrap();
    }

    /// Writes a far field in the openEMS nf2ff layout, with `θ ∈ [-180°, 180°)` and
    /// `φ ∈ [0°, 180°)` as in the openEMS examples, and in units that need normalization.
    /// The field is that at a radius of 3 m, of an excitation that is a pulse delayed by
    /// `OPENEMS_DELAY`.
    fn write_openems(path: &Path, beam: &Beam, frequencies: &[f64]) {
        let theta = (0..72)
            .map(|i| (i as f64 * 5.0 - 180.0).to_radians())
            .collect::<Vec<_>>();
        let phi = (0..36)
            .map(|j| (j as f64 * 5.0).to_radians())
            .collect::<Vec<_>>();
        let file = hdf5::File::create(path).unwrap();
        let mesh = file.create_group("Mesh").unwrap();
        for (name, values) in [
            ("theta", &theta),
            ("phi", &phi),
            ("r", &vec![OPENEMS_RADIUS]),
        ] {
            mesh.new_dataset::<f64>()
                .shape(values.len())
                .create(name)
                .unwrap()
                .write_raw(values)
                .unwrap();
        }
        let nf2ff = file.create_group("nf2ff").unwrap();
        for (name, values) in [
            ("Frequency", frequencies.to_vec()),
            ("Dmax", vec![1.5; frequencies.len()]),
            ("Prad", vec![2.0; frequencies.len()]),
        ] {
            nf2ff
                .new_attr::<f64>()
                .shape(values.len())
                .create(name)
                .unwrap()
                .write_raw(&values)
                .unwrap();
        }
        for component in ["E_theta", "E_phi"] {
            let fd = nf2ff
                .create_group(component)
                .unwrap()
                .create_group("FD")
                .unwrap();
            for (n, &f) in frequencies.iter().enumerate() {
                // stored with φ as the first dimension, in arbitrary units
                let (mut re, mut im) = (Vec::new(), Vec::new());
                for &p in &phi {
                    for &t in &theta {
                        let (e_theta, e_phi) = sample_field(beam, t, p, f);
                        let e = if component == "E_theta" {
                            e_theta
                        } else {
                            e_phi
                        } * openems_phase(f)
                            * 37.0;
                        re.push(e.re);
                        im.push(e.im);
                    }
                }
                for (part, values) in [("real", re), ("imag", im)] {
                    fd.new_dataset::<f64>()
                        .shape((phi.len(), theta.len()))
                        .create(format!("f{n}_{part}").as_str())
                        .unwrap()
                        .write_raw(&values)
                        .unwrap();
                }
            }
        }
    }

    const OPENEMS_RADIUS: f64 = 3.0;
    const OPENEMS_DELAY: f64 = 40e-9;

    /// Phase of the far field computed by openEMS: the propagation to the radius, and the
    /// delay of the excitation pulse.
    fn openems_phase(frequency: f64) -> Complex64 {
        Complex64::from_polar(1.0, -TAU * frequency * (OPENEMS_RADIUS / C + OPENEMS_DELAY))
    }

    #[test]
    fn openems_layout() {
        let directory = temp_dir("openems");
        let path = directory.join("nf2ff.h5");
        let dipole = Beam::short_dipole();
        let frequencies = [40e6, 60e6];
        write_openems(&path, &dipole, &frequencies);
        // without the excitation, the phase is referenced to the peak direction
        let loaded = Beam::load_openems(path.to_str().unwrap(), None, None).unwrap();
        assert_eq!(loaded.normalization(), "directivity");
        for f in frequencies {
            assert_same_field(&dipole, &loaded, f, 0.03);
        }
        // with the excitation, only its phase is removed
        let excitation = frequencies
            .iter()
            .map(|&f| Complex64::from_polar(2.0, -TAU * f * OPENEMS_DELAY) * Complex64::i())
            .collect::<Vec<_>>();
        let loaded =
            Beam::load_openems(path.to_str().unwrap(), None, Some(excitation.clone())).unwrap();
        let direction = Vec3::new(0.3, 0.4, 0.866).normalized();
        let ratio = loaded.field(direction, 40e6).0 / dipole.field(direction, 40e6).0;
        assert!((ratio / ratio.norm() - Complex64::new(0.0, -1.0)).norm() < 1e-2);
        let gain = Beam::load_openems(path.to_str().unwrap(), Some(vec![4.0, 4.0]), None).unwrap();
        let direction = Vec3::new(0.0, 0.0, 1.0);
        assert!((gain.gain(direction, 50e6) - 0.75).abs() < 1e-3);
        assert!(
            Beam::load_openems(path.to_str().unwrap(), None, Some(excitation[..1].to_vec()))
                .is_err()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
