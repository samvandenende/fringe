use std::{
    f64::consts::{PI, TAU},
    fmt::Write,
    sync::Arc,
};

use num_complex::{Complex32, Complex64};
use pyo3::prelude::*;

use super::{Vec3, beam_io, error::Error};

/// Speed of light in vacuum (m/s).
pub(crate) const C: f64 = 299792458.0;
/// Directivity of a short dipole.
const SHORT_DIPOLE_DIRECTIVITY: f64 = 1.5;

/// Complex 3D vector, used for far fields and receive vectors.
pub(crate) type CVec3 = [Complex64; 3];

/// Unconjugated dot product of two complex vectors.
pub(crate) fn cdot(a: CVec3, b: CVec3) -> Complex64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unconjugated dot product of a complex and a real vector.
pub(crate) fn cdot_real(a: CVec3, b: Vec3) -> Complex64 {
    a[0] * b.x + a[1] * b.y + a[2] * b.z
}

/// Euclidean norm of a complex vector.
pub(crate) fn cnorm(a: CVec3) -> f64 {
    a.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Unit vectors `θ̂` and `φ̂` of the spherical coordinate system at a direction.
///
/// At the poles, `φ` is taken as 0.
pub(crate) fn spherical_basis(direction: Vec3) -> (Vec3, Vec3) {
    let phi = direction.y.atan2(direction.x);
    let rho = direction.x.hypot(direction.y);
    let (cos_theta, sin_theta) = (direction.z, rho);
    let theta_hat = Vec3::new(cos_theta * phi.cos(), cos_theta * phi.sin(), -sin_theta);
    let phi_hat = Vec3::new(-phi.sin(), phi.cos(), 0.0);
    (theta_hat, phi_hat)
}

/// Unit vector in the direction of spherical angles `θ` and `φ` (radians).
pub(crate) fn direction_from_angles(theta: f64, phi: f64) -> Vec3 {
    Vec3::new(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    )
}

/// Unit vectors `θ̂` and `φ̂` at spherical angles `θ` and `φ` (radians).
///
/// Unlike [`spherical_basis`], this is well defined at the poles, where it depends on `φ`.
pub(crate) fn spherical_basis_at(theta: f64, phi: f64) -> (Vec3, Vec3) {
    let theta_hat = Vec3::new(
        theta.cos() * phi.cos(),
        theta.cos() * phi.sin(),
        -theta.sin(),
    );
    let phi_hat = Vec3::new(-phi.sin(), phi.cos(), 0.0);
    (theta_hat, phi_hat)
}

/// Converts spherical field components to a Cartesian vector.
pub(crate) fn to_cartesian(
    e_theta: Complex64,
    e_phi: Complex64,
    theta_hat: Vec3,
    phi_hat: Vec3,
) -> CVec3 {
    [
        e_theta * theta_hat.x + e_phi * phi_hat.x,
        e_theta * theta_hat.y + e_phi * phi_hat.y,
        e_theta * theta_hat.z + e_phi * phi_hat.z,
    ]
}

/// Orientation of a beam in the world frame.
///
/// The beam's own frame is the frame of the antenna model it was simulated in: `θ` is
/// measured from its +z axis and `φ` from its +x axis. The orientation places the model's
/// +z axis along `boresight` and its +x axis as close to `reference` as possible.
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orientation {
    pub(crate) x: Vec3,
    pub(crate) y: Vec3,
    pub(crate) z: Vec3,
}

#[pymethods]
impl Orientation {
    /// Creates an orientation from the direction of the beam's +z axis and a reference
    /// direction for its +x axis.
    ///
    /// # Arguments
    /// - `boresight`: Direction of the beam's +z axis in the world frame.
    /// - `reference`: Direction the beam's +x axis points as close to as possible.
    ///
    /// # Panics
    /// Panics if `boresight` is zero or not finite, or if `reference` is parallel to it.
    #[new]
    pub fn new(boresight: Vec3, reference: Vec3) -> Self {
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        assert!(
            finite(boresight) && boresight.norm() > 0.0,
            "boresight must be non-zero and finite"
        );
        assert!(finite(reference), "reference must be finite");
        let z = boresight.normalized();
        let x = reference - z * reference.dot(z);
        assert!(
            x.norm() > 1e-9 * reference.norm(),
            "reference must not be parallel to boresight"
        );
        let x = x.normalized();
        Orientation {
            x,
            y: z.cross(x),
            z,
        }
    }

    /// Returns the orientation in which the beam's frame equals the world frame.
    #[staticmethod]
    pub fn identity() -> Self {
        Orientation {
            x: Vec3::new(1.0, 0.0, 0.0),
            y: Vec3::new(0.0, 1.0, 0.0),
            z: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    /// Returns the direction of the beam's +z axis in the world frame.
    pub fn boresight(&self) -> Vec3 {
        self.z
    }

    /// Returns the direction of the beam's +x axis in the world frame.
    pub fn reference(&self) -> Vec3 {
        self.x
    }

    /// Returns this orientation rotated about its boresight by `angle` (radians), turning
    /// the +x axis towards the +y axis.
    pub fn rotated(&self, angle: f64) -> Orientation {
        let (sin, cos) = angle.sin_cos();
        Orientation {
            x: self.x * cos + self.y * sin,
            y: self.y * cos - self.x * sin,
            z: self.z,
        }
    }

    /// Expresses a world frame vector in the beam's frame.
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }

    /// Expresses a beam frame vector in the world frame.
    pub fn to_world(&self, v: Vec3) -> Vec3 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Orientation(boresight = {}, reference = {})",
            self.z.__repr__(),
            self.x.__repr__()
        )
    }
}

impl Orientation {
    /// Creates an orientation like [`Orientation::new`], but falls back to an arbitrary
    /// reference perpendicular to `boresight` if `reference` is parallel to it.
    pub(crate) fn new_or_any(boresight: Vec3, reference: Vec3) -> Self {
        let z = boresight.normalized();
        let x = reference - z * reference.dot(z);
        if x.norm() > 1e-9 * reference.norm().max(1e-300) {
            return Orientation::new(boresight, reference);
        }
        let axis = if z.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        Orientation::new(boresight, axis)
    }

    /// Expresses a complex beam frame vector in the world frame.
    pub(crate) fn to_world_c(self, v: CVec3) -> CVec3 {
        [
            v[0] * self.x.x + v[1] * self.y.x + v[2] * self.z.x,
            v[0] * self.x.y + v[1] * self.y.y + v[2] * self.z.y,
            v[0] * self.x.z + v[1] * self.y.z + v[2] * self.z.z,
        ]
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Orientation::identity()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OrientationJson {
    boresight: Vec3,
    reference: Vec3,
}

impl serde::Serialize for Orientation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        OrientationJson {
            boresight: self.z,
            reference: self.x,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Orientation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let json = OrientationJson::deserialize(deserializer)?;
        Ok(Orientation::new(json.boresight, json.reference))
    }
}

/// Attitude of a calibrator, which orients its transmitter's beam.
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attitude {
    pub(crate) kind: AttitudeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AttitudeKind {
    /// The beam has a fixed orientation in the world frame.
    Fixed(Orientation),
    /// The beam's boresight tracks `target`, its +x axis as close to `reference` as possible.
    PointAt { target: Vec3, reference: Vec3 },
}

#[pymethods]
impl Attitude {
    /// Creates an attitude with a fixed orientation in the world frame.
    #[staticmethod]
    pub fn fixed(orientation: Orientation) -> Self {
        Attitude {
            kind: AttitudeKind::Fixed(orientation),
        }
    }

    /// Creates an attitude whose boresight tracks a point, such as the array center.
    ///
    /// # Arguments
    /// - `target`: Point (m) the beam's +z axis points at.
    /// - `reference`: Direction the beam's +x axis points as close to as possible. If it is
    ///   parallel to the boresight, an arbitrary perpendicular direction is used.
    ///
    /// # Panics
    /// Panics if `target` or `reference` is not finite.
    #[staticmethod]
    pub fn point_at(target: Vec3, reference: Vec3) -> Self {
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        assert!(finite(target), "target must be finite");
        assert!(
            finite(reference) && reference.norm() > 0.0,
            "reference must be non-zero and finite"
        );
        Attitude {
            kind: AttitudeKind::PointAt { target, reference },
        }
    }

    /// Computes the orientation of the beam of a calibrator at `position`.
    pub fn orientation_at(&self, position: Vec3) -> Orientation {
        match self.kind {
            AttitudeKind::Fixed(orientation) => orientation,
            AttitudeKind::PointAt { target, reference } => {
                let boresight = target - position;
                if boresight.norm() == 0.0 {
                    Orientation::identity()
                } else {
                    Orientation::new_or_any(boresight, reference)
                }
            }
        }
    }

    pub fn __repr__(&self) -> String {
        match self.kind {
            AttitudeKind::Fixed(orientation) => {
                format!("Attitude.fixed({})", orientation.__repr__())
            }
            AttitudeKind::PointAt { target, reference } => format!(
                "Attitude.point_at(target = {}, reference = {})",
                target.__repr__(),
                reference.__repr__()
            ),
        }
    }
}

impl Default for Attitude {
    fn default() -> Self {
        Attitude::fixed(Orientation::identity())
    }
}

/// What the values of a tabulated beam represent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Normalization {
    /// `|E|²` is the directivity.
    Directivity,
    /// `|E|²` is the gain, including radiation efficiency.
    Gain,
    /// `|E|²` is the realized gain, including radiation efficiency and mismatch.
    RealizedGain,
    /// `|E|²` is relative; the effective area is given separately.
    Relative,
}

impl Normalization {
    pub(crate) fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_lowercase().replace([' ', '-'], "_").as_str() {
            "directivity" => Ok(Normalization::Directivity),
            "gain" => Ok(Normalization::Gain),
            "realized_gain" | "realised_gain" => Ok(Normalization::RealizedGain),
            "relative" => Ok(Normalization::Relative),
            _ => Err(format!(
                "unknown normalization '{name}', expected one of directivity, gain, realized_gain, relative"
            )),
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Normalization::Directivity => "directivity",
            Normalization::Gain => "gain",
            Normalization::RealizedGain => "realized_gain",
            Normalization::Relative => "relative",
        }
    }
}

/// Far-field pattern tabulated on a regular grid of `θ`, `φ` and frequency.
///
/// The field is stored as Cartesian vectors in the beam's frame, which avoids the
/// singularity of the spherical basis at the poles during interpolation.
pub(crate) struct TabulatedBeam {
    /// Frequencies (Hz), strictly ascending.
    pub(crate) frequencies: Vec<f64>,
    pub(crate) theta_start: f64,
    pub(crate) theta_step: f64,
    pub(crate) theta_count: usize,
    pub(crate) phi_start: f64,
    /// Number of `φ` values, spaced `2π / phi_count` apart around the full circle.
    pub(crate) phi_count: usize,
    pub(crate) normalization: Normalization,
    /// Effective area (m²) per unit `|E|²` at each frequency, for relative beams.
    pub(crate) effective_area: Option<Vec<f64>>,
    /// Indexed by `(frequency * theta_count + theta) * phi_count + phi`.
    pub(crate) field: Vec<[Complex32; 3]>,
    /// Maximum of `|E|²` over the grid at each frequency.
    pub(crate) peak: Vec<f64>,
}

impl TabulatedBeam {
    pub(crate) fn phi_step(&self) -> f64 {
        TAU / self.phi_count as f64
    }

    pub(crate) fn theta(&self, i: usize) -> f64 {
        self.theta_start + i as f64 * self.theta_step
    }

    pub(crate) fn phi(&self, j: usize) -> f64 {
        self.phi_start + j as f64 * self.phi_step()
    }

    fn index(&self, f: usize, i: usize, j: usize) -> usize {
        (f * self.theta_count + i) * self.phi_count + j
    }

    /// Finds the frequencies bracketing `frequency` and the weight of the upper one.
    /// Frequencies outside the tabulated range are clamped to it.
    fn locate_frequency(&self, frequency: f64) -> (usize, usize, f64) {
        let n = self.frequencies.len();
        if n == 1 || frequency <= self.frequencies[0] {
            return (0, 0, 0.0);
        }
        if frequency >= self.frequencies[n - 1] {
            return (n - 1, n - 1, 0.0);
        }
        let upper = self.frequencies.partition_point(|&f| f <= frequency);
        let (f0, f1) = (self.frequencies[upper - 1], self.frequencies[upper]);
        (upper - 1, upper, (frequency - f0) / (f1 - f0))
    }

    /// Interpolates a per-frequency quantity.
    fn interpolate_frequency(&self, values: &[f64], frequency: f64) -> f64 {
        let (f0, f1, w) = self.locate_frequency(frequency);
        values[f0] * (1.0 - w) + values[f1] * w
    }

    /// Interpolates the field in a direction (beam frame unit vector) at a frequency.
    ///
    /// The field is zero outside the tabulated `θ` range.
    pub(crate) fn field_at(&self, direction: Vec3, frequency: f64) -> CVec3 {
        let theta = direction.z.clamp(-1.0, 1.0).acos();
        let u = (theta - self.theta_start) / self.theta_step;
        let last = (self.theta_count - 1) as f64;
        if !(-1e-6..=last + 1e-6).contains(&u) {
            return [Complex64::ZERO; 3];
        }
        let u = u.clamp(0.0, last);
        let i0 = (u.floor() as usize).min(self.theta_count - 2);
        let wu = u - i0 as f64;

        let phi = direction.y.atan2(direction.x);
        let v = ((phi - self.phi_start) / self.phi_step()).rem_euclid(self.phi_count as f64);
        let j0 = (v.floor() as usize) % self.phi_count;
        let j1 = (j0 + 1) % self.phi_count;
        let wv = v - v.floor();

        let (f0, f1, wf) = self.locate_frequency(frequency);
        let mut result = [Complex64::ZERO; 3];
        for (f, weight_f) in [(f0, 1.0 - wf), (f1, wf)] {
            for (i, weight_i) in [(i0, 1.0 - wu), (i0 + 1, wu)] {
                for (j, weight_j) in [(j0, 1.0 - wv), (j1, wv)] {
                    let weight = weight_f * weight_i * weight_j;
                    if weight == 0.0 {
                        continue;
                    }
                    let value = self.field[self.index(f, i, j)];
                    for (r, c) in result.iter_mut().zip(value) {
                        *r += Complex64::new(c.re as f64, c.im as f64) * weight;
                    }
                }
            }
        }
        result
    }

    /// Builds a tabulated beam from far-field samples on an arbitrary regular grid.
    ///
    /// Samples may use any `θ`/`φ` convention, such as `θ ∈ [-180°, 180°)` with
    /// `φ ∈ [0°, 180°)`: every sample is mapped to the direction it represents, and these
    /// directions must form a regular grid with `θ ∈ [0, π]` and `φ` covering the full
    /// circle. Samples at a pole fill the whole pole row of the grid.
    ///
    /// # Arguments
    /// - `frequencies`: Frequencies (Hz) of the samples.
    /// - `angles`: `(θ, φ)` of every sample (radians), shared by all frequencies.
    /// - `fields`: `(E_θ, E_φ)` of every sample, per frequency.
    /// - `normalization`: What `|E|²` represents.
    /// - `effective_area`: Effective area (m²) per unit `|E|²` per frequency, for relative beams.
    ///
    /// # Errors
    /// Returns a description of the problem if the samples do not form a valid grid.
    pub(crate) fn from_samples(
        frequencies: Vec<f64>,
        angles: &[(f64, f64)],
        fields: Vec<Vec<(Complex64, Complex64)>>,
        normalization: Normalization,
        effective_area: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        if frequencies.is_empty() {
            return Err("a beam needs at least one frequency".into());
        }
        if fields.len() != frequencies.len() {
            return Err(format!(
                "{} frequencies, but fields for {}",
                frequencies.len(),
                fields.len()
            ));
        }
        if let Some(f) = fields.iter().find(|f| f.len() != angles.len()) {
            return Err(format!(
                "{} angles, but {} field values for a frequency",
                angles.len(),
                f.len()
            ));
        }
        if frequencies.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            return Err("frequencies must be positive".into());
        }
        // sort by frequency
        let mut order = (0..frequencies.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| frequencies[a].total_cmp(&frequencies[b]));
        if order
            .windows(2)
            .any(|w| frequencies[w[0]] == frequencies[w[1]])
        {
            return Err("frequencies must be distinct".into());
        }
        let sorted_frequencies = order.iter().map(|&i| frequencies[i]).collect::<Vec<_>>();
        let effective_area = match (normalization, effective_area) {
            (Normalization::Relative, None) => {
                return Err("a relative beam needs an effective area".into());
            }
            (Normalization::Relative, Some(area)) => {
                if area.len() != frequencies.len() {
                    return Err(format!(
                        "{} frequencies, but {} effective areas",
                        frequencies.len(),
                        area.len()
                    ));
                }
                if area.iter().any(|a| !a.is_finite() || *a <= 0.0) {
                    return Err("effective areas must be positive".into());
                }
                Some(order.iter().map(|&i| area[i]).collect())
            }
            (_, Some(_)) => {
                return Err("only relative beams take an effective area".into());
            }
            (_, None) => None,
        };

        if angles
            .iter()
            .any(|(theta, phi)| !theta.is_finite() || !phi.is_finite())
        {
            return Err("angles must be finite".into());
        }
        let samples = angles
            .iter()
            .map(|&(theta, phi)| Sample::new(theta, phi))
            .collect::<Vec<_>>();

        let thetas = unique_sorted(samples.iter().map(|s| s.theta));
        let phis = unique_sorted(samples.iter().filter(|s| !s.pole).map(|s| s.phi));
        if thetas.len() < 2 {
            return Err("a beam needs at least two distinct values of θ".into());
        }
        if phis.is_empty() {
            return Err("a beam needs samples away from the poles".into());
        }
        let theta_step = regular_step(&thetas).ok_or(
            "the samples do not form a regular grid in θ (after mapping to θ ∈ [0°, 180°])",
        )?;
        let phi_count = phis.len();
        let phi_step = TAU / phi_count as f64;
        let phi_start = phis[0];
        if phis
            .iter()
            .enumerate()
            .any(|(j, phi)| (phi - (phi_start + j as f64 * phi_step)).abs() > 1e-3 * phi_step)
        {
            return Err(format!(
                "the samples must form a regular grid in φ around the full circle (after mapping to θ ∈ [0°, 180°]); found {phi_count} values of φ from {:.3}° to {:.3}°",
                phis[0].to_degrees(),
                phis[phi_count - 1].to_degrees()
            ));
        }
        let theta_start = thetas[0];
        let theta_count = thetas.len();

        let mut beam = TabulatedBeam {
            frequencies: sorted_frequencies,
            theta_start,
            theta_step,
            theta_count,
            phi_start,
            phi_count,
            normalization,
            effective_area,
            field: vec![[Complex32::ZERO; 3]; frequencies.len() * theta_count * phi_count],
            peak: Vec::new(),
        };

        // grid index of every sample, `None` in phi for samples at a pole
        let indices = samples
            .iter()
            .map(|s| {
                let i = ((s.theta - theta_start) / theta_step).round() as usize;
                let j = (!s.pole).then(|| {
                    (((s.phi - phi_start) / phi_step).round() as usize).rem_euclid(phi_count)
                });
                (i, j)
            })
            .collect::<Vec<_>>();
        let mut filled = vec![false; theta_count * phi_count];
        for (f_sorted, &f) in order.iter().enumerate() {
            filled.fill(false);
            for (k, (sample, &(i, j))) in samples.iter().zip(&indices).enumerate() {
                let (e_theta, e_phi) = fields[f][k];
                let value = to_cartesian(e_theta, e_phi, sample.theta_hat, sample.phi_hat)
                    .map(|c| Complex32::new(c.re as f32, c.im as f32));
                // a sample at a pole fills the whole pole row
                let columns = match j {
                    Some(j) => j..j + 1,
                    None => 0..phi_count,
                };
                for j in columns {
                    // the first sample of a grid point wins
                    if !filled[i * phi_count + j] {
                        filled[i * phi_count + j] = true;
                        let index = beam.index(f_sorted, i, j);
                        beam.field[index] = value;
                    }
                }
            }
        }
        if let Some(missing) = filled.iter().position(|f| !f) {
            let (i, j) = (missing / phi_count, missing % phi_count);
            return Err(format!(
                "the grid has no sample at θ = {:.3}°, φ = {:.3}°",
                beam.theta(i).to_degrees(),
                beam.phi(j).to_degrees()
            ));
        }

        beam.peak = (0..beam.frequencies.len())
            .map(|f| {
                beam.field[beam.index(f, 0, 0)..beam.index(f + 1, 0, 0)]
                    .iter()
                    .map(|v| v.iter().map(|c| c.norm_sqr() as f64).sum::<f64>())
                    .fold(0.0, f64::max)
            })
            .collect();
        if beam.peak.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err("the field must be finite and not all zero at every frequency".into());
        }
        Ok(beam)
    }

    /// Spherical field components `(E_θ, E_φ)` at a grid point, in the grid's own basis.
    pub(crate) fn spherical_at(&self, f: usize, i: usize, j: usize) -> (Complex64, Complex64) {
        let (theta_hat, phi_hat) = spherical_basis_at(self.theta(i), self.phi(j));
        let v = self.field[self.index(f, i, j)].map(|c| Complex64::new(c.re as f64, c.im as f64));
        (cdot_real(v, theta_hat), cdot_real(v, phi_hat))
    }

    /// Integrates `|E|²` over the tabulated part of the sphere, divided by 4π.
    fn integrated(&self, f: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.theta_count {
            // trapezoidal weights in θ
            let edge = i == 0 || i == self.theta_count - 1;
            let weight = self.theta(i).sin() * self.theta_step * if edge { 0.5 } else { 1.0 };
            let ring = (0..self.phi_count)
                .map(|j| {
                    self.field[self.index(f, i, j)]
                        .iter()
                        .map(|c| c.norm_sqr() as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                * self.phi_step();
            sum += weight * ring;
        }
        sum / (2.0 * TAU)
    }

    fn covers_sphere(&self) -> bool {
        let last = self.theta(self.theta_count - 1);
        self.theta_start.abs() < 1e-6 && (last - PI).abs() < 1e-6
    }
}

/// A far-field sample mapped to the direction it represents.
struct Sample {
    /// `θ ∈ [0, π]` of the direction.
    theta: f64,
    /// `φ ∈ [0, 2π)` of the direction, meaningless at a pole.
    phi: f64,
    pole: bool,
    /// Basis the sample's components are given in.
    theta_hat: Vec3,
    phi_hat: Vec3,
}

impl Sample {
    fn new(theta: f64, phi: f64) -> Self {
        let direction = direction_from_angles(theta, phi);
        let (theta_hat, phi_hat) = spherical_basis_at(theta, phi);
        let pole = direction.x.hypot(direction.y) < 1e-9;
        Sample {
            theta: direction.z.clamp(-1.0, 1.0).acos(),
            phi: direction.y.atan2(direction.x).rem_euclid(TAU),
            pole,
            theta_hat,
            phi_hat,
        }
    }
}

/// Sorts values and merges those closer than 1e-6 rad, also across the wrap of `2π`.
fn unique_sorted(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let mut unique: Vec<f64> = Vec::new();
    for v in values {
        if unique.last().is_none_or(|last| v - last > 1e-6) {
            unique.push(v);
        }
    }
    if unique.len() > 1 && unique[0] + TAU - unique[unique.len() - 1] < 1e-6 {
        unique.pop();
    }
    unique
}

/// Returns the step of regularly spaced sorted values.
fn regular_step(values: &[f64]) -> Option<f64> {
    let n = values.len();
    let step = (values[n - 1] - values[0]) / (n - 1) as f64;
    values
        .iter()
        .enumerate()
        .all(|(i, v)| (v - (values[0] + i as f64 * step)).abs() <= 1e-3 * step)
        .then_some(step)
}

/// Value of a beam in a direction, in the units of its normalization.
pub(crate) enum Pattern {
    /// Square root of the relative gain of a beam without polarization.
    Scalar(f64),
    /// Far-field vector, with `|E|²` the gain-like quantity of the beam.
    Vector(CVec3),
}

/// Antenna beam pattern, used by receivers and transmitters.
///
/// A beam is defined in its own frame and placed in the world frame by an `Orientation`
/// (receivers) or `Attitude` (calibrators). Beams are either analytic or tabulated far
/// fields, imported from openEMS or built from arrays, such as far fields exported by
/// other antenna simulation software.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Beam {
    pub(crate) model: BeamModel,
}

#[derive(Clone)]
pub(crate) enum BeamModel {
    /// Legacy beam: relative gain `cos²θ`, effective area `min(λ²/4, 4 m²)`, no polarization.
    CosSquared,
    /// Isotropic beam with a fixed polarization in the `θ̂`/`φ̂` basis, unit gain.
    Isotropic {
        e_theta: Complex64,
        e_phi: Complex64,
    },
    /// Short dipole along the beam's x axis, directivity 1.5.
    ShortDipole,
    Tabulated(Arc<TabulatedBeam>),
}

#[pymethods]
impl Beam {
    /// Creates a tabulated beam from far-field samples.
    ///
    /// # Arguments
    /// - `frequencies`: Frequencies (Hz).
    /// - `theta`: Values of `θ` (radians), measured from the beam's +z axis.
    /// - `phi`: Values of `φ` (radians), measured from the beam's +x axis towards +y.
    /// - `e_theta`: `E_θ` on the grid, flattened in the order `[frequency][θ][φ]`
    ///   (`numpy.ravel` of an array of shape `(len(frequencies), len(theta), len(phi))`).
    /// - `e_phi`: `E_φ`, in the same order.
    /// - `normalization`: What `|E_θ|² + |E_φ|²` represents: `"directivity"`, `"gain"`,
    ///   `"realized_gain"` or `"relative"`.
    /// - `effective_area`: Effective area (m²) per unit `|E|²` at each frequency, required
    ///   for relative beams.
    ///
    /// The grid may use any `θ`/`φ` convention as long as the directions it represents
    /// form a regular grid with `φ` around the full circle.
    ///
    /// # Panics
    /// Panics if the samples do not form a valid grid.
    #[new]
    #[pyo3(signature = (frequencies, theta, phi, e_theta, e_phi, normalization = "directivity", effective_area = None))]
    pub fn new(
        frequencies: Vec<f64>,
        theta: Vec<f64>,
        phi: Vec<f64>,
        e_theta: Vec<Complex64>,
        e_phi: Vec<Complex64>,
        normalization: &str,
        effective_area: Option<Vec<f64>>,
    ) -> Self {
        let normalization = Normalization::parse(normalization).unwrap_or_else(|e| panic!("{e}"));
        let n = frequencies.len() * theta.len() * phi.len();
        assert!(
            e_theta.len() == n && e_phi.len() == n,
            "e_theta and e_phi must have len(frequencies) * len(theta) * len(phi) = {n} values"
        );
        let angles = theta
            .iter()
            .flat_map(|&t| phi.iter().map(move |&p| (t, p)))
            .collect::<Vec<_>>();
        let per_frequency = angles.len();
        let fields = (0..frequencies.len())
            .map(|f| {
                (0..per_frequency)
                    .map(|k| (e_theta[f * per_frequency + k], e_phi[f * per_frequency + k]))
                    .collect()
            })
            .collect();
        let beam = TabulatedBeam::from_samples(
            frequencies,
            &angles,
            fields,
            normalization,
            effective_area,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        Beam {
            model: BeamModel::Tabulated(Arc::new(beam)),
        }
    }

    /// Returns the legacy beam: relative gain `cos²θ` with `θ` measured from the beam's
    /// +z axis, effective area `min(λ²/4, 4 m²)` and no polarization.
    #[staticmethod]
    pub fn cos_squared() -> Self {
        Beam {
            model: BeamModel::CosSquared,
        }
    }

    /// Returns an isotropic beam with unit gain and a fixed polarization.
    ///
    /// # Arguments
    /// - `e_theta`, `e_phi`: Polarization in the `θ̂`/`φ̂` basis of the beam's frame,
    ///   normalized to unit length. With the `exp(+iωt)` convention of fringe,
    ///   `(1, -1j)` is right-hand circular for a wave traveling outwards.
    ///
    /// # Panics
    /// Panics if both components are zero.
    #[staticmethod]
    #[pyo3(signature = (e_theta = Complex64::ONE, e_phi = Complex64::ZERO))]
    pub fn isotropic(e_theta: Complex64, e_phi: Complex64) -> Self {
        let norm = (e_theta.norm_sqr() + e_phi.norm_sqr()).sqrt();
        assert!(
            norm > 0.0 && norm.is_finite(),
            "polarization must be non-zero and finite"
        );
        Beam {
            model: BeamModel::Isotropic {
                e_theta: e_theta / norm,
                e_phi: e_phi / norm,
            },
        }
    }

    /// Returns the beam of a short dipole along the beam's +x axis, with directivity 1.5.
    ///
    /// With the identity orientation the dipole is horizontal and the beam's boresight
    /// points to the zenith.
    #[staticmethod]
    pub fn short_dipole() -> Self {
        Beam {
            model: BeamModel::ShortDipole,
        }
    }

    /// Loads a far field computed by openEMS (`CalcNF2FF`, HDF5 output).
    ///
    /// The field is normalized to directivity using the `Dmax` stored with it, or to
    /// realized gain if the incident power of the excited port is given.
    ///
    /// The phase of the far field computed by openEMS includes the phase of the excitation
    /// signal, which differs per frequency (the delay of the excitation pulse), and the
    /// propagation to the radius of the far field. The latter is always removed. The
    /// excitation is removed by dividing by its phase if it is given, which keeps the phase
    /// response (group delay) of the antenna. Otherwise, the phase at each frequency is
    /// referenced to the field in the direction of the peak gain, which loses it.
    ///
    /// # Arguments
    /// - `filepath`: Path to the nf2ff HDF5 file.
    /// - `incident_power`: Incident power (W) of the excited port at each frequency of
    ///   the file (`port.P_inc`), to normalize to realized gain.
    /// - `excitation`: Complex excitation at each frequency of the file, whose phase is
    ///   removed: the incident voltage of the port (`port.uf_inc`) for the response of
    ///   the antenna into the port's resistance, including its mismatch, which goes with
    ///   `incident_power`.
    ///
    /// # Errors
    /// Returns an `Error` (an `IOError` in Python) if the file cannot be read or does not
    /// hold a valid far field.
    #[staticmethod]
    #[pyo3(signature = (filepath, incident_power = None, excitation = None))]
    pub fn load_openems(
        filepath: &str,
        incident_power: Option<Vec<f64>>,
        excitation: Option<Vec<Complex64>>,
    ) -> Result<Beam, Error> {
        beam_io::load_openems(
            filepath.as_ref(),
            incident_power.as_deref(),
            excitation.as_deref(),
        )
        .map(Beam::tabulated)
        .map_err(|e| Error::new(format!("Failed to load '{filepath}': {e}")))
    }

    /// Returns the kind of beam: `"cos_squared"`, `"isotropic"`, `"short_dipole"` or `"tabulated"`.
    pub fn kind(&self) -> &'static str {
        match self.model {
            BeamModel::CosSquared => "cos_squared",
            BeamModel::Isotropic { .. } => "isotropic",
            BeamModel::ShortDipole => "short_dipole",
            BeamModel::Tabulated(_) => "tabulated",
        }
    }

    /// Returns what `|E|²` represents: `"relative"` for the cos² beam, `"directivity"`
    /// for the other analytic beams, or the normalization of a tabulated beam.
    pub fn normalization(&self) -> &'static str {
        match &self.model {
            BeamModel::CosSquared => "relative",
            BeamModel::Isotropic { .. } | BeamModel::ShortDipole => "directivity",
            BeamModel::Tabulated(t) => t.normalization.name(),
        }
    }

    /// Returns the frequencies (Hz) of a tabulated beam, or `None` for an analytic beam.
    pub fn frequencies(&self) -> Option<Vec<f64>> {
        match &self.model {
            BeamModel::Tabulated(t) => Some(t.frequencies.clone()),
            _ => None,
        }
    }

    /// Returns `|E|²` in a direction of the beam's frame: the gain, directivity or
    /// relative gain, depending on the normalization.
    ///
    /// # Arguments
    /// - `direction`: Direction in the beam's frame.
    /// - `frequency`: Frequency (Hz).
    pub fn gain(&self, direction: Vec3, frequency: f64) -> f64 {
        match self.pattern(direction.normalized(), frequency) {
            Pattern::Scalar(s) => s * s,
            Pattern::Vector(v) => cnorm(v).powi(2),
        }
    }

    /// Returns the far field `(E_θ, E_φ)` in a direction of the beam's frame.
    ///
    /// For the cos² beam, which has no polarization, this is `(√gain, 0)`.
    pub fn field(&self, direction: Vec3, frequency: f64) -> (Complex64, Complex64) {
        let direction = direction.normalized();
        match self.pattern(direction, frequency) {
            Pattern::Scalar(s) => (Complex64::new(s, 0.0), Complex64::ZERO),
            Pattern::Vector(v) => {
                let (theta_hat, phi_hat) = spherical_basis(direction);
                (cdot_real(v, theta_hat), cdot_real(v, phi_hat))
            }
        }
    }

    /// Returns the effective area (m²) in a direction of the beam's frame.
    ///
    /// # Panics
    /// Panics if `frequency` is not positive.
    pub fn effective_area(&self, direction: Vec3, frequency: f64) -> f64 {
        assert!(frequency > 0.0, "frequency must be positive");
        self.gain(direction, frequency) * self.area_scale(frequency)
    }

    /// Returns a summary of the beam, to check an imported beam: its grid, frequencies,
    /// normalization, and per frequency the peak gain and the integrated gain over 4π,
    /// which is about 1 for directivity and the radiation efficiency for gain.
    pub fn diagnostics(&self) -> String {
        let BeamModel::Tabulated(t) = &self.model else {
            return format!("Analytic beam: {}", self.__repr__());
        };
        let mut out = String::new();
        let _ = writeln!(out, "Tabulated beam ({})", t.normalization.name());
        let _ = writeln!(
            out,
            "  θ: {:.3}° to {:.3}° in steps of {:.3}° ({} values)",
            t.theta_start.to_degrees(),
            t.theta(t.theta_count - 1).to_degrees(),
            t.theta_step.to_degrees(),
            t.theta_count
        );
        let _ = writeln!(
            out,
            "  φ: {:.3}° to {:.3}° in steps of {:.3}° ({} values)",
            t.phi_start.to_degrees(),
            t.phi(t.phi_count - 1).to_degrees(),
            t.phi_step().to_degrees(),
            t.phi_count
        );
        let _ = writeln!(
            out,
            "  frequencies: {:.6} MHz to {:.6} MHz ({} values)",
            t.frequencies[0] / 1e6,
            t.frequencies[t.frequencies.len() - 1] / 1e6,
            t.frequencies.len()
        );
        if !t.covers_sphere() {
            let _ = writeln!(
                out,
                "  the field is zero outside θ {:.1}° to {:.1}°, the integrated gain covers only that range",
                t.theta_start.to_degrees(),
                t.theta(t.theta_count - 1).to_degrees()
            );
        }
        let unit = if t.normalization == Normalization::Relative {
            "dB"
        } else {
            "dBi"
        };
        let _ = writeln!(
            out,
            "  {:>14}  {:>10}  {:>16}",
            "frequency", "peak", "integrated / 4π"
        );
        for (f, frequency) in t.frequencies.iter().enumerate() {
            let _ = writeln!(
                out,
                "  {:>10.4} MHz  {:>6.2} {unit:<3}  {:>16.4}",
                frequency / 1e6,
                10.0 * t.peak[f].log10(),
                t.integrated(f)
            );
        }
        out
    }

    pub fn __repr__(&self) -> String {
        match &self.model {
            BeamModel::CosSquared => "Beam.cos_squared()".into(),
            BeamModel::Isotropic { e_theta, e_phi } => {
                format!("Beam.isotropic(e_theta = {e_theta}, e_phi = {e_phi})")
            }
            BeamModel::ShortDipole => "Beam.short_dipole()".into(),
            BeamModel::Tabulated(t) => format!(
                "Beam(tabulated, {}, {} frequencies from {:.2} MHz to {:.2} MHz, {} x {} directions)",
                t.normalization.name(),
                t.frequencies.len(),
                t.frequencies[0] / 1e6,
                t.frequencies[t.frequencies.len() - 1] / 1e6,
                t.theta_count,
                t.phi_count
            ),
        }
    }
}

impl Beam {
    pub(crate) fn tabulated(beam: TabulatedBeam) -> Self {
        Beam {
            model: BeamModel::Tabulated(Arc::new(beam)),
        }
    }

    /// Value of the beam in a direction (unit vector in the beam's frame) at a frequency.
    pub(crate) fn pattern(&self, direction: Vec3, frequency: f64) -> Pattern {
        match &self.model {
            BeamModel::CosSquared => Pattern::Scalar(direction.z.abs()),
            BeamModel::Isotropic { e_theta, e_phi } => {
                let (theta_hat, phi_hat) = spherical_basis(direction);
                Pattern::Vector(to_cartesian(*e_theta, *e_phi, theta_hat, phi_hat))
            }
            BeamModel::ShortDipole => {
                // the component of the dipole axis perpendicular to the direction
                let axis = Vec3::new(1.0, 0.0, 0.0);
                let perpendicular = axis - direction * direction.x;
                let v = perpendicular * SHORT_DIPOLE_DIRECTIVITY.sqrt();
                Pattern::Vector([v.x, v.y, v.z].map(|c| Complex64::new(c, 0.0)))
            }
            BeamModel::Tabulated(t) => Pattern::Vector(t.field_at(direction, frequency)),
        }
    }

    /// Effective area (m²) per unit `|E|²` at a frequency.
    pub(crate) fn area_scale(&self, frequency: f64) -> f64 {
        let lambda = C / frequency;
        let isotropic = lambda * lambda / (2.0 * TAU);
        match &self.model {
            BeamModel::CosSquared => (lambda * lambda / 4.0).min(4.0),
            BeamModel::Isotropic { .. } | BeamModel::ShortDipole => isotropic,
            BeamModel::Tabulated(t) => match &t.effective_area {
                Some(area) => t.interpolate_frequency(area, frequency),
                None => isotropic,
            },
        }
    }

    /// Peak effective area (m²) at a frequency.
    pub(crate) fn peak_effective_area(&self, frequency: f64) -> f64 {
        let peak = match &self.model {
            BeamModel::CosSquared | BeamModel::Isotropic { .. } => 1.0,
            BeamModel::ShortDipole => SHORT_DIPOLE_DIRECTIVITY,
            BeamModel::Tabulated(t) => t.interpolate_frequency(&t.peak, frequency),
        };
        peak * self.area_scale(frequency)
    }

    /// Whether `|E|²` is an absolute gain, as required for transmitters.
    pub(crate) fn is_absolute(&self) -> bool {
        match &self.model {
            BeamModel::CosSquared => false,
            BeamModel::Isotropic { .. } | BeamModel::ShortDipole => true,
            BeamModel::Tabulated(t) => t.normalization != Normalization::Relative,
        }
    }

    /// Whether the beam has no polarization.
    pub(crate) fn is_scalar(&self) -> bool {
        matches!(self.model, BeamModel::CosSquared)
    }

    pub(crate) fn as_tabulated(&self) -> Option<&Arc<TabulatedBeam>> {
        match &self.model {
            BeamModel::Tabulated(t) => Some(t),
            _ => None,
        }
    }

    /// Whether two beams are the same beam.
    pub(crate) fn same(&self, other: &Beam) -> bool {
        match (&self.model, &other.model) {
            (BeamModel::CosSquared, BeamModel::CosSquared)
            | (BeamModel::ShortDipole, BeamModel::ShortDipole) => true,
            (
                BeamModel::Isotropic { e_theta, e_phi },
                BeamModel::Isotropic {
                    e_theta: t,
                    e_phi: p,
                },
            ) => e_theta == t && e_phi == p,
            (BeamModel::Tabulated(a), BeamModel::Tabulated(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Checks that a tabulated beam covers a frequency range.
    ///
    /// # Panics
    /// Panics with a message naming `what` if it does not.
    pub(crate) fn assert_covers(&self, fmin: f64, fmax: f64, what: &str) {
        if let BeamModel::Tabulated(t) = &self.model {
            let (first, last) = (t.frequencies[0], t.frequencies[t.frequencies.len() - 1]);
            let tolerance = 1e-9 * fmax;
            assert!(
                first <= fmin + tolerance && last >= fmax - tolerance,
                "the beam of {what} covers {:.6} MHz to {:.6} MHz, but must cover {:.6} MHz to {:.6} MHz",
                first / 1e6,
                last / 1e6,
                fmin / 1e6,
                fmax / 1e6
            );
        }
    }
}

impl Default for Beam {
    fn default() -> Self {
        Beam::cos_squared()
    }
}

impl std::fmt::Debug for Beam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.__repr__())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Far field `(E_θ, E_φ)` of a vector beam at `θ`, `φ`, in the spherical basis at those
    /// angles (which differs from the canonical basis for `θ < 0`).
    pub(crate) fn sample_field(
        beam: &Beam,
        theta: f64,
        phi: f64,
        frequency: f64,
    ) -> (Complex64, Complex64) {
        let Pattern::Vector(v) = beam.pattern(direction_from_angles(theta, phi), frequency) else {
            panic!("vector beam expected");
        };
        let (theta_hat, phi_hat) = spherical_basis_at(theta, phi);
        (cdot_real(v, theta_hat), cdot_real(v, phi_hat))
    }

    /// Tabulates an analytic beam on a regular grid.
    pub(crate) fn tabulate(
        beam: &Beam,
        frequencies: &[f64],
        theta_step_deg: f64,
        phi_step_deg: f64,
    ) -> Beam {
        let theta = (0..=(180.0 / theta_step_deg).round() as usize)
            .map(|i| (i as f64 * theta_step_deg).to_radians())
            .collect::<Vec<_>>();
        let phi = (0..(360.0 / phi_step_deg).round() as usize)
            .map(|j| (j as f64 * phi_step_deg).to_radians())
            .collect::<Vec<_>>();
        let (mut e_theta, mut e_phi) = (Vec::new(), Vec::new());
        for &f in frequencies {
            for &t in &theta {
                for &p in &phi {
                    let (a, b) = sample_field(beam, t, p, f);
                    e_theta.push(a);
                    e_phi.push(b);
                }
            }
        }
        Beam::new(
            frequencies.to_vec(),
            theta,
            phi,
            e_theta,
            e_phi,
            "directivity",
            None,
        )
    }

    fn random_directions(n: usize) -> Vec<Vec3> {
        // deterministic quasi-random directions
        (0..n)
            .map(|i| {
                let u = (i as f64 * 0.618034).fract() * 2.0 - 1.0;
                let p = (i as f64 * 0.754878).fract() * TAU;
                let s = (1.0 - u * u).sqrt();
                Vec3::new(s * p.cos(), s * p.sin(), u)
            })
            .collect()
    }

    #[test]
    fn orientation_is_orthonormal() {
        let o = Orientation::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0));
        for (a, b) in [(o.x, o.y), (o.y, o.z), (o.z, o.x)] {
            assert!(a.dot(b).abs() < 1e-12);
        }
        assert!((o.x.cross(o.y) - o.z).norm() < 1e-12);
        let v = Vec3::new(0.3, -1.2, 0.7);
        assert!((o.to_world(o.to_local(v)) - v).norm() < 1e-12);
        let r = o.rotated(TAU / 4.0);
        assert!((r.x - o.y).norm() < 1e-12);
    }

    #[test]
    fn short_dipole_directivity() {
        let beam = Beam::short_dipole();
        // broadside and end-fire
        assert!((beam.gain(Vec3::new(0.0, 0.0, 1.0), 50e6) - 1.5).abs() < 1e-12);
        assert!(beam.gain(Vec3::new(1.0, 0.0, 0.0), 50e6) < 1e-12);
        let t = tabulate(&beam, &[50e6], 2.0, 2.0);
        let BeamModel::Tabulated(t) = &t.model else {
            unreachable!()
        };
        assert!((t.integrated(0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn tabulated_matches_analytic() {
        let beam = Beam::short_dipole();
        let tabulated = tabulate(&beam, &[40e6, 60e6], 1.0, 1.0);
        for direction in random_directions(500) {
            let (a, b) = (
                beam.field(direction, 50e6),
                tabulated.field(direction, 50e6),
            );
            let error = (a.0 - b.0).norm() + (a.1 - b.1).norm();
            assert!(error < 1e-3, "{direction:?}: {a:?} vs {b:?}");
        }
    }

    /// Field of crossed dipoles along x and y, fed in quadrature: smooth, complex and
    /// polarized differently in every direction.
    fn turnstile(direction: Vec3) -> CVec3 {
        let perpendicular = |axis: Vec3| axis - direction * direction.dot(axis);
        let x = perpendicular(Vec3::new(1.0, 0.0, 0.0));
        let y = perpendicular(Vec3::new(0.0, 1.0, 0.0));
        [
            Complex64::new(x.x, y.x),
            Complex64::new(x.y, y.y),
            Complex64::new(x.z, y.z),
        ]
    }

    /// Tabulates the turnstile on a grid of `θ` and `φ` (degrees).
    fn tabulate_turnstile(theta: &[f64], phi: &[f64]) -> Beam {
        let (mut e_theta, mut e_phi) = (Vec::new(), Vec::new());
        for &t in theta {
            for &p in phi {
                let (t, p) = (t.to_radians(), p.to_radians());
                let v = turnstile(direction_from_angles(t, p));
                let (theta_hat, phi_hat) = spherical_basis_at(t, p);
                e_theta.push(cdot_real(v, theta_hat));
                e_phi.push(cdot_real(v, phi_hat));
            }
        }
        let radians = |a: &[f64]| a.iter().map(|x| x.to_radians()).collect();
        Beam::new(
            vec![1e6],
            radians(theta),
            radians(phi),
            e_theta,
            e_phi,
            "directivity",
            None,
        )
    }

    #[test]
    fn grid_conventions_are_equivalent() {
        // θ ∈ [-180°, 180°) with φ ∈ [0°, 180°) describes the same sphere as θ ∈ [0°, 180°]
        // with φ ∈ [0°, 360°)
        let openems = tabulate_turnstile(
            &(0..36).map(|i| i as f64 * 10.0 - 180.0).collect::<Vec<_>>(),
            &(0..18).map(|j| j as f64 * 10.0).collect::<Vec<_>>(),
        );
        let BeamModel::Tabulated(t) = &openems.model else {
            unreachable!()
        };
        assert_eq!((t.theta_count, t.phi_count), (19, 36));
        let standard = tabulate_turnstile(
            &(0..=18).map(|i| i as f64 * 10.0).collect::<Vec<_>>(),
            &(0..37).map(|j| j as f64 * 10.0).collect::<Vec<_>>(),
        );
        for direction in random_directions(200)
            .into_iter()
            .chain([Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)])
        {
            let Pattern::Vector(exact) = standard.pattern(direction, 1e6) else {
                unreachable!()
            };
            let Pattern::Vector(other) = openems.pattern(direction, 1e6) else {
                unreachable!()
            };
            let reference = turnstile(direction);
            for v in [exact, other] {
                let error = cnorm([0, 1, 2].map(|i| v[i] - reference[i]));
                assert!(error < 3e-2, "{direction:?}: {v:?} vs {reference:?}");
            }
        }
    }

    #[test]
    fn rejects_incomplete_grid() {
        // φ = 240° is missing at θ = 1 rad
        let error = TabulatedBeam::from_samples(
            vec![1e6],
            &[
                (0.5, 0.0),
                (0.5, TAU / 3.0),
                (0.5, 2.0 * TAU / 3.0),
                (1.0, 0.0),
                (1.0, TAU / 3.0),
            ],
            vec![vec![(Complex64::ONE, Complex64::ZERO); 5]],
            Normalization::Directivity,
            None,
        )
        .err()
        .unwrap();
        assert!(
            error.contains("no sample at θ = 57.296°, φ = 240.000°"),
            "{error}"
        );
    }

    #[test]
    fn rejects_partial_phi_coverage() {
        // φ only covers a quarter circle
        let angles = [0.5, 1.0]
            .iter()
            .flat_map(|&t| [0.0, 0.5, 1.0].map(|p| (t, p)))
            .collect::<Vec<_>>();
        let error = TabulatedBeam::from_samples(
            vec![1e6],
            &angles,
            vec![vec![(Complex64::ONE, Complex64::ZERO); 6]],
            Normalization::Directivity,
            None,
        )
        .err()
        .unwrap();
        assert!(error.contains("full circle"), "{error}");
    }
}
