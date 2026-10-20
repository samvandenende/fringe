use crate::Vec3;
use crate::library::beam::{Attitude, Beam, Orientation};
use num_complex::Complex32;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use std::sync::Arc;

/// Speed of light in vacuum (m/s).
const C: f64 = 299792458.0;
/// Boltzmann constant (J/K).
const K_B: f64 = 1.380649e-23;
/// Noise power spectral density `k_B * T` of one kelvin, in Jy m² (1 Jy m² = 1e-26 W/Hz).
pub(crate) const JY_M2_PER_KELVIN: f64 = K_B * 1e26;

/// System noise temperature (K) of the receivers, constant or tabulated over frequency.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, FromPyObject)]
#[serde(untagged)]
pub enum NoiseTemperature {
    Constant(f64),
    /// `[frequency (Hz), temperature (K)]` pairs with increasing frequencies, interpolated
    /// linearly in frequency and held constant beyond the first and last frequency.
    Table(Vec<[f64; 2]>),
}

/// A temperature (K) as a float, a table as a list of `(frequency (Hz), temperature (K))`.
impl<'py> IntoPyObject<'py> for NoiseTemperature {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            NoiseTemperature::Constant(t) => t.into_bound_py_any(py),
            NoiseTemperature::Table(table) => table
                .into_iter()
                .map(|[f, t]| (f, t))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
        }
    }
}

impl NoiseTemperature {
    /// Placeholder for array files that give no noise temperature.
    fn missing() -> Self {
        NoiseTemperature::Constant(f64::NAN)
    }

    pub(crate) fn is_missing(&self) -> bool {
        matches!(self, NoiseTemperature::Constant(t) if t.is_nan())
    }

    /// The temperature (K) at a frequency (Hz).
    pub(crate) fn at(&self, frequency: f64) -> f64 {
        match self {
            NoiseTemperature::Constant(t) => *t,
            NoiseTemperature::Table(table) => {
                let i = table.partition_point(|&[f, _]| f <= frequency);
                if i == 0 {
                    return table[0][1];
                }
                if i == table.len() {
                    return table[i - 1][1];
                }
                let ([f0, t0], [f1, t1]) = (table[i - 1], table[i]);
                t0 + (t1 - t0) * (frequency - f0) / (f1 - f0)
            }
        }
    }

    fn validate(&self, [fmin, fmax]: [f64; 2]) {
        match self {
            NoiseTemperature::Constant(t) => assert!(
                t.is_finite() && *t >= 0.0,
                "system_noise_temperature must be non-negative"
            ),
            NoiseTemperature::Table(table) => {
                assert!(
                    !table.is_empty(),
                    "system_noise_temperature table must not be empty"
                );
                assert!(
                    table
                        .iter()
                        .all(|&[f, t]| f.is_finite() && f > 0.0 && t.is_finite() && t >= 0.0),
                    "system_noise_temperature table must have positive frequencies and non-negative temperatures"
                );
                assert!(
                    table.windows(2).all(|w| w[0][0] < w[1][0]),
                    "system_noise_temperature table frequencies must be increasing"
                );
                let (first, last) = (table[0][0], table[table.len() - 1][0]);
                let tolerance = 1e-9 * fmax;
                assert!(
                    first <= fmin + tolerance && last >= fmax - tolerance,
                    "the system_noise_temperature table covers {:.6} MHz to {:.6} MHz, but must cover {:.6} MHz to {:.6} MHz",
                    first / 1e6,
                    last / 1e6,
                    fmin / 1e6,
                    fmax / 1e6
                );
            }
        }
    }
}

/// Array model and its signal acquisition parameters.
///
/// This structure defines:
/// - array geometry
/// - sampling configuration
/// - frequency conversion parameters
/// - simulation noise characteristics
///
/// Every antenna (receiver) has a beam and an orientation that places the beam in the
/// world frame. A receiver is one polarization channel: a dual-polarized antenna is two
/// receivers at the same position, see `Array::dual_polarized`.
///
/// The output samples are in √Jy, relative to the reference area `A_ref(f)` of the array,
/// the largest peak effective area of its beams: a source of flux density `S` received with
/// effective area `A_e` has a mean power of `S * A_e / A_ref`, averaged over the sample
/// frequency. A single polarization receives half the flux density of an unpolarized source.
#[pyclass(from_py_object)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Array {
    pub(crate) sample_frequency: f64,
    pub(crate) downmix_frequency: f64,
    pub(crate) bandpass: [f64; 2],
    pub(crate) sample_window_size: usize,
    #[serde(default = "NoiseTemperature::missing")]
    pub(crate) system_noise_temperature: NoiseTemperature,
    pub(crate) antenna_positions: Vec<Vec3>,
    /// Beams of the array, written to and read from files by `save_array` and `load_array`.
    #[serde(skip, default = "default_beams")]
    pub(crate) beams: Vec<Beam>,
    /// Index into `beams` per antenna, all 0 if empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) antenna_beams: Vec<usize>,
    /// Orientation per antenna, all the identity if empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) antenna_orientations: Vec<Orientation>,
    /// Up direction of the local horizon, below which nothing is received.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) horizon: Option<Vec3>,
}

fn default_beams() -> Vec<Beam> {
    vec![Beam::cos_squared()]
}

#[pymethods]
impl Array {
    /// Creates a new antenna array configuration.
    ///
    /// # Arguments
    /// - `antenna_positions`: Positions of antennas in the array.
    /// - `sample_frequency`: ADC sampling frequency (Hz).
    /// - `downmix_frequency`: Frequency used for downconversion (Hz).
    /// - `bandpass_fmin`: Lower cutoff frequency of the bandpass filter (Hz).
    /// - `bandpass_fmax`: Upper cutoff frequency of the bandpass filter (Hz).
    /// - `sample_window_size`: Number of samples per FFT window (must be power of two).
    /// - `system_noise_temperature`: System noise temperature (K) of every receiver, as a
    ///   number, or as `(frequency (Hz), temperature (K))` pairs with increasing
    ///   frequencies that cover the bandpass, interpolated linearly in frequency. A
    ///   receiver's noise power spectral density is `k_B * T`, which is received as a flux
    ///   density of `k_B * T / A_ref`, with `A_ref` the reference area of the array. The sky
    ///   is simulated from the sources, so `T` should not include the sky's contribution.
    /// - `beams`: Beams of the antennas, `[Beam.cos_squared()]` if not given.
    /// - `antenna_beams`: Index into `beams` per antenna, all 0 if not given.
    /// - `antenna_orientations`: Orientation of the beam per antenna, all the identity
    ///   (boresight at the zenith, +z) if not given.
    /// - `horizon`: Up direction of the local horizon. If given, nothing is received from
    ///   below the horizon.
    ///
    /// # Panics
    /// Panics if:
    /// - antenna_positions is empty
    /// - sample_frequency is not positive
    /// - downmix_frequency is negative
    /// - bandpass bounds are invalid or exceed Nyquist limit
    /// - sample_window_size is not a power of two
    /// - system_noise_temperature is negative, or a table that is not increasing in
    ///   frequency or does not cover the bandpass
    /// - beams is empty, or a tabulated beam does not cover the bandpass
    /// - antenna_beams or antenna_orientations does not have a value per antenna, or
    ///   antenna_beams refers to a beam that does not exist
    /// - horizon is zero or not finite
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (antenna_positions, sample_frequency, downmix_frequency, bandpass_fmin, bandpass_fmax, sample_window_size, system_noise_temperature, beams = None, antenna_beams = None, antenna_orientations = None, horizon = None))]
    pub fn new(
        antenna_positions: Vec<Vec3>,
        sample_frequency: f64,
        downmix_frequency: f64,
        bandpass_fmin: f64,
        bandpass_fmax: f64,
        sample_window_size: usize,
        system_noise_temperature: NoiseTemperature,
        beams: Option<Vec<Beam>>,
        antenna_beams: Option<Vec<usize>>,
        antenna_orientations: Option<Vec<Orientation>>,
        horizon: Option<Vec3>,
    ) -> Self {
        let array = Array {
            antenna_positions,
            sample_frequency,
            downmix_frequency,
            bandpass: [bandpass_fmin, bandpass_fmax],
            sample_window_size,
            system_noise_temperature,
            beams: beams.unwrap_or_else(default_beams),
            antenna_beams: antenna_beams.unwrap_or_default(),
            antenna_orientations: antenna_orientations.unwrap_or_default(),
            horizon: horizon.map(|h| h.normalized()),
        };
        array.validate();
        array
    }

    /// Creates an array of dual-polarized antennas.
    ///
    /// Every antenna is two receivers at the same position, with the same beam rotated by
    /// 90° about its boresight: receiver `2 i` is the X polarization of antenna `i`, with
    /// the antenna's orientation, and receiver `2 i + 1` its Y polarization.
    ///
    /// # Arguments
    /// - `antenna_positions` up to `system_noise_temperature`: As for `Array(...)`.
    /// - `beam`: Beam of one polarization, `Beam.short_dipole()` if not given.
    /// - `orientations`: Orientation of the X polarization per antenna, the identity if
    ///   not given, which with the short dipole puts X along +x and Y along +y.
    /// - `horizon`: As for `Array(...)`.
    ///
    /// # Panics
    /// Panics under the same conditions as `Array(...)`.
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (antenna_positions, sample_frequency, downmix_frequency, bandpass_fmin, bandpass_fmax, sample_window_size, system_noise_temperature, beam = None, orientations = None, horizon = None))]
    pub fn dual_polarized(
        antenna_positions: Vec<Vec3>,
        sample_frequency: f64,
        downmix_frequency: f64,
        bandpass_fmin: f64,
        bandpass_fmax: f64,
        sample_window_size: usize,
        system_noise_temperature: NoiseTemperature,
        beam: Option<Beam>,
        orientations: Option<Vec<Orientation>>,
        horizon: Option<Vec3>,
    ) -> Self {
        let orientations =
            orientations.unwrap_or_else(|| vec![Orientation::identity(); antenna_positions.len()]);
        assert!(
            orientations.len() == antenna_positions.len(),
            "orientations must have a value per antenna"
        );
        let array = Array {
            antenna_positions: antenna_positions.iter().flat_map(|&p| [p, p]).collect(),
            sample_frequency,
            downmix_frequency,
            bandpass: [bandpass_fmin, bandpass_fmax],
            sample_window_size,
            system_noise_temperature,
            beams: vec![beam.unwrap_or_else(Beam::short_dipole)],
            antenna_beams: Vec::new(),
            antenna_orientations: orientations
                .iter()
                .flat_map(|o| [*o, o.rotated(std::f64::consts::FRAC_PI_2)])
                .collect(),
            horizon: horizon.map(|h| h.normalized()),
        };
        array.validate();
        array
    }

    /// Returns the sampling frequency in Hz.
    pub fn sample_frequency(&self) -> f64 {
        self.sample_frequency
    }

    /// Returns the downmix frequency in Hz.
    pub fn downmix_frequency(&self) -> f64 {
        self.downmix_frequency
    }

    /// Returns the lower bound of the bandpass filter (f_min) in Hz.
    pub fn bandpass_fmin(&self) -> f64 {
        self.bandpass[0]
    }

    /// Returns the upper bound of the bandpass filter (f_max) in Hz.
    pub fn bandpass_fmax(&self) -> f64 {
        self.bandpass[1]
    }

    /// Returns the sample window size.
    pub fn sample_window_size(&self) -> usize {
        self.sample_window_size
    }

    /// Returns the system noise temperature as given: a temperature (K), or a list of
    /// `(frequency (Hz), temperature (K))` pairs.
    pub fn system_noise_temperature(&self) -> NoiseTemperature {
        self.system_noise_temperature.clone()
    }

    /// Returns the system noise temperature (K) at a frequency (Hz), interpolated in a
    /// table and held constant beyond its first and last frequency.
    pub fn system_noise_temperature_at(&self, frequency: f64) -> f64 {
        self.system_noise_temperature.at(frequency)
    }

    /// Returns the antenna positions.
    pub fn antenna_positions(&self) -> Vec<Vec3> {
        self.antenna_positions.clone()
    }

    /// Returns the beams of the array.
    pub fn beams(&self) -> Vec<Beam> {
        self.beams.clone()
    }

    /// Returns the index into `beams` of every antenna.
    pub fn antenna_beams(&self) -> Vec<usize> {
        (0..self.antenna_positions.len())
            .map(|a| self.antenna_beam(a))
            .collect()
    }

    /// Returns the orientation of every antenna.
    pub fn antenna_orientations(&self) -> Vec<Orientation> {
        (0..self.antenna_positions.len())
            .map(|a| self.antenna_orientation(a))
            .collect()
    }

    /// Returns the up direction of the local horizon, if any.
    pub fn horizon(&self) -> Option<Vec3> {
        self.horizon
    }

    /// Returns the reference area `A_ref` (m²) at a frequency (Hz): the largest peak
    /// effective area of the beams used by the antennas. Output samples are in √Jy
    /// relative to this area.
    ///
    /// # Panics
    /// Panics if `frequency` is not positive.
    pub fn reference_area(&self, frequency: f64) -> f64 {
        assert!(frequency > 0.0, "frequency must be positive");
        self.used_beams()
            .iter()
            .map(|&b| self.beams[b].peak_effective_area(frequency))
            .fold(0.0, f64::max)
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(
    sample_frequency = {:.2}MHz,
    downmix_frequency = {:.2}MHz,
    bandpass = ({:.2} - {:.2})MHz,
    sample_window_size = {},
    system_noise_temperature = {},
    antenna_positions = [...{} items...],
    beams = [{}],
    horizon = {}
)",
            self.sample_frequency / 1e6,
            self.downmix_frequency / 1e6,
            self.bandpass[0] / 1e6,
            self.bandpass[1] / 1e6,
            self.sample_window_size,
            match &self.system_noise_temperature {
                NoiseTemperature::Constant(t) => format!("{t:.2}K"),
                NoiseTemperature::Table(table) => format!(
                    "[...{} points, {:.2} - {:.2}K...]",
                    table.len(),
                    table.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                    table.iter().map(|p| p[1]).fold(0.0, f64::max)
                ),
            },
            self.antenna_positions.len(),
            self.beams
                .iter()
                .map(|b| format!("{b:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            self.horizon.map_or("None".into(), |h| h.__repr__())
        )
    }

    pub(crate) fn validate(&self) {
        assert!(
            !self.antenna_positions.is_empty(),
            "antenna_positions must contain at least one antenna position"
        );
        assert!(
            self.sample_frequency.is_finite() && self.sample_frequency > 0.0,
            "sample_frequency must be positive"
        );
        assert!(
            self.downmix_frequency.is_finite() && self.downmix_frequency >= 0.0,
            "downmix_frequency must be non-negative"
        );
        assert!(
            self.bandpass[0].is_finite()
                && self.bandpass[0] > 0.0
                && self.bandpass[0] < self.bandpass[1],
            "bandpass_fmin must be positive and smaller than bandpass_fmax"
        );
        assert!(
            self.bandpass[1].is_finite() && self.bandpass[1] <= self.sample_frequency / 2.0,
            "bandpass_fmax must be at most half the sample frequency"
        );
        assert!(
            self.sample_window_size.is_power_of_two(),
            "Sample window size must be a power of 2"
        );
        self.system_noise_temperature.validate(self.physical_band());
        let n = self.antenna_positions.len();
        assert!(
            !self.beams.is_empty(),
            "beams must contain at least one beam"
        );
        assert!(
            self.antenna_beams.is_empty() || self.antenna_beams.len() == n,
            "antenna_beams must have a value per antenna"
        );
        assert!(
            self.antenna_beams.iter().all(|&b| b < self.beams.len()),
            "antenna_beams refers to a beam that does not exist"
        );
        assert!(
            self.antenna_orientations.is_empty() || self.antenna_orientations.len() == n,
            "antenna_orientations must have a value per antenna"
        );
        if let Some(h) = self.horizon {
            assert!(
                h.x.is_finite() && h.y.is_finite() && h.z.is_finite() && h.norm() > 0.0,
                "horizon must be non-zero and finite"
            );
        }
        let [fmin, fmax] = self.physical_band();
        for &b in &self.used_beams() {
            self.beams[b].assert_covers(fmin, fmax, &format!("antenna beam {b}"));
        }
    }
}

impl Array {
    /// Index into `beams` of an antenna.
    pub(crate) fn antenna_beam(&self, antenna: usize) -> usize {
        self.antenna_beams.get(antenna).copied().unwrap_or(0)
    }

    /// Orientation of an antenna.
    pub(crate) fn antenna_orientation(&self, antenna: usize) -> Orientation {
        self.antenna_orientations
            .get(antenna)
            .copied()
            .unwrap_or_default()
    }

    /// Indices of the beams used by at least one antenna, ascending.
    pub(crate) fn used_beams(&self) -> Vec<usize> {
        if self.antenna_beams.is_empty() {
            return vec![0];
        }
        let mut used = self.antenna_beams.clone();
        used.sort_unstable();
        used.dedup();
        used
    }

    /// Noise power spectral density `k_B * T` (Jy m²) of every receiver at a frequency (Hz).
    pub(crate) fn system_noise_intensity(&self, frequency: f64) -> f64 {
        JY_M2_PER_KELVIN * self.system_noise_temperature.at(frequency)
    }

    /// The bandpass in physical (not downmixed) frequencies (Hz).
    pub(crate) fn physical_band(&self) -> [f64; 2] {
        self.bandpass.map(|f| f + self.downmix_frequency)
    }
}

/// Source in the simulated sky model.
///
/// Each source emits a frequency-dependent signal characterized by a reference
/// flux density (Jy) and up to third-order spectral indices, and is located at a
/// fixed direction vector. The flux density at frequency `f` is
///
/// `I(f) = I0 * exp(a1 * x + a2 * x^2 + a3 * x^3)`, with `x = ln(f / f0)`,
///
/// which is a power law with spectral index `a1` when `a2 = a3 = 0`.
///
/// A source may be polarized. Its Stokes parameters at frequency `f` are
///
/// `Q = p I cos 2χ`, `U = p I sin 2χ`, `V = v I`, with `χ = χ0 + RM λ²`,
///
/// with `p` the linear polarization fraction, `χ0` the polarization angle, `RM` the
/// rotation measure and `v` the circular polarization fraction. Polarization is given in
/// the basis of north (towards +z, increasing declination) and east (increasing right
/// ascension) on the sky, with `χ` measured from north through east. The coherency of the
/// field `(E_north, E_east)` is `½ [[I + Q, U + iV], [U - iV, I - Q]]`.
#[pyclass(from_py_object)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Source {
    pub(crate) direction: Vec3,
    pub(crate) reference_frequency: f64,
    pub(crate) reference_intensity: f64,
    pub(crate) spectral_index: f64,
    #[serde(default)]
    pub(crate) spectral_index_2: f64,
    #[serde(default)]
    pub(crate) spectral_index_3: f64,
    #[serde(default)]
    pub(crate) polarization_fraction: f64,
    #[serde(default)]
    pub(crate) polarization_angle: f64,
    #[serde(default)]
    pub(crate) circular_fraction: f64,
    #[serde(default)]
    pub(crate) rotation_measure: f64,
}

#[pymethods]
impl Source {
    /// Creates a new signal source.
    ///
    /// # Arguments
    /// - `direction`: Unit vector indicating source direction in space.
    /// - `reference_frequency`: Frequency (Hz) at which the intensity is defined.
    /// - `reference_intensity`: Flux density (Jy) at the reference frequency.
    /// - `spectral_index`: First-order (power-law) spectral index of the source.
    /// - `spectral_index_2`: Second-order spectral index (curvature), defaults to 0.
    /// - `spectral_index_3`: Third-order spectral index, defaults to 0.
    /// - `polarization_fraction`: Linear polarization fraction `p`, defaults to 0.
    /// - `polarization_angle`: Polarization angle `χ0` (radians) from north through east,
    ///   defaults to 0.
    /// - `circular_fraction`: Circular polarization fraction `v` (signed), defaults to 0.
    /// - `rotation_measure`: Rotation measure `RM` (rad/m²), defaults to 0.
    ///
    /// # Panics
    /// Panics if:
    /// - reference_frequency is not positive
    /// - reference_intensity is negative
    /// - any spectral index is not finite
    /// - polarization_fraction is negative, or the source is more than fully polarized
    /// - polarization_angle or rotation_measure is not finite
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (direction, reference_frequency, reference_intensity, spectral_index, spectral_index_2 = 0.0, spectral_index_3 = 0.0, polarization_fraction = 0.0, polarization_angle = 0.0, circular_fraction = 0.0, rotation_measure = 0.0))]
    pub fn new(
        direction: Vec3,
        reference_frequency: f64,
        reference_intensity: f64,
        spectral_index: f64,
        spectral_index_2: f64,
        spectral_index_3: f64,
        polarization_fraction: f64,
        polarization_angle: f64,
        circular_fraction: f64,
        rotation_measure: f64,
    ) -> Self {
        let source = Source {
            direction: direction.normalized(),
            reference_frequency,
            reference_intensity,
            spectral_index,
            spectral_index_2,
            spectral_index_3,
            polarization_fraction,
            polarization_angle,
            circular_fraction,
            rotation_measure,
        };
        source.validate();
        source
    }

    /// Returns the source direction vector.
    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    /// Returns the reference frequency in Hz used for spectral intensity scaling.
    pub fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    /// Returns the flux density in Jy at the reference frequency.
    pub fn reference_intensity(&self) -> f64 {
        self.reference_intensity
    }

    /// Returns the first-order (power-law) spectral index.
    pub fn spectral_index(&self) -> f64 {
        self.spectral_index
    }

    /// Returns the second-order spectral index (curvature).
    pub fn spectral_index_2(&self) -> f64 {
        self.spectral_index_2
    }

    /// Returns the third-order spectral index.
    pub fn spectral_index_3(&self) -> f64 {
        self.spectral_index_3
    }

    /// Returns the linear polarization fraction.
    pub fn polarization_fraction(&self) -> f64 {
        self.polarization_fraction
    }

    /// Returns the polarization angle (radians), from north through east.
    pub fn polarization_angle(&self) -> f64 {
        self.polarization_angle
    }

    /// Returns the circular polarization fraction.
    pub fn circular_fraction(&self) -> f64 {
        self.circular_fraction
    }

    /// Returns the rotation measure (rad/m²).
    pub fn rotation_measure(&self) -> f64 {
        self.rotation_measure
    }

    /// Computes the Stokes parameters `(I, Q, U, V)` in Jy at a given frequency (Hz).
    ///
    /// # Panics
    /// Panics if `frequency <= 0.0`.
    pub fn stokes(&self, frequency: f64) -> (f64, f64, f64, f64) {
        assert!(frequency > 0.0, "frequency must be positive");
        let [i, q, u, v] = self.stokes_unchecked(frequency);
        (i, q, u, v)
    }

    /// Computes the flux density in Jy at a given frequency (Hz) using the
    /// log-polynomial spectral model.
    ///
    /// # Panics
    /// Panics if `frequency <= 0.0`.
    pub fn intensity(&self, frequency: f64) -> f64 {
        assert!(frequency > 0.0, "frequency must be positive");
        self.intensity_unchecked(frequency)
    }

    fn __repr__(&self) -> String {
        format!(
            "Source(direction={}, reference_frequency={:.2}MHz, reference_intensity={:.2}Jy, spectral_index={:.3}, spectral_index_2={:.3}, spectral_index_3={:.3}, polarization_fraction={:.3}, polarization_angle={:.3}, circular_fraction={:.3}, rotation_measure={:.3})",
            self.direction.__repr__(),
            self.reference_frequency / 1e6,
            self.reference_intensity,
            self.spectral_index,
            self.spectral_index_2,
            self.spectral_index_3,
            self.polarization_fraction,
            self.polarization_angle,
            self.circular_fraction,
            self.rotation_measure
        )
    }

    pub(crate) fn validate(&self) {
        assert!(
            self.reference_frequency.is_finite() && self.reference_frequency > 0.0,
            "reference_frequency must be positive"
        );
        assert!(
            self.reference_intensity.is_finite() && self.reference_intensity >= 0.0,
            "reference_intensity must be non-negative"
        );
        assert!(
            self.spectral_index.is_finite()
                && self.spectral_index_2.is_finite()
                && self.spectral_index_3.is_finite(),
            "spectral indices must be finite"
        );
        assert!(
            self.polarization_fraction.is_finite()
                && self.circular_fraction.is_finite()
                && self.polarization_fraction >= 0.0
                && self.polarization_fraction.hypot(self.circular_fraction) <= 1.0 + 1e-9,
            "polarization fractions must satisfy p >= 0 and p² + v² <= 1"
        );
        assert!(
            self.polarization_angle.is_finite() && self.rotation_measure.is_finite(),
            "polarization_angle and rotation_measure must be finite"
        );
    }
}

impl Source {
    /// Flux density in Jy at a positive `frequency` (Hz), see [`Source::intensity`].
    pub(crate) fn intensity_unchecked(&self, frequency: f64) -> f64 {
        let x = (frequency / self.reference_frequency).ln();
        self.reference_intensity
            * (x * (self.spectral_index + x * (self.spectral_index_2 + x * self.spectral_index_3)))
                .exp()
    }

    /// Stokes parameters `[I, Q, U, V]` in Jy at a positive `frequency` (Hz), see
    /// [`Source::stokes`].
    pub(crate) fn stokes_unchecked(&self, frequency: f64) -> [f64; 4] {
        let i = self.intensity_unchecked(frequency);
        let lambda = C / frequency;
        let chi = self.polarization_angle + self.rotation_measure * lambda * lambda;
        let p = self.polarization_fraction * i;
        [
            i,
            p * (2.0 * chi).cos(),
            p * (2.0 * chi).sin(),
            self.circular_fraction * i,
        ]
    }
}

/// Transmitter of a calibrator, configured like a software defined radio.
///
/// The transmitter plays its complex baseband buffer cyclically at `sample_rate`,
/// upconverted to the carrier `frequency`. Buffer samples are relative levels: the buffer
/// is scaled such that playing it cyclically radiates `power`, whatever its RMS. The
/// reconstruction filter is applied after this scaling, so any part of the buffer's
/// spectrum outside `bandwidth` reduces the radiated power.
///
/// A transmitter without a beam is isotropic, with `power` its equivalent isotropically
/// radiated power, and its polarization matched to every receiver. A transmitter with a
/// beam radiates `G_tx * power` towards a receiver, with `G_tx` the gain of the beam in
/// that direction, and the received signal includes the polarization mismatch and the
/// phase of both beams.
///
/// A calibrator at distance `d` is received with a power flux density of
/// `G_tx * P / (4π d²)` (W/m²), times the effective area of the receiver relative to the
/// reference area of the array. It is spread over the sample frequency `f_s` of the array,
/// so a calibrator whose signal is white over `f_s` is received like a fully polarized sky
/// source of `G_tx * P / (4π d² f_s)` W/m²/Hz, i.e. that divided by 1e-26 in Jy.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Transmitter {
    pub(crate) frequency: f64,
    pub(crate) sample_rate: f64,
    pub(crate) power: f64,
    pub(crate) bandwidth: Option<f64>,
    pub(crate) buffer: Arc<[Complex32]>,
    pub(crate) start_time: f64,
    pub(crate) beam: Option<Beam>,
}

#[pymethods]
impl Transmitter {
    /// Creates a new transmitter.
    ///
    /// # Arguments
    /// - `frequency`: Carrier frequency (Hz).
    /// - `sample_rate`: Rate at which buffer samples are played out (Hz).
    /// - `power`: Power (W) when playing the buffer: the equivalent isotropically radiated
    ///   power without a beam, or the power fed to the beam, whose gain is applied, with one.
    /// - `buffer`: Complex baseband samples, transmitted cyclically.
    /// - `bandwidth`: Optional bandwidth of the reconstruction filter (Hz).
    /// - `start_time`: Time (s) at which the first buffer sample is transmitted.
    /// - `beam`: Optional beam of the transmitter, oriented by the calibrator's attitude.
    ///   Outside the frequencies of a tabulated beam, its nearest frequency is used.
    ///
    /// # Panics
    /// Panics if:
    /// - frequency is negative
    /// - sample_rate is not positive
    /// - power is negative
    /// - bandwidth is not positive
    /// - buffer is empty, all zero or contains non-finite samples
    /// - beam is relative (as `Beam.cos_squared()`), or tabulated without covering frequency
    #[new]
    #[pyo3(signature = (frequency, sample_rate, power, buffer, bandwidth = None, start_time = 0.0, beam = None))]
    pub fn new(
        frequency: f64,
        sample_rate: f64,
        power: f64,
        buffer: Vec<Complex32>,
        bandwidth: Option<f64>,
        start_time: f64,
        beam: Option<Beam>,
    ) -> Self {
        let transmitter = Transmitter {
            frequency,
            sample_rate,
            power,
            bandwidth,
            buffer: buffer.into(),
            start_time,
            beam,
        };
        transmitter.validate();
        transmitter
    }

    /// Returns the carrier frequency in Hz.
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Returns the rate in Hz at which buffer samples are played out.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Returns the power in W: the equivalent isotropically radiated power without a beam,
    /// or the power fed to the beam with one.
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Returns the bandwidth of the reconstruction filter in Hz, if any.
    pub fn bandwidth(&self) -> Option<f64> {
        self.bandwidth
    }

    /// Returns the transmitted buffer.
    pub fn buffer(&self) -> Vec<Complex32> {
        self.buffer.to_vec()
    }

    /// Returns the time in s at which the first buffer sample is transmitted.
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Returns the beam of the transmitter, if any.
    pub fn beam(&self) -> Option<Beam> {
        self.beam.clone()
    }

    /// Sets the beam of the transmitter, or makes it isotropic with `None`.
    ///
    /// # Panics
    /// Panics if beam is relative, or tabulated without covering the carrier frequency.
    #[pyo3(signature = (beam = None))]
    pub fn set_beam(&mut self, beam: Option<Beam>) {
        self.beam = beam;
        self.validate();
    }

    /// Sets the carrier frequency in Hz.
    ///
    /// # Panics
    /// Panics if frequency is negative.
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
        self.validate();
    }

    /// Sets the rate in Hz at which buffer samples are played out.
    ///
    /// # Panics
    /// Panics if sample_rate is not positive.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.validate();
    }

    /// Sets the power in W: the equivalent isotropically radiated power without a beam, or
    /// the power fed to the beam with one.
    ///
    /// # Panics
    /// Panics if power is negative.
    pub fn set_power(&mut self, power: f64) {
        self.power = power;
        self.validate();
    }

    /// Sets the bandwidth of the reconstruction filter in Hz, or removes the filter with `None`.
    ///
    /// # Panics
    /// Panics if bandwidth is not positive.
    #[pyo3(signature = (bandwidth = None))]
    pub fn set_bandwidth(&mut self, bandwidth: Option<f64>) {
        self.bandwidth = bandwidth;
        self.validate();
    }

    /// Sets the buffer that is transmitted cyclically.
    ///
    /// # Panics
    /// Panics if buffer is empty, all zero or contains non-finite samples.
    pub fn set_buffer(&mut self, buffer: Vec<Complex32>) {
        self.buffer = buffer.into();
        self.validate();
    }

    /// Sets the time in s at which the first buffer sample is transmitted.
    ///
    /// # Panics
    /// Panics if start_time is not finite.
    pub fn set_start_time(&mut self, start_time: f64) {
        self.start_time = start_time;
        self.validate();
    }

    fn __repr__(&self) -> String {
        let bandwidth = self
            .bandwidth
            .map_or("None".to_string(), |b| format!("{:.2}MHz", b / 1e6));
        let beam = self
            .beam
            .as_ref()
            .map_or("None".to_string(), |b| format!("{b:?}"));
        format!(
            "Transmitter(frequency = {:.2}MHz, sample_rate = {:.2}MHz, power = {}W, bandwidth = {}, buffer = [...{} samples...], start_time = {}s, beam = {})",
            self.frequency / 1e6,
            self.sample_rate / 1e6,
            self.power,
            bandwidth,
            self.buffer.len(),
            self.start_time,
            beam
        )
    }

    pub(crate) fn validate(&self) {
        assert!(
            self.frequency.is_finite() && self.frequency >= 0.0,
            "frequency must be non-negative"
        );
        assert!(
            self.sample_rate.is_finite() && self.sample_rate > 0.0,
            "sample_rate must be positive"
        );
        assert!(
            self.power.is_finite() && self.power >= 0.0,
            "power must be non-negative"
        );
        assert!(
            self.bandwidth.is_none_or(|b| b.is_finite() && b > 0.0),
            "bandwidth must be positive"
        );
        assert!(!self.buffer.is_empty(), "buffer must not be empty");
        assert!(
            self.buffer.iter().all(|s| s.is_finite()),
            "buffer must only contain finite samples"
        );
        assert!(
            self.buffer.iter().any(|s| *s != Complex32::ZERO),
            "buffer must not be all zero"
        );
        assert!(self.start_time.is_finite(), "start_time must be finite");
        if let Some(beam) = &self.beam {
            assert!(
                beam.is_absolute(),
                "a transmitter beam must hold directivity, gain or realized gain, not a relative gain"
            );
            beam.assert_covers(self.frequency, self.frequency, "the transmitter");
        }
    }
}

/// Calibrator used to model a known reference emitter (e.g. a satellite).
///
/// The calibrator transmits a deterministic signal from a position that moves
/// with constant acceleration:
/// `p(t) = position + velocity * (t - epoch) + acceleration * (t - epoch)^2 / 2`.
///
/// Its attitude orients the beam of its transmitter. The beam is evaluated once per
/// sample window, for the geometry at the middle of the window.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Calibrator {
    pub(crate) position: Vec3,
    pub(crate) velocity: Vec3,
    pub(crate) acceleration: Vec3,
    pub(crate) epoch: f64,
    pub(crate) transmitter: Transmitter,
    pub(crate) attitude: Attitude,
}

#[pymethods]
impl Calibrator {
    /// Creates a new calibrator.
    ///
    /// # Arguments
    /// - `position`: Position (m) of the calibrator at `epoch`.
    /// - `transmitter`: Transmitter of the calibrator.
    /// - `velocity`: Velocity (m/s) at `epoch`, zero if not given.
    /// - `acceleration`: Constant acceleration (m/s²), zero if not given.
    /// - `epoch`: Time (s) at which `position` and `velocity` are valid.
    /// - `attitude`: Attitude orienting the transmitter's beam, the identity orientation
    ///   if not given.
    ///
    /// # Panics
    /// Panics if any of the kinematic parameters is not finite.
    #[new]
    #[pyo3(signature = (position, transmitter, velocity = None, acceleration = None, epoch = 0.0, attitude = None))]
    pub fn new(
        position: Vec3,
        transmitter: Transmitter,
        velocity: Option<Vec3>,
        acceleration: Option<Vec3>,
        epoch: f64,
        attitude: Option<Attitude>,
    ) -> Self {
        let calibrator = Calibrator {
            position,
            velocity: velocity.unwrap_or_default(),
            acceleration: acceleration.unwrap_or_default(),
            epoch,
            transmitter,
            attitude: attitude.unwrap_or_default(),
        };
        calibrator.validate();
        calibrator
    }

    /// Returns the calibrator position at `epoch`.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Returns the calibrator velocity at `epoch`.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Returns the calibrator acceleration.
    pub fn acceleration(&self) -> Vec3 {
        self.acceleration
    }

    /// Returns the time at which `position` and `velocity` are valid.
    pub fn epoch(&self) -> f64 {
        self.epoch
    }

    /// Returns the calibrator's transmitter.
    pub fn transmitter(&self) -> Transmitter {
        self.transmitter.clone()
    }

    /// Returns the calibrator's attitude.
    pub fn attitude(&self) -> Attitude {
        self.attitude
    }

    /// Replaces the calibrator's attitude.
    pub fn set_attitude(&mut self, attitude: Attitude) {
        self.attitude = attitude;
    }

    /// Replaces the kinematic state of the calibrator.
    ///
    /// # Arguments
    /// - `position`: Position (m) at `epoch`.
    /// - `velocity`: Velocity (m/s) at `epoch`, zero if not given.
    /// - `acceleration`: Constant acceleration (m/s²), zero if not given.
    /// - `epoch`: Time (s) at which `position` and `velocity` are valid.
    ///
    /// # Panics
    /// Panics if any of the kinematic parameters is not finite.
    #[pyo3(signature = (position, velocity = None, acceleration = None, epoch = 0.0))]
    pub fn set_state(
        &mut self,
        position: Vec3,
        velocity: Option<Vec3>,
        acceleration: Option<Vec3>,
        epoch: f64,
    ) {
        self.position = position;
        self.velocity = velocity.unwrap_or_default();
        self.acceleration = acceleration.unwrap_or_default();
        self.epoch = epoch;
        self.validate();
    }

    /// Replaces the calibrator's transmitter.
    pub fn set_transmitter(&mut self, transmitter: Transmitter) {
        self.transmitter = transmitter;
    }

    /// Computes the calibrator position at a given time.
    ///
    /// # Arguments
    /// - `time`: Time (s).
    ///
    /// # Returns
    /// The position (m) at `time`.
    pub fn position_at(&self, time: f64) -> Vec3 {
        let dt = time - self.epoch;
        self.position + self.velocity * dt + self.acceleration * (0.5 * dt * dt)
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibrator(position = {}, velocity = {}, acceleration = {}, epoch = {}s, transmitter = {}, attitude = {})",
            self.position.__repr__(),
            self.velocity.__repr__(),
            self.acceleration.__repr__(),
            self.epoch,
            self.transmitter.__repr__(),
            self.attitude.__repr__()
        )
    }

    pub(crate) fn validate(&self) {
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        assert!(finite(self.position), "position must be finite");
        assert!(finite(self.velocity), "velocity must be finite");
        assert!(finite(self.acceleration), "acceleration must be finite");
        assert!(self.epoch.is_finite(), "epoch must be finite");
        self.transmitter.validate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(spectral_index_2: f64, spectral_index_3: f64) -> Source {
        Source::new(
            Vec3::new(0.0, 0.0, 1.0),
            75e6,
            2.0,
            -0.7,
            spectral_index_2,
            spectral_index_3,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    #[test]
    fn noise_temperature_interpolates_and_holds() {
        let table = NoiseTemperature::Table(vec![[40e6, 2000.0], [60e6, 1000.0], [100e6, 500.0]]);
        assert_eq!(table.at(30e6), 2000.0);
        assert_eq!(table.at(40e6), 2000.0);
        assert!((table.at(50e6) - 1500.0).abs() < 1e-9);
        assert!((table.at(80e6) - 750.0).abs() < 1e-9);
        assert_eq!(table.at(100e6), 500.0);
        assert_eq!(table.at(120e6), 500.0);
        assert_eq!(NoiseTemperature::Constant(300.0).at(75e6), 300.0);
        table.validate([40e6, 100e6]);
    }

    #[test]
    #[should_panic(expected = "must cover")]
    fn noise_temperature_table_must_cover_bandpass() {
        NoiseTemperature::Table(vec![[50e6, 1000.0], [90e6, 500.0]]).validate([40e6, 100e6]);
    }

    #[test]
    #[should_panic(expected = "increasing")]
    fn noise_temperature_table_must_increase() {
        NoiseTemperature::Table(vec![[100e6, 500.0], [40e6, 1000.0]]).validate([40e6, 100e6]);
    }

    #[test]
    fn intensity_reduces_to_power_law() {
        let source = source(0.0, 0.0);
        for frequency in [10e6, 75e6, 150e6] {
            let expected = 2.0 * (frequency / 75e6f64).powf(-0.7);
            assert!((source.intensity(frequency) - expected).abs() <= 1e-12 * expected);
        }
    }

    #[test]
    fn intensity_includes_higher_orders() {
        let source = source(-0.3, 0.1);
        assert_eq!(source.intensity(75e6), 2.0);

        let x = 2f64.ln();
        let expected = 2.0 * (-0.7 * x - 0.3 * x * x + 0.1 * x * x * x).exp();
        let intensity = source.intensity(150e6);
        assert!((intensity - expected).abs() <= 1e-12 * expected);
    }

    #[test]
    #[should_panic(expected = "spectral indices must be finite")]
    fn rejects_non_finite_spectral_index() {
        source(f64::NAN, 0.0);
    }
}
