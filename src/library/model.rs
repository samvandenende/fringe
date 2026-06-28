use crate::Vec3;
use pyo3::prelude::*;

/// Array model and its signal acquisition parameters.
///
/// This structure defines:
/// - array geometry
/// - sampling configuration
/// - frequency conversion parameters
/// - simulation noise characteristics
#[pyclass(from_py_object)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Array {
    pub(crate) sample_frequency: f64,
    pub(crate) downmix_frequency: f64,
    pub(crate) bandpass: [f64; 2],
    pub(crate) sample_window_size: usize,
    pub(crate) system_noise_intensity: f64,
    pub(crate) antenna_positions: Vec<Vec3>,
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
    /// - `system_noise_intensity`: System noise intensity.
    ///
    /// # Panics
    /// Panics if:
    /// - antenna_positions is empty
    /// - sample_frequency is not positive
    /// - downmix_frequency is negative
    /// - bandpass bounds are invalid or exceed Nyquist limit
    /// - sample_window_size is not a power of two
    /// - system_noise_intensity is negative
    #[new]
    pub fn new(
        antenna_positions: Vec<Py<Vec3>>,
        sample_frequency: f64,
        downmix_frequency: f64,
        bandpass_fmin: f64,
        bandpass_fmax: f64,
        sample_window_size: usize,
        system_noise_intensity: f64,
    ) -> Self {
        Python::attach(|py| -> Array {
            let antenna_positions = antenna_positions.iter().map(|p| *p.borrow(py)).collect();
            let array = Array {
                antenna_positions,
                sample_frequency,
                downmix_frequency,
                bandpass: [bandpass_fmin, bandpass_fmax],
                sample_window_size,
                system_noise_intensity,
            };
            array.validate();
            array
        })
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

    /// Returns the system noise intensity.
    pub fn system_noise_intensity(&self) -> f64 {
        self.system_noise_intensity
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(
    sample_frequency = {:.2}MHz,
    downmix_frequency = {:.2}MHz,
    bandpass = ({:.2} - {:.2})MHz,
    sample_window_size = {},
    system_noise_intensity = {:.2},
    antenna_positions = [...{} items...]
)",
            self.sample_frequency / 1e6,
            self.downmix_frequency / 1e6,
            self.bandpass[0] / 1e6,
            self.bandpass[1] / 1e6,
            self.sample_window_size,
            self.system_noise_intensity,
            self.antenna_positions.len()
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
        assert!(
            self.system_noise_intensity.is_finite() && self.system_noise_intensity >= 0.0,
            "system noise intensity must be non-negative"
        );
    }
}

/// Source in the simulated sky model.
///
/// Each source emits a frequency-dependent signal characterized by a reference
/// intensity and spectral index, and is located at a fixed direction vector.
#[pyclass(from_py_object)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Source {
    pub(crate) direction: Vec3,
    pub(crate) reference_frequency: f64,
    pub(crate) reference_intensity: f64,
    pub(crate) spectral_index: f64,
}

#[pymethods]
impl Source {
    /// Creates a new signal source.
    ///
    /// # Arguments
    /// - `direction`: Unit vector indicating source direction in space.
    /// - `reference_frequency`: Frequency at which intensity is defined.
    /// - `reference_intensity`: Signal strength at the reference frequency.
    /// - `spectral_index`: Power-law spectral index of the source.
    ///
    /// # Panics
    /// Panics if:
    /// - reference_frequency is not positive
    /// - reference_intensity is negative
    #[new]
    pub fn new(
        direction: Vec3,
        reference_frequency: f64,
        reference_intensity: f64,
        spectral_index: f64,
    ) -> Self {
        let source = Source {
            direction: direction.normalized(),
            reference_frequency,
            reference_intensity,
            spectral_index,
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

    /// Returns the reference intensity at the reference frequency.
    pub fn reference_intensity(&self) -> f64 {
        self.reference_intensity
    }

    /// Returns the spectral index used in the power-law intensity model.
    pub fn spectral_index(&self) -> f64 {
        self.spectral_index
    }

    /// Computes the intensity at a given frequency using a power-law model.
    ///
    /// # Panics
    /// Panics if `frequency <= 0.0`.
    pub fn intensity(&self, frequency: f64) -> f64 {
        assert!(frequency > 0.0, "frequency must be positive");
        self.reference_intensity * (frequency / self.reference_frequency).powf(self.spectral_index)
    }

    fn __repr__(&self) -> String {
        format!(
            "Source(direction={}, reference_frequency={:.2}MHz, reference_intensity={:.2}, spectral_index={:.3})",
            self.direction.__repr__(),
            self.reference_frequency / 1e6,
            self.reference_intensity,
            self.spectral_index
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
    }
}

/// Calibrator used to model a known reference emitter (e.g. a satellite).
///
/// The calibrator acts as a deterministic signal source which can
/// be used for system calibration.
#[pyclass(from_py_object)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Calibrator {
    pub(crate) position: Vec3,
    pub(crate) intensity: f64,
}

#[pymethods]
impl Calibrator {
    /// Creates a new calibrator.
    ///
    /// # Arguments
    /// - `position`: Position of the calibration source.
    /// - `intensity`: Signal intensity of the calibrator.
    ///
    /// # Panics
    /// Panics if intensity is negative.
    #[new]
    pub fn new(position: Vec3, intensity: f64) -> Self {
        let calibrator = Calibrator {
            position,
            intensity,
        };
        calibrator.validate();
        calibrator
    }

    /// Returns the calibrator position.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Returns the calibrator intensity.
    pub fn intensity(&self) -> f64 {
        self.intensity
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibrator(position = {:?}, intensity = {})",
            self.position, self.intensity
        )
    }

    pub(crate) fn validate(&self) {
        assert!(self.intensity >= 0.0, "intensity must be non-negative");
    }
}
