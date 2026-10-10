use crate::Vec3;
use num_complex::Complex32;
use pyo3::prelude::*;
use std::sync::Arc;

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

/// Transmitter of a calibrator, configured like a software defined radio.
///
/// The transmitter plays its complex baseband buffer cyclically at `sample_rate`,
/// upconverted to the carrier `frequency`. Buffer samples are relative levels, like
/// the full scale of a DAC: `power` is the radiated power for a buffer with unit RMS.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Transmitter {
    pub(crate) frequency: f64,
    pub(crate) sample_rate: f64,
    pub(crate) power: f64,
    pub(crate) bandwidth: Option<f64>,
    pub(crate) buffer: Arc<[Complex32]>,
    pub(crate) start_time: f64,
}

#[pymethods]
impl Transmitter {
    /// Creates a new transmitter.
    ///
    /// # Arguments
    /// - `frequency`: Carrier frequency (Hz).
    /// - `sample_rate`: Rate at which buffer samples are played out (Hz).
    /// - `power`: Equivalent isotropically radiated power (W) for a buffer with unit RMS.
    /// - `buffer`: Complex baseband samples, transmitted cyclically.
    /// - `bandwidth`: Optional bandwidth of the reconstruction filter (Hz).
    /// - `start_time`: Time (s) at which the first buffer sample is transmitted.
    ///
    /// # Panics
    /// Panics if:
    /// - frequency is negative
    /// - sample_rate is not positive
    /// - power is negative
    /// - bandwidth is not positive
    /// - buffer is empty or contains non-finite samples
    #[new]
    #[pyo3(signature = (frequency, sample_rate, power, buffer, bandwidth = None, start_time = 0.0))]
    pub fn new(
        frequency: f64,
        sample_rate: f64,
        power: f64,
        buffer: Vec<Complex32>,
        bandwidth: Option<f64>,
        start_time: f64,
    ) -> Self {
        let transmitter = Transmitter {
            frequency,
            sample_rate,
            power,
            bandwidth,
            buffer: buffer.into(),
            start_time,
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

    /// Returns the radiated power in W for a buffer with unit RMS.
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

    /// Sets the radiated power in W for a buffer with unit RMS.
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
    /// Panics if buffer is empty or contains non-finite samples.
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
        format!(
            "Transmitter(frequency = {:.2}MHz, sample_rate = {:.2}MHz, power = {}W, bandwidth = {}, buffer = [...{} samples...], start_time = {}s)",
            self.frequency / 1e6,
            self.sample_rate / 1e6,
            self.power,
            bandwidth,
            self.buffer.len(),
            self.start_time
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
        assert!(self.start_time.is_finite(), "start_time must be finite");
    }
}

/// Calibrator used to model a known reference emitter (e.g. a satellite).
///
/// The calibrator transmits a deterministic signal from a position that moves
/// with constant acceleration:
/// `p(t) = position + velocity * (t - epoch) + acceleration * (t - epoch)^2 / 2`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Calibrator {
    pub(crate) position: Vec3,
    pub(crate) velocity: Vec3,
    pub(crate) acceleration: Vec3,
    pub(crate) epoch: f64,
    pub(crate) transmitter: Transmitter,
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
    ///
    /// # Panics
    /// Panics if any of the kinematic parameters is not finite.
    #[new]
    #[pyo3(signature = (position, transmitter, velocity = None, acceleration = None, epoch = 0.0))]
    pub fn new(
        position: Vec3,
        transmitter: Transmitter,
        velocity: Option<Vec3>,
        acceleration: Option<Vec3>,
        epoch: f64,
    ) -> Self {
        let calibrator = Calibrator {
            position,
            velocity: velocity.unwrap_or_default(),
            acceleration: acceleration.unwrap_or_default(),
            epoch,
            transmitter,
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
            "Calibrator(position = {}, velocity = {}, acceleration = {}, epoch = {}s, transmitter = {})",
            self.position.__repr__(),
            self.velocity.__repr__(),
            self.acceleration.__repr__(),
            self.epoch,
            self.transmitter.__repr__()
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
