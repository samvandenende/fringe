use crate::{Array, Calibrator, Source};
use csv::{ReaderBuilder, WriterBuilder};
use num_complex::Complex32;
use pyo3::PyResult;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::f32::consts::TAU;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

/// 3D Cartesian vector
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vec3 {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub z: f64,
}

#[pymethods]
impl Vec3 {
    /// Creates a new 3D vector.
    ///
    /// # Arguments
    /// - `x`: X component
    /// - `y`: Y component
    /// - `z`: Z component
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn __repr__(&self) -> String {
        format!("Vec3({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }

    fn __add__(&self, rhs: Vec3) -> Vec3 {
        *self + rhs
    }

    fn __sub__(&self, rhs: Vec3) -> Vec3 {
        *self - rhs
    }

    fn __mul__(&self, rhs: f64) -> Vec3 {
        *self * rhs
    }

    fn __rmul__(&self, rhs: f64) -> Vec3 {
        rhs * *self
    }

    fn __truediv__(&self, rhs: f64) -> Vec3 {
        *self / rhs
    }

    /// In-place vector addition.
    pub fn add_inplace(&mut self, other: Vec3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }

    /// In-place vector subtraction.
    pub fn sub_inplace(&mut self, other: Vec3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }

    /// Scales the vector by a scalar.
    pub fn scale(&mut self, s: f64) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }

    /// Normalizes the vector in-place to unit length.
    ///
    /// If the vector has zero magnitude, it remains unchanged.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n != 0.0 {
            self.x /= n;
            self.y /= n;
            self.z /= n;
        }
    }

    /// Computes the dot product with another vector.
    ///
    /// # Arguments
    /// - `other`: Right-hand-side vector
    pub fn dot(&self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product with another vector.
    ///
    /// # Arguments
    /// - `other`: Right-hand-side vector
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the squared Euclidean norm (square magnitude).
    pub fn norm2(&self) -> f64 {
        self.dot(*self)
    }

    /// Returns the Euclidean norm (magnitude).
    pub fn norm(&self) -> f64 {
        self.norm2().sqrt()
    }

    /// Returns a normalized copy of the vector.
    ///
    /// If the vector has zero magnitude, it is returned unchanged.
    pub fn normalized(&self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 { *self } else { *self / n }
    }

    /// Converts the vector into a tuple representation.
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    /// Constructs a unit vector from spherical coordinates (RA, Dec).
    ///
    /// # Arguments
    /// * `ra` — Right ascension in radians
    /// * `dec` — Declination in radians
    #[staticmethod]
    pub fn from_ra_dec(ra: f64, dec: f64) -> Self {
        let x = dec.cos() * ra.cos();
        let y = dec.cos() * ra.sin();
        let z = dec.sin();
        Vec3::new(x, y, z).normalized()
    }

    /// Converts the vector into spherical coordinates (RA, Dec) in radians.
    ///
    /// Returns:
    /// * `(ra, dec)` where
    ///   - `ra ∈ [0, 2π)`
    ///   - `dec ∈ [-π/2, π/2]`
    pub fn to_ra_dec(&self) -> (f64, f64) {
        let normalized = self.normalized();

        let z_clamped = normalized.z.clamp(-1.0, 1.0);

        let dec = z_clamped.asin();
        let mut ra = normalized.y.atan2(normalized.x);

        if ra < 0.0 {
            ra += std::f64::consts::TAU;
        }

        (ra, dec)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

/// Phases used for signal generation.
///
/// This structure stores random phase offsets for:
/// - system noise per antenna
/// - per-source spectral phases
/// - deterministic calibrator signal
pub(crate) struct Phases {
    pub(crate) system_noise: Vec<f32>,
    pub(crate) sources: Arc<Vec<f32>>,
    pub(crate) calibrator_signal: Arc<Vec<f32>>,
    _original_calibrator_signal: Vec<f32>,
}

impl Phases {
    /// Creates a new `Phases` with random phases.
    ///
    /// # Arguments
    /// - `rng`: Random number generator used to seed phase values
    /// - `array`: Antenna array configuration (used for sizing)
    /// - `num_sources`: Number of signal sources in the simulation
    /// - `frequency_resolution`: FFT oversampling factor
    ///
    /// # Returns
    /// Fully initialized phases
    pub(crate) fn new(
        rng: &mut impl rand::Rng,
        array: &Array,
        num_sources: usize,
        frequency_resolution: usize,
    ) -> Self {
        let num_spectrum_bins = array.sample_window_size * frequency_resolution;

        let calibrator_signal = (0..num_spectrum_bins)
            .map(|_| rng.random_range(0.0..TAU))
            .collect::<Vec<f32>>();

        let mut phases = Phases {
            system_noise: Vec::new(),
            sources: Arc::new(Vec::new()),
            calibrator_signal: Arc::new(Vec::new()),
            _original_calibrator_signal: calibrator_signal,
        };

        phases.update(rng, array, num_sources, frequency_resolution);

        phases
    }

    /// Updates all stochastic phase components with new random values.
    ///
    /// This regenerates:
    /// - system noise phases per antenna
    /// - per-source spectral phases
    /// - time-shifted calibrator signal phases
    ///
    /// # Arguments
    /// - `rng`: Random number generator
    /// - `array`: Antenna array configuration
    /// - `num_sources`: Number of active sources
    /// - `frequency_resolution`: FFT oversampling factor
    pub(crate) fn update(
        &mut self,
        rng: &mut impl rand::Rng,
        array: &Array,
        num_sources: usize,
        frequency_resolution: usize,
    ) {
        let num_spectrum_bins = array.sample_window_size * frequency_resolution;

        let num_system_noise_phases = array.antenna_positions.len() * num_spectrum_bins;
        let num_source_phases = num_sources * num_spectrum_bins;

        self.system_noise = (0..num_system_noise_phases)
            .map(|_| rng.random_range(0.0..TAU))
            .collect();
        self.sources = Arc::new(
            (0..num_source_phases)
                .map(|_| rng.random_range(0.0..TAU))
                .collect(),
        );

        assert_eq!(self._original_calibrator_signal.len(), num_spectrum_bins);
        let mut calibrator_signal = self._original_calibrator_signal.clone();
        let cal_signal_period = num_spectrum_bins as f64 / array.sample_frequency;
        let time_delay = rng.random::<f64>() * cal_signal_period;
        for (i, phase) in calibrator_signal.iter_mut().enumerate() {
            use std::f64::consts::TAU;
            let freq = fft_bin_frequency(num_spectrum_bins, array.sample_frequency, i);
            *phase += ((TAU * freq * time_delay) % TAU) as f32;
        }
        self.calibrator_signal = Arc::new(calibrator_signal);
    }
}

/// Computes the frequency corresponding to a given FFT bin index.
///
/// # Arguments
/// - `num_bins`: Total number of FFT bins
/// - `sample_freq`: Sampling frequency in Hz
/// - `bin`: Bin index
///
/// # Returns
/// Frequency in Hz corresponding to the given bin
pub fn fft_bin_frequency(num_bins: usize, sample_freq: f64, bin: usize) -> f64 {
    let half_n: usize = num_bins / 2;

    if bin < half_n {
        bin as f64 * sample_freq / num_bins as f64
    } else {
        (bin as i64 - num_bins as i64) as f64 * sample_freq / num_bins as f64
    }
}

/// Saves the array configuration to a file in JSON format.
///
/// # Arguments
/// - `array`: The array.
/// - `filepath`: Destination path where the array will be written.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be created or written
#[pyfunction]
pub fn save_array(array: &Array, filepath: &str) -> PyResult<()> {
    let file = std::fs::File::create(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to create file '{}': {}", filepath, e)))?;

    let writer = std::io::BufWriter::new(file);

    serde_json::to_writer_pretty(writer, array)
        .map_err(|e| PyIOError::new_err(format!("Failed to serialize Array: {}", e)))?;

    Ok(())
}

/// Loads an array configuration from a JSON file.
///
/// # Arguments
/// - `filepath`: Path to the file containing a serialized `Array`.
///
/// # Returns
/// A reconstructed `Array` instance.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be read or parsed.
///
/// # Panics
/// Panics if:
/// - antenna_positions is empty
/// - sample_frequency is not positive
/// - downmix_frequency is negative
/// - bandpass bounds are invalid or exceed Nyquist limit
/// - sample_window_size is not a power of two
/// - system_noise_intensity is negative
#[pyfunction]
pub fn load_array(filepath: &str) -> PyResult<Array> {
    let file = std::fs::File::open(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file '{}': {}", filepath, e)))?;

    let reader = std::io::BufReader::new(file);

    let array: Array = serde_json::from_reader(reader)
        .map_err(|e| PyIOError::new_err(format!("Failed to deserialize Array: {}", e)))?;

    array.validate();

    Ok(array)
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SourceCsv {
    right_ascension: f64,
    declination: f64,
    reference_frequency: f64,
    reference_intensity: f64,
    spectral_index: f64,
}

impl From<Source> for SourceCsv {
    fn from(source: Source) -> Self {
        let (right_ascension, declination) = source.direction.to_ra_dec();
        SourceCsv {
            right_ascension,
            declination,
            reference_frequency: source.reference_frequency,
            reference_intensity: source.reference_intensity,
            spectral_index: source.spectral_index,
        }
    }
}

impl From<SourceCsv> for Source {
    fn from(source: SourceCsv) -> Self {
        let direction = Vec3::from_ra_dec(source.right_ascension, source.declination);
        Source::new(
            direction,
            source.reference_frequency,
            source.reference_intensity,
            source.spectral_index,
        )
    }
}

/// Saves the list of sources to a file in CSV format.
///
/// # Arguments
/// - `sources`: The list of sources.
/// - `filepath`: Destination path where the sources will be written.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be created or written
#[pyfunction]
pub fn save_sources(sources: Vec<Source>, filepath: &str) -> PyResult<()> {
    let file = std::fs::File::create(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to create '{}': {}", filepath, e)))?;

    let writer = std::io::BufWriter::new(file);

    let mut csv_writer = WriterBuilder::new().has_headers(true).from_writer(writer);

    for src in sources {
        let src_csv = SourceCsv::from(src);
        csv_writer
            .serialize(src_csv)
            .map_err(|e| PyIOError::new_err(format!("CSV serialize error: {}", e)))?;
    }

    csv_writer
        .flush()
        .map_err(|e| PyIOError::new_err(format!("CSV flush error: {}", e)))?;

    Ok(())
}

/// Loads a list of sources from a CSV file.
///
/// # Arguments
/// - `filepath`: Path to the file containing a serialized list of `Source`s.
///
/// # Returns
/// A reconstructed list of `Source`s.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be read or parsed.
///
/// # Panics
/// Panics if, for a `Source`:
/// - reference_frequency is not positive
/// - reference_intensity is negative
#[pyfunction]
pub fn load_sources(filepath: &str) -> PyResult<Vec<Source>> {
    let file = std::fs::File::open(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to open '{}': {}", filepath, e)))?;

    let reader = std::io::BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut sources = Vec::new();
    for result in csv_reader.deserialize() {
        let src_csv: SourceCsv =
            result.map_err(|e| PyIOError::new_err(format!("CSV deserialize error: {}", e)))?;
        let src = Source::from(src_csv);
        sources.push(src);
    }

    Ok(sources)
}

/// Saves the calibrator to a file in JSON format.
///
/// # Arguments
/// - `calibrator`: The calibrator.
/// - `filepath`: Destination path where the calibrator will be written.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be created or written
#[pyfunction]
pub fn save_calibrator(calibrator: &Calibrator, filepath: &str) -> PyResult<()> {
    let file = std::fs::File::create(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to create file '{}': {}", filepath, e)))?;

    let writer = std::io::BufWriter::new(file);

    serde_json::to_writer_pretty(writer, calibrator)
        .map_err(|e| PyIOError::new_err(format!("Failed to serialize Array: {}", e)))?;

    Ok(())
}

/// Loads a calibrator from a JSON file.
///
/// # Arguments
/// - `filepath`: Path to the file containing a serialized `Calibrator`.
///
/// # Returns
/// A reconstructed `Calibrator` instance.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be read or parsed.
///
/// # Panics
/// Panics if intensity is negative.
#[pyfunction]
pub fn load_calibrator(filepath: &str) -> PyResult<Calibrator> {
    let file = std::fs::File::open(filepath)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file '{}': {}", filepath, e)))?;

    let reader = std::io::BufReader::new(file);

    let calibrator: Calibrator = serde_json::from_reader(reader)
        .map_err(|e| PyIOError::new_err(format!("Failed to deserialize Array: {}", e)))?;

    calibrator.validate();

    Ok(calibrator)
}

pub(crate) fn normalize_and_truncate(
    samples: &[Complex32],
    sample_window_size: usize,
) -> Vec<Complex32> {
    let excess = samples.len() - sample_window_size;
    let start = excess / 2;
    let norm = (samples.len() as f32).sqrt();
    samples
        .iter()
        .skip(start)
        .take(sample_window_size)
        .map(|s| s.unscale(norm))
        .collect()
}
