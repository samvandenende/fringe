use crate::{Array, Calibrator, Source, Transmitter};
use csv::{ReaderBuilder, WriterBuilder};
use num_complex::Complex32;
use pyo3::PyResult;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::f32::consts::TAU;
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;
use std::sync::Arc;

/// 3D Cartesian vector
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
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
pub(crate) struct Phases {
    pub(crate) system_noise: Vec<f32>,
    pub(crate) sources: Arc<Vec<f32>>,
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
        let mut phases = Phases {
            system_noise: Vec::new(),
            sources: Arc::new(Vec::new()),
        };

        phases.update(rng, array, num_sources, frequency_resolution);

        phases
    }

    /// Updates all stochastic phase components with new random values.
    ///
    /// This regenerates:
    /// - system noise phases per antenna
    /// - per-source spectral phases
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

#[derive(serde::Serialize, serde::Deserialize)]
struct CalibratorJson {
    position: Vec3,
    #[serde(default)]
    velocity: Vec3,
    #[serde(default)]
    acceleration: Vec3,
    #[serde(default)]
    epoch: f64,
    transmitter: TransmitterJson,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TransmitterJson {
    frequency: f64,
    sample_rate: f64,
    power: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bandwidth: Option<f64>,
    #[serde(default)]
    start_time: f64,
    /// Path of the buffer file, relative to the calibrators file.
    buffer: String,
}

/// Saves a list of calibrators to a file in JSON format.
///
/// The transmit buffers are written next to it as raw interleaved little-endian
/// 32-bit float IQ files (`cf32_le`), named `<file stem>.<index>.cf32`.
/// Calibrators sharing a buffer share a buffer file.
///
/// # Arguments
/// - `calibrators`: The list of calibrators.
/// - `filepath`: Destination path where the calibrators will be written.
///
/// # Errors
/// Returns a Python `IOError` if a file cannot be created or written
#[pyfunction]
pub fn save_calibrators(calibrators: Vec<Calibrator>, filepath: &str) -> PyResult<()> {
    let path = Path::new(filepath);
    let directory = path.parent().unwrap_or(Path::new(""));
    let stem = path
        .file_stem()
        .ok_or_else(|| PyIOError::new_err(format!("Invalid file path '{}'", filepath)))?
        .to_string_lossy();

    let mut buffer_files: Vec<(Arc<[Complex32]>, String)> = Vec::new();
    let mut entries = Vec::with_capacity(calibrators.len());
    for calibrator in calibrators {
        let transmitter = &calibrator.transmitter;
        let existing = buffer_files
            .iter()
            .find(|(buffer, _)| Arc::ptr_eq(buffer, &transmitter.buffer));
        let buffer_file = match existing {
            Some((_, name)) => name.clone(),
            None => {
                let name = format!("{}.{}.cf32", stem, buffer_files.len());
                write_cf32(&directory.join(&name), &transmitter.buffer)?;
                buffer_files.push((transmitter.buffer.clone(), name.clone()));
                name
            }
        };
        entries.push(CalibratorJson {
            position: calibrator.position,
            velocity: calibrator.velocity,
            acceleration: calibrator.acceleration,
            epoch: calibrator.epoch,
            transmitter: TransmitterJson {
                frequency: transmitter.frequency,
                sample_rate: transmitter.sample_rate,
                power: transmitter.power,
                bandwidth: transmitter.bandwidth,
                start_time: transmitter.start_time,
                buffer: buffer_file,
            },
        });
    }

    let file = std::fs::File::create(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to create file '{}': {}", filepath, e)))?;

    let writer = std::io::BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &entries)
        .map_err(|e| PyIOError::new_err(format!("Failed to serialize calibrators: {}", e)))?;

    Ok(())
}

/// Loads a list of calibrators from a JSON file.
///
/// # Arguments
/// - `filepath`: Path to the file containing a serialized list of `Calibrator`s.
///
/// # Returns
/// A reconstructed list of `Calibrator`s.
///
/// # Errors
/// Returns a Python `IOError` if a file cannot be read or parsed.
///
/// # Panics
/// Panics if, for a `Calibrator`:
/// - a kinematic parameter is not finite
/// - a transmitter parameter is invalid
#[pyfunction]
pub fn load_calibrators(filepath: &str) -> PyResult<Vec<Calibrator>> {
    let path = Path::new(filepath);
    let directory = path.parent().unwrap_or(Path::new(""));

    let file = std::fs::File::open(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open file '{}': {}", filepath, e)))?;

    let reader = std::io::BufReader::new(file);

    let entries: Vec<CalibratorJson> = serde_json::from_reader(reader)
        .map_err(|e| PyIOError::new_err(format!("Failed to deserialize calibrators: {}", e)))?;

    let mut buffers: Vec<(String, Arc<[Complex32]>)> = Vec::new();
    let mut calibrators = Vec::with_capacity(entries.len());
    for entry in entries {
        let name = entry.transmitter.buffer;
        let buffer = match buffers.iter().find(|(n, _)| *n == name) {
            Some((_, buffer)) => buffer.clone(),
            None => {
                let buffer: Arc<[Complex32]> = read_cf32(&directory.join(&name))?.into();
                buffers.push((name, buffer.clone()));
                buffer
            }
        };
        let transmitter = Transmitter {
            frequency: entry.transmitter.frequency,
            sample_rate: entry.transmitter.sample_rate,
            power: entry.transmitter.power,
            bandwidth: entry.transmitter.bandwidth,
            buffer,
            start_time: entry.transmitter.start_time,
        };
        let calibrator = Calibrator {
            position: entry.position,
            velocity: entry.velocity,
            acceleration: entry.acceleration,
            epoch: entry.epoch,
            transmitter,
        };
        calibrator.validate();
        calibrators.push(calibrator);
    }

    Ok(calibrators)
}

/// Writes samples to a raw interleaved little-endian 32-bit float IQ file.
///
/// # Arguments
/// - `path`: Destination path where the samples will be written.
/// - `samples`: The samples.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be created or written
fn write_cf32(path: &Path, samples: &[Complex32]) -> PyResult<()> {
    let bytes = samples
        .iter()
        .flat_map(|s| [s.re.to_le_bytes(), s.im.to_le_bytes()])
        .flatten()
        .collect::<Vec<u8>>();
    std::fs::write(path, bytes).map_err(|e| {
        PyIOError::new_err(format!("Failed to write file '{}': {}", path.display(), e))
    })
}

/// Reads samples from a raw interleaved little-endian 32-bit float IQ file.
///
/// # Arguments
/// - `path`: Path to the file containing the samples.
///
/// # Returns
/// The samples.
///
/// # Errors
/// Returns a Python `IOError` if the file cannot be read or its size is not a multiple of 8 bytes.
fn read_cf32(path: &Path) -> PyResult<Vec<Complex32>> {
    let bytes = std::fs::read(path).map_err(|e| {
        PyIOError::new_err(format!("Failed to read file '{}': {}", path.display(), e))
    })?;
    if bytes.len() % 8 != 0 {
        return Err(PyIOError::new_err(format!(
            "File '{}' is not a cf32 file: its size is not a multiple of 8 bytes",
            path.display()
        )));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            Complex32::new(
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            )
        })
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrators_round_trip() {
        let directory = std::env::temp_dir().join(format!("fringe-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("calibrators.json");

        let shared = Transmitter::new(
            75e6,
            20e6,
            2.5,
            vec![Complex32::new(1.0, -0.5), Complex32::new(-0.25, 0.75)],
            Some(15e6),
            0.5,
        );
        let calibrators = vec![
            Calibrator::new(
                Vec3::new(1.0, 2.0, 3.0),
                shared.clone(),
                Some(Vec3::new(4.0, 5.0, 6.0)),
                Some(Vec3::new(7.0, 8.0, 9.0)),
                10.0,
            ),
            Calibrator::new(Vec3::new(-1.0, 0.0, 1e5), shared, None, None, 0.0),
        ];
        save_calibrators(calibrators.clone(), path.to_str().unwrap()).unwrap();
        let loaded = load_calibrators(path.to_str().unwrap()).unwrap();

        // calibrators sharing a buffer share a buffer file
        assert!(directory.join("calibrators.0.cf32").exists());
        assert!(!directory.join("calibrators.1.cf32").exists());
        assert!(Arc::ptr_eq(
            &loaded[0].transmitter.buffer,
            &loaded[1].transmitter.buffer
        ));
        for (a, b) in calibrators.iter().zip(&loaded) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
            assert_eq!(a.acceleration, b.acceleration);
            assert_eq!(a.epoch, b.epoch);
            assert_eq!(a.transmitter.frequency, b.transmitter.frequency);
            assert_eq!(a.transmitter.sample_rate, b.transmitter.sample_rate);
            assert_eq!(a.transmitter.power, b.transmitter.power);
            assert_eq!(a.transmitter.bandwidth, b.transmitter.bandwidth);
            assert_eq!(a.transmitter.start_time, b.transmitter.start_time);
            assert_eq!(a.transmitter.buffer, b.transmitter.buffer);
        }

        std::fs::remove_dir_all(directory).unwrap();
    }
}
