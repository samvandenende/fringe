use std::sync::Arc;

pub use beam::{Attitude, Beam, Orientation};
pub use beam_io::{load_beam, save_beam};
use calibrator::{CalibratorWindow, ConditionedBuffer};
pub use error::Error;
pub use model::*;
use num_complex::Complex32;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use response::{ArrayResponse, SkyTable};
pub use utils::*;

mod beam;
mod beam_io;
mod calibrator;
mod cpu;
mod error;
mod gpu;
mod model;
mod response;
mod utils;

pub const DEFAULT_FREQUENCY_RESOLUTION: usize = 4;

/// Simulation for antenna signal generation.
#[pyclass]
pub struct Simulation {
    rng: ChaCha8Rng,
    runtime: Runtime,
    array: Array,
    response: Arc<ArrayResponse>,
    sources: Vec<Source>,
    sky: Arc<SkyTable>,
    calibrators: Vec<Calibrator>,
    conditioned_buffers: Vec<Arc<ConditionedBuffer>>,
    random_phases: Option<Phases>,
    frequency_resolution: usize,
    time: f64,
}

#[pymethods]
impl Simulation {
    /// Creates a new simulation.
    ///
    /// # Arguments
    /// - `runtime`: Type of execution backend. Must be `"cpu"` or `"gpu"`.
    /// - `array`: Antenna array configuration.
    /// - `frequency_resolution` - FFT oversampling factor.
    /// - `rng_seed`: Optional RNG seed for reproducible phase generation.
    ///
    /// # Returns
    /// A simulation ready for configuration.
    #[new]
    #[pyo3(signature = (runtime, array, frequency_resolution = DEFAULT_FREQUENCY_RESOLUTION, rng_seed = None))]
    pub fn new(
        runtime: String,
        array: Array,
        frequency_resolution: usize,
        rng_seed: Option<u64>,
    ) -> Self {
        assert!(
            frequency_resolution.is_power_of_two(),
            "frequency resolution must be a power of 2"
        );

        let response = Arc::new(ArrayResponse::new(
            &array,
            array.sample_window_size * frequency_resolution,
        ));
        let sky = Arc::new(SkyTable::new(&response, &[]));
        let runtime = Runtime::new(&runtime, &array, &response, frequency_resolution);

        let seed = rng_seed.unwrap_or_else(|| {
            let mut rng = rand::rng();
            rng.random()
        });

        let rng = ChaCha8Rng::seed_from_u64(seed);

        Simulation {
            rng,
            runtime,
            array,
            response,
            sources: Vec::new(),
            sky,
            calibrators: Vec::new(),
            conditioned_buffers: Vec::new(),
            random_phases: None,
            frequency_resolution,
            time: 0.0,
        }
    }

    /// Set or update the source list used in the simulation.
    ///
    /// These sources will be used on the next call to `Simulation::start`
    ///
    /// The response of every receiver to every source is tabulated here, so calling this
    /// takes a while for large sky models with tabulated beams.
    ///
    /// # Arguments
    /// - `sources`: List of `Source` objects defining the sky model.
    pub fn set_sources(&mut self, sources: Vec<Source>) {
        self.sky = Arc::new(SkyTable::new(&self.response, &sources));
        self.sources = sources;
    }

    /// Set or update the calibrators used in the simulation.
    ///
    /// These calibrators will be used on the next call to `Simulation::start`.
    /// Transmit buffers that did not change since the previous call are reused
    /// without being processed again.
    ///
    /// # Arguments
    /// - `calibrators`: List of `Calibrator` objects.
    pub fn set_calibrators(&mut self, calibrators: Vec<Calibrator>) {
        self.conditioned_buffers = calibrators
            .iter()
            .map(|calibrator| {
                self.conditioned_buffers
                    .iter()
                    .find(|buffer| buffer.matches(&calibrator.transmitter))
                    .cloned()
                    .unwrap_or_else(|| {
                        Arc::new(ConditionedBuffer::new(&self.array, &calibrator.transmitter))
                    })
            })
            .collect();
        self.calibrators = calibrators;
    }

    /// Returns the time (s) at which the next sample window starts if
    /// `Simulation::start` is called without a time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Start simulation of a batch of time-domain signals.
    ///
    /// The simulation work is dispatched to the configured runtime.
    /// `Simulation::finish` must be called to obtain the results before
    /// a next call to `Simulation::start`.
    ///
    /// # Arguments
    /// - `time`: Time (s) at which the first sample of the window is received.
    ///   If not given, the window directly follows the previous one.
    ///
    /// # Panics
    /// Panics if time is not finite.
    #[pyo3(signature = (time = None))]
    pub fn start(&mut self, time: Option<f64>) {
        let start_time = time.unwrap_or(self.time);
        assert!(start_time.is_finite(), "time must be finite");
        self.time = start_time + self.array.sample_window_size as f64 / self.array.sample_frequency;

        let calibrators = (!self.calibrators.is_empty()).then(|| {
            Arc::new(CalibratorWindow::new(
                &self.array,
                &self.response,
                &self.calibrators,
                &self.conditioned_buffers,
                self.array.sample_window_size * self.frequency_resolution,
                start_time,
            ))
        });

        if let Some(phases) = &mut self.random_phases {
            phases.update(
                &mut self.rng,
                &self.array,
                self.sources.len(),
                self.frequency_resolution,
            );
        } else {
            self.random_phases = Some(Phases::new(
                &mut self.rng,
                &self.array,
                self.sources.len(),
                self.frequency_resolution,
                self.response.source_polarizations(),
            ));
        }

        self.runtime.start(
            &self.array,
            &self.response,
            &self.sources,
            &self.sky,
            calibrators.as_ref(),
            self.random_phases.as_ref().unwrap(),
        )
    }

    /// Collects simulation results from the runtime.
    ///
    /// Must be called after `Simulation::start`.
    ///
    /// # Returns
    /// A 2D array of single precision complex-valued antenna samples, of shape
    /// (antennas, samples). Cast it to `np.complex128` if computations on it need double
    /// precision.
    #[pyo3(name = "finish")]
    fn finish_array<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<Complex32>> {
        let samples = self.finish();
        let window_size = self.array.sample_window_size;
        let array = PyArray2::zeros(py, [samples.len(), window_size], false);
        // SAFETY: the array was just created, nothing else refers to it
        let data = unsafe { array.as_slice_mut() }.expect("a new array is contiguous");
        for (row, samples) in data.chunks_exact_mut(window_size.max(1)).zip(&samples) {
            row.copy_from_slice(samples);
        }
        array
    }
}

impl Simulation {
    /// Collects simulation results from the runtime.
    ///
    /// Must be called after `Simulation::start`.
    ///
    /// # Returns
    /// A 2D vector of complex-valued antenna samples:
    /// - Outer dimension: antennas in the array
    /// - Inner dimension: time-domain samples
    pub fn finish(&mut self) -> Vec<Vec<Complex32>> {
        self.runtime.finish()
    }
}

/// A simulation runtime.
///
/// The runtime is either a multithreaded CPU runtime or a GPU-accelerated runtime.
enum Runtime {
    Cpu(Box<cpu::Runtime>),
    Gpu(Box<gpu::Runtime>),
}

impl Runtime {
    /// Creates a new runtime of the desired type.
    ///
    /// # Arguments
    /// - `runtime`: The desired runtime, can be either `"cpu"` or `"gpu"`.
    /// - `array`: Antenna array configuration used to size internal resources.
    /// - `response`: Receive response of the array.
    /// - `frequency_resolution`: Number of frequency bins used per sample window.
    ///
    /// # Returns
    /// A newly initialized `Runtime`.
    fn new(
        runtime: &str,
        array: &Array,
        response: &ArrayResponse,
        frequency_resolution: usize,
    ) -> Self {
        match runtime.trim().to_lowercase().as_str() {
            "cpu" => Runtime::Cpu(Box::new(cpu::Runtime::new(frequency_resolution))),
            "gpu" => Runtime::Gpu(Box::new(gpu::Runtime::new(
                array,
                response,
                frequency_resolution,
            ))),
            _ => panic!("Runtime must be one of [cpu, gpu]"),
        }
    }

    /// Start the simulation
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `response`: Receive response of the array.
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `sky`: Response of the receivers to the sources.
    /// - `calibrators`: Precomputed calibrator signals, if there are any calibrators.
    /// - `phases`: Precomputed phase information for system noise and sources.
    fn start(
        &mut self,
        array: &Array,
        response: &Arc<ArrayResponse>,
        sources: &[Source],
        sky: &Arc<SkyTable>,
        calibrators: Option<&Arc<CalibratorWindow>>,
        phases: &Phases,
    ) {
        match self {
            Runtime::Cpu(runtime) => {
                runtime.start(array, response, sources, sky, calibrators, phases)
            }
            Runtime::Gpu(runtime) => {
                runtime.start(array, response, sources, sky, calibrators, phases)
            }
        }
    }

    /// Finish the running simulation.
    /// Blocks until the runtime has finished.
    ///
    /// # Returns
    /// Simulated time-domain antenna data.
    fn finish(&mut self) -> Vec<Vec<Complex32>> {
        match self {
            Runtime::Cpu(runtime) => runtime.finish(),
            Runtime::Gpu(runtime) => runtime.finish(),
        }
    }
}
