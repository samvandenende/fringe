pub use model::*;
use num_complex::Complex32;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
pub use utils::*;

mod cpu;
mod gpu;
mod model;
mod utils;

pub const DEFAULT_FREQUENCY_RESOLUTION: usize = 4;

/// Simulation for antenna signal generation.
#[pyclass]
pub struct Simulation {
    rng: ChaCha8Rng,
    runtime: Runtime,
    array: Array,
    sources: Vec<Source>,
    calibrator: Option<Calibrator>,
    random_phases: Option<Phases>,
    frequency_resolution: usize,
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

        let runtime = Runtime::new(&runtime, &array, frequency_resolution);

        let seed = rng_seed.unwrap_or_else(|| {
            let mut rng = rand::rng();
            rng.random()
        });

        let rng = ChaCha8Rng::seed_from_u64(seed);

        Simulation {
            rng,
            runtime,
            array,
            sources: Vec::new(),
            calibrator: None,
            random_phases: None,
            frequency_resolution,
        }
    }

    /// Set or update the source list used in the simulation.
    ///
    /// These sources will be used on the next call to `Simulation::start`
    ///
    /// # Arguments
    /// - `sources`: List of `Source` objects defining the sky model.
    pub fn set_sources(&mut self, sources: Vec<Source>) {
        self.sources = sources;
    }

    /// Set or update the calibrator source.
    ///
    /// This calibrator will be used on the next call to `Simulation::start`
    ///
    /// # Arguments
    /// - `calibrator`: `Calibrator` object with position and intensity.
    pub fn set_calibrator(&mut self, calibrator: Calibrator) {
        self.calibrator = Some(calibrator);
    }

    /// Invalidates and regenerates the calibrator signal phase model on next run.
    pub fn regenerate_calibrator_signal(&mut self) {
        self.random_phases = None;
    }

    /// Start simulation of a batch of time-domain signals.
    ///
    /// The simulation work is dispatched to the configured runtime.
    /// `Simulation::finish` must be called to obtain the results before
    /// a next call to `Simulation::start`.
    pub fn start(&mut self) {
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
            ));
        }

        self.runtime.start(
            &self.array,
            &self.sources,
            self.calibrator.as_ref(),
            self.random_phases.as_ref().unwrap(),
        )
    }

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

    pub fn calibrator_frequency_domain_signal(&self) -> Vec<Complex32> {
        todo!()
    }
}

/// A simulation runtime.
///
/// The runtime is either a multithreaded CPU runtime or a GPU-accelerated runtime.
enum Runtime {
    Cpu(cpu::Runtime),
    Gpu(gpu::Runtime),
}

impl Runtime {
    /// Creates a new runtime of the desired type.
    ///
    /// # Arguments
    /// - `runtime`: The desired runtime, can be either `"cpu"` or `"gpu"`.
    /// - `array`: Antenna array configuration used to size internal resources.
    /// - `frequency_resolution`: Number of frequency bins used per sample window.
    ///
    /// # Returns
    /// A newly initialized `Runtime`.
    fn new(runtime: &str, array: &Array, frequency_resolution: usize) -> Self {
        match runtime.trim().to_lowercase().as_str() {
            "cpu" => Runtime::Cpu(cpu::Runtime::new(frequency_resolution)),
            "gpu" => Runtime::Gpu(gpu::Runtime::new(array, frequency_resolution)),
            _ => panic!("Runtime must be one of [cpu, gpu]"),
        }
    }

    /// Start the simulation
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrator`: Optional external calibrator.
    /// - `phases`: Precomputed phase information for system noise, sources,
    ///   and calibration signals.
    fn start(
        &mut self,
        array: &Array,
        sources: &[Source],
        calibrator: Option<&Calibrator>,
        phases: &Phases,
    ) {
        match self {
            Runtime::Cpu(runtime) => runtime.start(array, sources, calibrator, phases),
            Runtime::Gpu(runtime) => runtime.start(array, sources, calibrator, phases),
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
