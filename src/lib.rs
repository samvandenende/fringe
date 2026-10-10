use pyo3::prelude::*;
mod library;
pub use library::{
    Array, Calibrator, DEFAULT_FREQUENCY_RESOLUTION, Simulation, Source, Transmitter, Vec3,
    load_array, load_calibrators, load_sources, save_array, save_calibrators, save_sources,
};

#[pymodule]
mod fringe {
    #[pymodule_export]
    use super::{
        Array, Calibrator, Simulation, Source, Transmitter, Vec3, load_array, load_calibrators,
        load_sources, save_array, save_calibrators, save_sources,
    };
}
