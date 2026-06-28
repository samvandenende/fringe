use pyo3::prelude::*;
mod library;
pub use library::{
    Array, Calibrator, DEFAULT_FREQUENCY_RESOLUTION, Simulation, Source, Vec3, load_array,
    load_calibrator, load_sources, save_array, save_calibrator, save_sources,
};

#[pymodule]
mod fringe {
    #[pymodule_export]
    use super::{
        Array, Calibrator, Simulation, Source, Vec3, load_array, load_calibrator, load_sources,
        save_array, save_calibrator, save_sources,
    };
}
