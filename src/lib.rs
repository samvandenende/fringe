use pyo3::prelude::*;
mod library;
pub use library::{
    Array, Attitude, Beam, Calibrator, DEFAULT_FREQUENCY_RESOLUTION, Error, NoiseTemperature,
    Orientation, Simulation, Source, Transmitter, Vec3, load_array, load_beam, load_calibrators,
    load_sources, save_array, save_beam, save_calibrators, save_sources,
};

#[pymodule]
mod fringe {
    #[pymodule_export]
    use super::{
        Array, Attitude, Beam, Calibrator, Orientation, Simulation, Source, Transmitter, Vec3,
        load_array, load_beam, load_calibrators, load_sources, save_array, save_beam,
        save_calibrators, save_sources,
    };
}
