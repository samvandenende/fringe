//! The Rust API, used without a Python interpreter.

use fringe::{
    Array, Beam, Calibrator, NoiseTemperature, Orientation, Simulation, Source, Transmitter, Vec3,
    load_array, load_beam, load_calibrators, load_sources, save_array, save_calibrators,
    save_sources,
};
use num_complex::Complex32;

fn positions() -> Vec<Vec3> {
    (0..8)
        .map(|i| Vec3::new(3.0 * (i % 4) as f64, 3.0 * (i / 4) as f64, 0.0))
        .collect()
}

fn source(ra: f64, dec: f64) -> Source {
    Source::new(
        Vec3::from_ra_dec(ra, dec),
        50e6,
        1e3,
        -0.7,
        0.0,
        0.0,
        0.2,
        0.3,
        0.0,
        0.0,
    )
}

fn calibrator() -> Calibrator {
    let buffer = (0..64)
        .map(|i| Complex32::from_polar(1.0, i as f32 * 0.7))
        .collect();
    let transmitter = Transmitter::new(45e6, 10e6, 1.0, buffer, None, 0.0, None);
    Calibrator::new(Vec3::new(1e3, 2e3, 5e4), transmitter, None, None, 0.0, None)
}

/// Scratch directory of a test, removed when it is dropped.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("fringe-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        ScratchDir(path)
    }

    fn file(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn simulates_without_python() {
    let noise = NoiseTemperature::Table(vec![[30e6, 400.0], [70e6, 200.0]]);
    let array = Array::new(
        positions(),
        20e6,
        40e6,
        2e6,
        8e6,
        256,
        noise.clone(),
        None,
        None,
        None,
        None,
    );
    assert_eq!(array.system_noise_temperature(), noise);
    assert_eq!(array.system_noise_temperature_at(50e6), 300.0);

    let mut simulation = Simulation::new("cpu".to_owned(), array, 4, Some(1));
    simulation.set_sources(vec![source(0.1, 1.2), source(2.0, 0.9)]);
    simulation.set_calibrators(vec![calibrator()]);
    simulation.start(Some(0.0));
    let samples = simulation.finish();

    assert_eq!(samples.len(), 8);
    assert!(samples.iter().all(|s| s.len() == 256));
    assert!(samples.iter().flatten().all(|v| v.is_finite()));
    assert!(samples.iter().flatten().any(|v| v.norm() > 0.0));
}

#[test]
fn dual_polarized_array_with_beam() {
    let orientations = vec![Orientation::identity(); 8];
    let array = Array::dual_polarized(
        positions(),
        20e6,
        40e6,
        2e6,
        8e6,
        256,
        NoiseTemperature::Constant(100.0),
        Some(Beam::short_dipole()),
        Some(orientations),
        Some(Vec3::new(0.0, 0.0, 1.0)),
    );
    assert_eq!(array.antenna_positions().len(), 16);

    let mut simulation = Simulation::new("cpu".to_owned(), array, 4, Some(2));
    simulation.set_sources(vec![source(0.5, 1.0)]);
    simulation.start(None);
    assert_eq!(simulation.finish().len(), 16);
}

#[test]
fn saves_and_loads_files() {
    let dir = ScratchDir::new("rust-api");
    let noise = NoiseTemperature::Table(vec![[30e6, 400.0], [70e6, 200.0]]);
    let array = Array::new(
        positions(),
        20e6,
        40e6,
        2e6,
        8e6,
        256,
        noise.clone(),
        None,
        None,
        None,
        None,
    );
    save_array(&array, &dir.file("array.json")).unwrap();
    let loaded = load_array(&dir.file("array.json")).unwrap();
    assert_eq!(loaded.antenna_positions(), array.antenna_positions());
    assert_eq!(loaded.system_noise_temperature(), noise);

    save_sources(vec![source(0.1, 1.2)], &dir.file("sky.csv")).unwrap();
    assert_eq!(load_sources(&dir.file("sky.csv")).unwrap().len(), 1);

    save_calibrators(vec![calibrator()], &dir.file("calibrators.json")).unwrap();
    let calibrators = load_calibrators(&dir.file("calibrators.json")).unwrap();
    assert_eq!(calibrators[0].transmitter().buffer().len(), 64);
}

#[test]
fn errors_are_readable_without_python() {
    let dir = ScratchDir::new("rust-api-errors");
    let missing = dir.file("missing.json");

    let Err(error) = load_array(&missing) else {
        panic!("loaded a missing file");
    };
    assert!(error.to_string().contains("missing.json"), "{error}");
    assert!(format!("{error:?}").contains("missing.json"));

    let Err(error) = load_beam(&missing) else {
        panic!("loaded a missing file");
    };
    assert!(error.message().contains("missing.json"), "{error}");
    let error: Box<dyn std::error::Error> = Box::new(error);
    assert!(!error.to_string().is_empty());

    assert!(Beam::load_openems(&missing, None, None).is_err());
}
