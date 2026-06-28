use std::path::Path;

use clap::Parser;
use fringe::{self, Simulation, load_array, load_calibrator, load_sources};
use hdf5::{File, Result};
use num_complex::Complex32;

fn parse_runtime(s: &str) -> Result<String, String> {
    let normalized = s.trim().to_ascii_lowercase();

    match normalized.as_str() {
        "cpu" | "gpu" => Ok(normalized),
        _ => Err(format!("invalid runtime '{s}', expected 'cpu' or 'gpu'")),
    }
}

#[derive(Parser, Debug)]
struct Input {
    /// Execution backend: "cpu" or "gpu"
    #[arg(
        long,
        short,
        default_value = "cpu",
        value_parser = parse_runtime
    )]
    runtime: String,

    /// Frequency resolution (must be >= 1)
    #[arg(
        long, short, default_value_t = fringe::DEFAULT_FREQUENCY_RESOLUTION,
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    frequency_resolution: usize,

    /// The seed used for random number generation. When provided, the simulation will be reproducible. Otherwise a random seed will be used.
    #[arg(long, short = 'n')]
    rng_seed: Option<u64>,

    /// The array model. File must be in JSON format.
    #[arg(short, long, value_name = "FILE")]
    array: String,

    /// The sources comprising the sky model to be simulated. File must be in CSV format.
    #[arg(short, long, value_name = "FILE")]
    sources: String,

    /// The calibrator model. File must be in JSON format.
    #[arg(short, long, value_name = "FILE")]
    calibrator: Option<String>,

    /// File to write output to. Must be in HDF5 format if it exists. Otherwise a new file will be created.
    #[arg(short, long, value_name = "FILE")]
    output_file: String,

    /// Name of dataset to use for the output
    #[arg(short = 'd', long, value_name = "STRING")]
    output_dataset: String,
}

fn main() {
    let input = Input::parse();

    let array = load_array(&input.array).expect("Failed to load array");
    let sources = load_sources(&input.sources).expect("Failed to load sources");
    let calibrator = input
        .calibrator
        .as_ref()
        .map(|path| load_calibrator(path).expect("Failed to load calibrator"));

    let mut simulation = Simulation::new(
        input.runtime,
        array,
        input.frequency_resolution,
        input.rng_seed,
    );

    simulation.set_sources(sources);
    if let Some(calibrator) = calibrator {
        simulation.set_calibrator(calibrator);
    }

    simulation.start();
    let result = simulation.finish();

    let outpath = Path::new(&input.output_file);
    let output = if outpath.exists() {
        File::open_rw(outpath).expect("Failed to open output file")
    } else {
        File::create(outpath).expect("Failed to create output file")
    };

    let rows = result.len();
    let cols = result[0].len();
    let flat = result.into_iter().flatten().collect::<Vec<_>>();

    let dataset = output
        .new_dataset::<Complex32>()
        .shape((rows, cols))
        .create(input.output_dataset.as_str())
        .expect("Failed to create output dataset");

    dataset.write(&flat).expect("Failed to write output");
}
