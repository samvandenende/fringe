#![allow(clippy::too_many_arguments)]

use num_complex::{Complex32, Complex64};
use rustfft::{Fft, FftPlanner};
use std::{
    f64::consts::TAU,
    sync::Arc,
    thread::{self, JoinHandle},
};

use crate::library::normalize_and_truncate;

use super::{Array, Calibrator, Phases, Source, Vec3, utils::fft_bin_frequency};

/// Number of worker threads used to parallelize signal simulation.
const NUM_CPUS: usize = 10;
/// Speed of light in vacuum (m/s).
const C: f64 = 299792458.0;

type WorkerHandle = Option<JoinHandle<Vec<Vec<Complex32>>>>;

/// CPU runtime for parallel generation of simulated antenna data.
pub(crate) struct Runtime {
    frequency_resolution: usize,
    fft_planner: FftPlanner<f32>,
    workers: [WorkerHandle; NUM_CPUS],
}

impl Runtime {
    /// Creates a new runtime instance.
    ///
    /// # Arguments
    /// - `frequency_resolution` - FFT oversampling factor.
    ///
    /// # Returns
    /// A newly initialized `Runtime`.
    pub(crate) fn new(frequency_resolution: usize) -> Self {
        Runtime {
            frequency_resolution,
            fft_planner: FftPlanner::new(),
            workers: [const { None }; NUM_CPUS],
        }
    }

    /// Start the simulation.
    ///
    /// Spawns worker threads to generate simulated antenna data in parallel.
    /// The antenna array is partitioned evenly across available CPU cores.
    /// Each worker processes a contiguous subset of antennas.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrator`: Optional external calibrator.
    /// - `phases`: Precomputed phase information for system noise, sources,
    ///   and calibration signals.
    pub(crate) fn start(
        &mut self,
        array: &Array,
        sources: &[Source],
        calibrator: Option<&Calibrator>,
        phases: &Phases,
    ) {
        let antennas_per_thread = array.antenna_positions.len().div_ceil(NUM_CPUS);
        for i in 0..NUM_CPUS {
            let start = i * antennas_per_thread;
            let end = ((i + 1) * antennas_per_thread).min(array.antenna_positions.len());
            if start >= end {
                break;
            }

            let spectrum_size = array.sample_window_size * self.frequency_resolution;
            let antenna_positions = array.antenna_positions[start..end].to_vec();
            let antennas_system_noise_phases =
                phases.system_noise[(start * spectrum_size)..(end * spectrum_size)].to_vec();
            let calibrator = calibrator
                .cloned()
                .unwrap_or(Calibrator::new(Vec3::new(0.0, 0.0, 1.0), 0.0));
            self.workers[i] = spawn_worker(
                antenna_positions,
                array.sample_frequency,
                array.downmix_frequency,
                array.bandpass,
                array.sample_window_size,
                array.system_noise_intensity,
                self.frequency_resolution,
                sources.to_vec(),
                calibrator,
                antennas_system_noise_phases,
                phases.sources.clone(),
                phases.calibrator_signal.clone(),
                &mut self.fft_planner,
            );
        }
    }

    /// Finish the running simulation.
    ///
    /// Finalizes all active workers and collects their results. This method
    /// blocks until all worker threads have completed execution.
    ///
    /// # Returns
    /// Simulated time-domain antenna data.
    pub(crate) fn finish(&mut self) -> Vec<Vec<Complex32>> {
        let mut output = Vec::new();
        for worker in self.workers.iter_mut() {
            let Some(handle) = worker.take() else {
                return output;
            };
            let data = handle.join().unwrap();
            output.extend_from_slice(&data);
        }
        output
    }
}

/// Spawns a worker thread that simulates a subset of the array's antennas.
///
/// # Arguments
///
/// * `antenna_positions` - Positions of the antennas assigned to this worker.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `downmix_frequency` - Frequency used to downconvert the RF signal.
/// * `bandpass` - Simulated frequency range in Hz.
/// * `sample_window_size` - Number of output samples per antenna.
/// * `system_noise_intensity` - Receiver system noise spectral density.
/// * `frequency_resolution` - FFT oversampling factor.
/// * `sources` - Astronomical sources to simulate.
/// * `calibrator` - Calibration source.
/// * `antennas_system_noise_phases` - Per-antenna receiver noise phases.
/// * `source_phases` - Precomputed random phases for the astronomical sources.
/// * `calibrator_signal_phases` - Precomputed random phases for the calibrator.
/// * `fft_planner` - FFT planner used to construct the inverse FFT.
///
/// # Returns
///
/// A handle to the spawned worker thread.
fn spawn_worker(
    antenna_positions: Vec<Vec3>,
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass: [f64; 2],
    sample_window_size: usize,
    system_noise_intensity: f64,
    frequency_resolution: usize,
    sources: Vec<Source>,
    calibrator: Calibrator,
    antennas_system_noise_phases: Vec<f32>,
    source_phases: Arc<Vec<f32>>,
    calibrator_signal_phases: Arc<Vec<f32>>,
    fft_planner: &mut FftPlanner<f32>,
) -> WorkerHandle {
    let fft = fft_planner.plan_fft_inverse(sample_window_size * frequency_resolution);
    Some(thread::spawn(move || {
        antennas_system_noise_phases
            .chunks(sample_window_size * frequency_resolution)
            .zip(antenna_positions)
            .map(|(antenna_system_noise_phases, position)| {
                simulate_sample_window(
                    position,
                    sample_frequency,
                    downmix_frequency,
                    bandpass,
                    sample_window_size,
                    system_noise_intensity,
                    frequency_resolution,
                    &sources,
                    &calibrator,
                    antenna_system_noise_phases,
                    &source_phases,
                    &calibrator_signal_phases,
                    &fft,
                )
            })
            .collect::<Vec<_>>()
    }))
}

/// Simulates one time-domain sample window for a single antenna.
///
/// The signal is first synthesized in the frequency domain, transformed to
/// the time domain using an inverse FFT, and truncated to the requested
/// sample window length.
///
/// # Arguments
///
/// * `antenna_position` - Cartesian position of the antenna.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `downmix_frequency` - Frequency used to downconvert the RF signal to baseband.
/// * `bandpass` - Simulated frequency range in Hz.
/// * `sample_window_size` - Number of output time-domain samples.
/// * `system_noise_intensity` - Receiver system noise spectral density.
/// * `frequency_resolution` - FFT oversampling factor.
/// * `sources` - Astronomical sources to simulate.
/// * `calibrator` - Optional calibration source.
/// * `antenna_system_noise_phases` - Random phases for receiver noise.
/// * `source_phases` - Precomputed random phases for all simulated sources.
/// * `calibrator_signal_phases` - Precomputed random phases for the calibrator.
/// * `fft` - Inverse FFT plan used to transform the synthesized spectrum.
///
/// # Returns
///
/// A vector containing the simulated complex///
/// Each antenna is simulated independently and converted from the frequency
/// domain to the time domain using a shared inverse FFT plan. time-domain samples.
fn simulate_sample_window(
    antenna_position: Vec3,
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass: [f64; 2],
    sample_window_size: usize,
    system_noise_intensity: f64,
    frequency_resolution: usize,
    sources: &[Source],
    calibrator: &Calibrator,
    antenna_system_noise_phases: &[f32],
    source_phases: &[f32],
    calibrator_signal_phases: &[f32],
    fft: &Arc<dyn Fft<f32>>,
) -> Vec<Complex32> {
    let spectrum_size = sample_window_size * frequency_resolution;
    let mut simulated_spectrum = simulate_spectrum(
        antenna_position,
        sample_frequency,
        downmix_frequency,
        bandpass,
        spectrum_size,
        system_noise_intensity,
        sources,
        calibrator,
        antenna_system_noise_phases,
        source_phases,
        calibrator_signal_phases,
    );
    fft.process(&mut simulated_spectrum); // in-place FFT, simulated_spectrum now contains time domain samples
    normalize_and_truncate(&simulated_spectrum, sample_window_size)
}

/// Synthesizes the frequency-domain spectrum for a single antenna.
///
/// The synthesized spectrum is the sum of receiver system noise, the optional
/// calibration source, and all sky source contributions.
///
/// # Arguments
///
/// * `antenna_position` - Cartesian position of the antenna.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `downmix_frequency` - Frequency used to downconvert the RF signal to baseband.
/// * `bandpass` - Simulated frequency range in Hz.
/// * `spectrum_size` - Number of frequency bins to synthesize.
/// * `system_noise_intensity` - Receiver system noise spectral density.
/// * `sources` - Astronomical sources to simulate.
/// * `calibrator` - Calibration source.
/// * `antenna_system_noise_phases` - Random phases for receiver noise.
/// * `source_phases` - Precomputed random phases for all simulated sources.
/// * `calibrator_signal_phases` - Precomputed random phases for the calibrator.
///
/// # Returns
///
/// A synthesized complex spectrum suitable for inverse FFT processing.
fn simulate_spectrum(
    antenna_position: Vec3,
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass: [f64; 2],
    spectrum_size: usize,
    system_noise_intensity: f64,
    sources: &[Source],
    calibrator: &Calibrator,
    antenna_system_noise_phases: &[f32],
    source_phases: &[f32],
    calibrator_signal_phases: &[f32],
) -> Vec<Complex32> {
    let mut spectrum = vec![Complex64::ZERO; spectrum_size];
    let bandpass_index_min =
        (bandpass[0] * spectrum_size as f64 / sample_frequency).ceil() as usize;
    let bandpass_index_max =
        (bandpass[1] * spectrum_size as f64 / sample_frequency).floor() as usize;

    simulate_spectrum_system_noise_contribution(
        &mut spectrum,
        sample_frequency,
        downmix_frequency,
        bandpass_index_min,
        bandpass_index_max,
        spectrum_size,
        system_noise_intensity,
        antenna_system_noise_phases,
    );

    simulate_spectrum_calibrator_contribution(
        &mut spectrum,
        antenna_position,
        sample_frequency,
        bandpass_index_min,
        bandpass_index_max,
        spectrum_size,
        calibrator,
        calibrator_signal_phases,
    );

    simulate_spectrum_source_contributions(
        &mut spectrum,
        antenna_position,
        sample_frequency,
        downmix_frequency,
        bandpass_index_min,
        bandpass_index_max,
        spectrum_size,
        sources,
        source_phases,
    );

    spectrum
        .iter()
        .map(|v| Complex32::new(v.re as _, v.im as _))
        .collect()
}

/// Adds the receiver system noise contribution to a synthesized spectrum.
///
/// Noise is generated independently for each frequency bin within the
/// configured bandpass. The effective aperture area is approximated as
/// λ²/4 and capped at 4 m².
///
/// # Arguments
///
/// * `spectrum` - Spectrum to modify in place.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `downmix_frequency` - Frequency used to downconvert the RF signal.
/// * `bandpass_index_min` - First frequency bin included in the bandpass.
/// * `bandpass_index_max` - Last frequency bin included in the bandpass.
/// * `spectrum_size` - Total number of frequency bins.
/// * `system_noise_intensity` - Receiver system noise spectral density.
/// * `antenna_system_noise_phases` - Random phase assigned to each frequency bin.
fn simulate_spectrum_system_noise_contribution(
    spectrum: &mut [Complex64],
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass_index_min: usize,
    bandpass_index_max: usize,
    spectrum_size: usize,
    system_noise_intensity: f64,
    antenna_system_noise_phases: &[f32],
) {
    for ((index, spectrum_value), phase) in spectrum
        .iter_mut()
        .enumerate()
        .zip(antenna_system_noise_phases)
        .take(bandpass_index_max + 1)
        .skip(bandpass_index_min)
    {
        let bin_frequency = fft_bin_frequency(spectrum_size, sample_frequency, index);
        let phys_frequency = bin_frequency + downmix_frequency;
        let lambda = C / phys_frequency;
        let effective_aperture = (lambda * lambda / 4.0).min(4.0);
        let sqrt_amplitude = (system_noise_intensity / effective_aperture).sqrt();
        let phase = *phase as f64;
        *spectrum_value += Complex64::from_polar(sqrt_amplitude, phase);
    }
}

/// Adds the calibration source contribution to a synthesized spectrum.
///
/// The calibrator is modeled as a point source with free-space propagation,
/// geometric delay, and a uniform gain pattern.
///
/// # Arguments
///
/// * `spectrum` - Spectrum to modify in place.
/// * `antenna_position` - Cartesian position of the antenna.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `bandpass_index_min` - First frequency bin included in the bandpass.
/// * `bandpass_index_max` - Last frequency bin included in the bandpass.
/// * `spectrum_size` - Total number of frequency bins.
/// * `calibrator` - Calibration source.
/// * `calibrator_signal_phases` - Random phase assigned to each frequency bin.
fn simulate_spectrum_calibrator_contribution(
    spectrum: &mut [Complex64],
    antenna_position: Vec3,
    sample_frequency: f64,
    bandpass_index_min: usize,
    bandpass_index_max: usize,
    spectrum_size: usize,
    calibrator: &Calibrator,
    calibrator_signal_phases: &[f32],
) {
    let calibrator_direction = calibrator.position - antenna_position;
    let calibrator_distance = calibrator_direction.norm();
    let calibrator_direction_z = (calibrator_direction / calibrator_distance).z;
    let calibrator_gain = calibrator_direction_z * calibrator_direction_z;
    let sqrt_amplitude =
        (calibrator_gain * calibrator.intensity / (2.0 * TAU)).sqrt() / calibrator_distance;
    let delay = calibrator_distance / C;
    for ((index, spectrum_value), phase) in spectrum
        .iter_mut()
        .enumerate()
        .zip(calibrator_signal_phases)
        .take(bandpass_index_max + 1)
        .skip(bandpass_index_min)
    {
        let bin_frequency = fft_bin_frequency(spectrum_size, sample_frequency, index);
        let phase = *phase as f64 + (TAU * bin_frequency * delay);
        *spectrum_value += Complex64::from_polar(sqrt_amplitude, phase);
    }
}

/// Adds the contributions from all astronomical sources to a synthesized spectrum.
///
/// Each source is modeled using its reference intensity, spectral index and
/// geometric delay.
///
/// # Arguments
///
/// * `spectrum` - Spectrum to modify in place.
/// * `antenna_position` - Cartesian position of the antenna.
/// * `sample_frequency` - ADC sample frequency in Hz.
/// * `downmix_frequency` - Frequency used to downconvert the RF signal.
/// * `bandpass_index_min` - First frequency bin included in the bandpass.
/// * `bandpass_index_max` - Last frequency bin included in the bandpass.
/// * `spectrum_size` - Total number of frequency bins.
/// * `sources` - Astronomical sources to simulate.
/// * `source_phases` - Precomputed random phases for every source and frequency bin.
fn simulate_spectrum_source_contributions(
    spectrum: &mut [Complex64],
    antenna_position: Vec3,
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass_index_min: usize,
    bandpass_index_max: usize,
    spectrum_size: usize,
    sources: &[Source],
    source_phases: &[f32],
) {
    let source_geometric_delays = sources
        .iter()
        .map(|s| s.direction.dot(antenna_position) / C)
        .collect::<Vec<f64>>();

    for ((source_phases, source), delay) in source_phases
        .chunks(spectrum_size)
        .zip(sources)
        .zip(source_geometric_delays)
    {
        let source_z = source.direction.normalized().z;
        let source_gain = source_z * source_z;
        for ((index, spectrum_value), phase) in spectrum
            .iter_mut()
            .enumerate()
            .zip(source_phases)
            .take(bandpass_index_max + 1)
            .skip(bandpass_index_min)
        {
            let bin_frequency = fft_bin_frequency(spectrum_size, sample_frequency, index);
            let source_frequency = bin_frequency + downmix_frequency;
            let amplitude = source_gain
                * source.reference_intensity
                * (source_frequency / source.reference_frequency).powf(source.spectral_index);
            let phase = *phase as f64 + TAU * source_frequency * delay;
            *spectrum_value += Complex64::from_polar(amplitude.sqrt(), phase);
        }
    }
}
