use num_complex::{Complex32, Complex64};
use rustfft::{Fft, FftPlanner};
use std::{
    f64::consts::TAU,
    ops::{Range, RangeInclusive},
    sync::Arc,
    thread::{self, JoinHandle},
};

use crate::library::normalize_and_truncate;

use super::{
    Array, Phases, Source, Vec3,
    calibrator::CalibratorWindow,
    utils::{bandpass_bins, fft_bin_frequency},
};

/// Speed of light in vacuum (m/s).
const C: f64 = 299792458.0;
/// Bytes of the source fields of a tile of sources.
const FIELD_TILE_BYTES: usize = 64 << 20;
/// Bytes of the accumulated source spectra of a batch of antennas.
const ANTENNA_BATCH_BYTES: usize = 512 << 20;
/// Bins of the bandpass that are processed together, so that the spectra of the antennas of
/// a thread stay in its cache while all sources are added to them. The phasor of the
/// geometric delay is computed exactly at the start of every chunk, which bounds the
/// rounding error of its single precision recurrence.
const CHUNK: usize = 512;
/// Independent phasor recurrences per antenna and source, interleaved over the bins.
const LANES: usize = 8;

/// CPU runtime for parallel generation of simulated antenna data.
pub(crate) struct Runtime {
    frequency_resolution: usize,
    fft_planner: FftPlanner<f32>,
    calibrator_fft_planner: FftPlanner<f64>,
    worker: Option<JoinHandle<Vec<Vec<Complex32>>>>,
}

/// Inputs of a sample window.
struct Window {
    sample_frequency: f64,
    downmix_frequency: f64,
    bandpass: [f64; 2],
    sample_window_size: usize,
    spectrum_size: usize,
    system_noise_intensity: f64,
    antenna_positions: Vec<Vec3>,
    sources: Vec<Source>,
    calibrators: Option<Arc<CalibratorWindow>>,
    system_noise_phases: Arc<Vec<f32>>,
    source_phases: Arc<Vec<f32>>,
    fft: Arc<dyn Fft<f32>>,
    calibrator_fft: Arc<dyn Fft<f64>>,
}

impl Window {
    /// Physical frequency (Hz) of a bin of the synthesis window.
    fn frequency(&self, bin: usize) -> f64 {
        fft_bin_frequency(self.spectrum_size, self.sample_frequency, bin) + self.downmix_frequency
    }

    /// Range of the bins within the bandpass.
    fn bandpass_bins(&self) -> RangeInclusive<usize> {
        bandpass_bins(self.bandpass, self.sample_frequency, self.spectrum_size)
    }
}

/// Rounds a complex value to single precision.
fn to_f32(value: Complex64) -> Complex32 {
    Complex32::new(value.re as f32, value.im as f32)
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
            calibrator_fft_planner: FftPlanner::new(),
            worker: None,
        }
    }

    /// Start the simulation.
    ///
    /// Simulates the sample window on a background thread, which uses all available CPU
    /// cores.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrators`: Precomputed calibrator signals, if there are any calibrators.
    /// - `phases`: Precomputed phase information for system noise and sources.
    pub(crate) fn start(
        &mut self,
        array: &Array,
        sources: &[Source],
        calibrators: Option<&Arc<CalibratorWindow>>,
        phases: &Phases,
    ) {
        let spectrum_size = array.sample_window_size * self.frequency_resolution;
        let window = Window {
            sample_frequency: array.sample_frequency,
            downmix_frequency: array.downmix_frequency,
            bandpass: array.bandpass,
            sample_window_size: array.sample_window_size,
            spectrum_size,
            system_noise_intensity: array.system_noise_intensity,
            antenna_positions: array.antenna_positions.clone(),
            sources: sources.to_vec(),
            calibrators: calibrators.cloned(),
            system_noise_phases: phases.system_noise.clone(),
            source_phases: phases.sources.clone(),
            fft: self.fft_planner.plan_fft_inverse(spectrum_size),
            calibrator_fft: self.calibrator_fft_planner.plan_fft_forward(spectrum_size),
        };
        self.worker = Some(thread::spawn(move || simulate_window(&window)));
    }

    /// Finish the running simulation.
    ///
    /// Blocks until the sample window started by `start` has been simulated.
    ///
    /// # Returns
    /// Simulated time-domain antenna data.
    pub(crate) fn finish(&mut self) -> Vec<Vec<Complex32>> {
        self.worker
            .take()
            .map(|worker| worker.join().unwrap())
            .unwrap_or_default()
    }
}

/// Simulates one sample window for all antennas.
///
/// The sources are received in the frequency domain. The field of a source in every bin of
/// the bandpass is the same for all antennas apart from the geometric delay, so it is
/// computed once per tile of sources, and every antenna adds it delayed by its geometric
/// phase, which is a phasor that rotates linearly over the bins. The source spectra are
/// then combined with the receiver noise and the calibrators, transformed to the time domain
/// and truncated to the sample window.
///
/// The fields are computed in double precision, but summed in single precision, like the
/// output samples. The fields of a source and the source spectra of an antenna are stored as
/// the real parts of the bins of the bandpass followed by their imaginary parts.
fn simulate_window(window: &Window) -> Vec<Vec<Complex32>> {
    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let num_antennas = window.antenna_positions.len();
    let num_band_bins = window.bandpass_bins().count();

    let batch_len = (ANTENNA_BATCH_BYTES / (num_band_bins * size_of::<Complex32>()).max(1))
        .clamp(1, num_antennas);
    let mut output = Vec::with_capacity(num_antennas);
    for first in (0..num_antennas).step_by(batch_len) {
        let antennas = first..(first + batch_len).min(num_antennas);
        let mut spectra = vec![0.0; antennas.len() * 2 * num_band_bins];
        if num_band_bins > 0 {
            add_sources(&mut spectra, antennas.clone(), window, threads);
        }

        // receiver noise and calibrators, and the transform to the time domain
        let per_thread = antennas.len().div_ceil(threads);
        thread::scope(|scope| {
            let workers: Vec<_> = spectra
                .chunks(per_thread * 2 * num_band_bins.max(1))
                .enumerate()
                .map(|(i, spectra)| {
                    let first = antennas.start + i * per_thread;
                    scope.spawn(move || {
                        (first..)
                            .zip(spectra.chunks(2 * num_band_bins.max(1)))
                            .map(|(antenna, sources)| {
                                simulate_sample_window(antenna, window, sources)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for worker in workers {
                output.extend(worker.join().unwrap());
            }
        });
    }
    output
}

/// Adds the contributions of all sources to the spectra of a batch of antennas.
///
/// # Arguments
/// - `spectra`: Spectra of the bins within the bandpass, per antenna of the batch, real
///   parts followed by imaginary parts.
/// - `antennas`: Indices of the antennas of the batch.
/// - `window`: Inputs of the sample window.
/// - `threads`: Number of threads to use.
fn add_sources(spectra: &mut [f32], antennas: Range<usize>, window: &Window, threads: usize) {
    let num_band_bins = window.bandpass_bins().count();
    let field_len = 2 * num_band_bins;
    let tile_len = (FIELD_TILE_BYTES / (num_band_bins * size_of::<Complex32>())).max(1);
    let mut fields = vec![0.0; tile_len.min(window.sources.len()) * field_len];
    for first in (0..window.sources.len()).step_by(tile_len) {
        let tile = first..(first + tile_len).min(window.sources.len());
        let fields = &mut fields[..tile.len() * field_len];

        let per_thread = tile.len().div_ceil(threads);
        thread::scope(|scope| {
            for (i, fields) in fields.chunks_mut(per_thread * field_len).enumerate() {
                let first = tile.start + i * per_thread;
                scope.spawn(move || {
                    for (source, field) in (first..).zip(fields.chunks_mut(field_len)) {
                        source_field(field, source, window);
                    }
                });
            }
        });

        let fields = &*fields;
        let per_thread = antennas.len().div_ceil(threads);
        thread::scope(|scope| {
            for (i, spectra) in spectra.chunks_mut(per_thread * field_len).enumerate() {
                let first = antennas.start + i * per_thread;
                let tile = tile.clone();
                scope.spawn(move || {
                    receive_sources(spectra, first, window, tile, fields);
                });
            }
        });
    }
}

/// Computes the field of a source in every bin of the bandpass, including its random
/// phases, received with a cos²(θ) gain.
///
/// # Arguments
/// - `field`: Field of the source, real parts followed by imaginary parts.
/// - `s`: Index of the source.
/// - `window`: Inputs of the sample window.
fn source_field(field: &mut [f32], s: usize, window: &Window) {
    let source = &window.sources[s];
    let num_band_bins = field.len() / 2;
    let phases = &window.source_phases[s * num_band_bins..(s + 1) * num_band_bins];
    let z = source.direction.normalized().z;
    let gain = z * z;
    let (field_re, field_im) = field.split_at_mut(num_band_bins);
    for (((re, im), bin), &phase) in field_re
        .iter_mut()
        .zip(field_im)
        .zip(window.bandpass_bins())
        .zip(phases)
    {
        let amplitude = (gain * source.intensity_unchecked(window.frequency(bin))).sqrt();
        let value = to_f32(Complex64::from_polar(amplitude, phase as f64));
        (*re, *im) = (value.re, value.im);
    }
}

/// Adds the sources of a tile to the spectra of consecutive antennas, delayed geometrically.
///
/// # Arguments
/// - `spectra`: Spectra of the bins within the bandpass, per antenna, real parts followed by
///   imaginary parts.
/// - `first_antenna`: Index of the antenna of the first spectrum.
/// - `window`: Inputs of the sample window.
/// - `tile`: Indices of the sources of the tile.
/// - `fields`: Fields of the sources of the tile, per source real parts followed by
///   imaginary parts.
fn receive_sources(
    spectra: &mut [f32],
    first_antenna: usize,
    window: &Window,
    tile: Range<usize>,
    fields: &[f32],
) {
    let num_band_bins = window.bandpass_bins().count();
    let bin_width = window.sample_frequency / window.spectrum_size as f64;
    let first_frequency = window.frequency(*window.bandpass_bins().start());
    let antennas = first_antenna..first_antenna + spectra.len() / (2 * num_band_bins);

    for start in (0..num_band_bins).step_by(CHUNK) {
        let bins = start..(start + CHUNK).min(num_band_bins);
        let frequency = first_frequency + start as f64 * bin_width;
        for (s, field) in tile.clone().zip(fields.chunks(2 * num_band_bins)) {
            let direction = window.sources[s].direction;
            let (field_re, field_im) = field.split_at(num_band_bins);
            for (antenna, spectrum) in antennas.clone().zip(spectra.chunks_mut(2 * num_band_bins)) {
                let (re, im) = spectrum.split_at_mut(num_band_bins);
                let delay = direction.dot(window.antenna_positions[antenna]) / C;
                add_delayed(
                    &mut re[bins.clone()],
                    &mut im[bins.clone()],
                    &field_re[bins.clone()],
                    &field_im[bins.clone()],
                    frequency,
                    bin_width,
                    delay,
                );
            }
        }
    }
}

/// Adds a spectrum delayed by `delay` (s): `spectrum[k] += field[k] e^{i 2π f_k delay}`,
/// with `f_k = first_frequency + k bin_width`, for spectra given as real and imaginary parts.
///
/// The phasor is advanced from bin to bin by multiplication, in `LANES` interleaved
/// recurrences. Their start and step are computed in double precision.
///
/// The parts are separate arguments, which tells the compiler that they do not overlap, so
/// that the lanes are vectorized. Inlined, the loop is about 20% slower.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn add_delayed(
    spectrum_re: &mut [f32],
    spectrum_im: &mut [f32],
    field_re: &[f32],
    field_im: &[f32],
    first_frequency: f64,
    bin_width: f64,
    delay: f64,
) {
    // the phasors of the lanes follow from the first one and the rotation over one bin
    let rotation = Complex64::from_polar(1.0, TAU * bin_width * delay);
    let mut phasor = Complex64::from_polar(1.0, TAU * first_frequency * delay);
    let mut phasor_re = [0.0; LANES];
    let mut phasor_im = [0.0; LANES];
    for lane in 0..LANES {
        (phasor_re[lane], phasor_im[lane]) = (phasor.re as f32, phasor.im as f32);
        phasor *= rotation;
    }
    let step = to_f32(rotation.powu(LANES as u32));

    let (spectrum_re, spectrum_re_rest) = spectrum_re.as_chunks_mut::<LANES>();
    let (spectrum_im, spectrum_im_rest) = spectrum_im.as_chunks_mut::<LANES>();
    let (field_re, field_re_rest) = field_re.as_chunks::<LANES>();
    let (field_im, field_im_rest) = field_im.as_chunks::<LANES>();
    for (((spectrum_re, spectrum_im), field_re), field_im) in spectrum_re
        .iter_mut()
        .zip(spectrum_im)
        .zip(field_re)
        .zip(field_im)
    {
        for lane in 0..LANES {
            let (a, b) = (field_re[lane], field_im[lane]);
            let (p, q) = (phasor_re[lane], phasor_im[lane]);
            spectrum_re[lane] += a * p - b * q;
            spectrum_im[lane] += a * q + b * p;
            phasor_re[lane] = p * step.re - q * step.im;
            phasor_im[lane] = p * step.im + q * step.re;
        }
    }
    for lane in 0..spectrum_re_rest.len() {
        let (a, b) = (field_re_rest[lane], field_im_rest[lane]);
        let (p, q) = (phasor_re[lane], phasor_im[lane]);
        spectrum_re_rest[lane] += a * p - b * q;
        spectrum_im_rest[lane] += a * q + b * p;
    }
}

/// Simulates one time-domain sample window for a single antenna.
///
/// The spectrum of the sources is combined with the receiver noise and the calibrators,
/// transformed to the time domain using an inverse FFT, and truncated to the requested
/// sample window length.
///
/// # Arguments
///
/// * `antenna` - Index of the antenna in the array.
/// * `window` - Inputs of the sample window.
/// * `sources` - Spectrum of the sources in the bins within the bandpass, real parts followed
///   by imaginary parts.
///
/// # Returns
///
/// A vector containing the simulated complex time-domain samples.
fn simulate_sample_window(antenna: usize, window: &Window, sources: &[f32]) -> Vec<Complex32> {
    let mut spectrum = vec![Complex32::ZERO; window.spectrum_size];
    let num_band_bins = sources.len() / 2;
    let noise_phases =
        &window.system_noise_phases[antenna * num_band_bins..(antenna + 1) * num_band_bins];
    simulate_spectrum_system_noise_contribution(&mut spectrum, window, noise_phases);

    if let Some(calibrators) = &window.calibrators {
        simulate_spectrum_calibrator_contributions(&mut spectrum, antenna, window, calibrators);
    }

    let (sources_re, sources_im) = sources.split_at(num_band_bins);
    for ((value, &re), &im) in spectrum[window.bandpass_bins()]
        .iter_mut()
        .zip(sources_re)
        .zip(sources_im)
    {
        *value += Complex32::new(re, im);
    }

    window.fft.process(&mut spectrum); // in-place FFT, spectrum now contains time domain samples
    normalize_and_truncate(&spectrum, window.sample_window_size)
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
/// * `window` - Inputs of the sample window.
/// * `antenna_system_noise_phases` - Random phase assigned to each frequency bin of the bandpass.
fn simulate_spectrum_system_noise_contribution(
    spectrum: &mut [Complex32],
    window: &Window,
    antenna_system_noise_phases: &[f32],
) {
    for (index, &phase) in window.bandpass_bins().zip(antenna_system_noise_phases) {
        let lambda = C / window.frequency(index);
        let effective_aperture = (lambda * lambda / 4.0).min(4.0);
        let amplitude = (window.system_noise_intensity / effective_aperture).sqrt();
        spectrum[index] += to_f32(Complex64::from_polar(amplitude, phase as f64));
    }
}

/// Adds the contributions from all calibrators to a synthesized spectrum.
///
/// The calibrator signals are evaluated in the time domain, transformed to the
/// frequency domain and added within the bandpass, so that they are filtered
/// exactly like the sky.
///
/// # Arguments
///
/// * `spectrum` - Spectrum to modify in place.
/// * `antenna` - Index of the antenna in the array.
/// * `window` - Inputs of the sample window.
/// * `calibrators` - Precomputed calibrator signals.
fn simulate_spectrum_calibrator_contributions(
    spectrum: &mut [Complex32],
    antenna: usize,
    window: &Window,
    calibrators: &CalibratorWindow,
) {
    // The antenna gain does not depend on frequency, so all calibrators can share one FFT.
    let mut signal = calibrators.antenna_signal(antenna);
    window.calibrator_fft.process(&mut signal);
    // undo the scaling of the inverse FFT and the normalization of the output samples
    let scale = (spectrum.len() as f64).sqrt().recip();
    for index in window.bandpass_bins() {
        spectrum[index] += to_f32(signal[index] * scale);
    }
}
