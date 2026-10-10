use std::{f64::consts::TAU, sync::Arc};

use num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;

use super::{Array, Calibrator, Transmitter, Vec3};

/// Speed of light in vacuum (m/s).
const C: f64 = 299792458.0;
/// Oversampling factor of conditioned transmit buffers.
const OVERSAMPLING: usize = 16;
/// Relative Doppler shift the transmit buffer prefilter leaves room for (|v| up to 30 km/s).
const DOPPLER_GUARD: f64 = 1e-4;
/// Maximum number of samples covered by a single delay polynomial.
const SEGMENT_LEN: usize = 4096;
/// Number of fixed-point iterations used to solve the light-time equation.
/// Each iteration reduces the error by a factor v/c.
const LIGHT_TIME_ITERATIONS: usize = 4;

/// Transmit buffer prepared for interpolation.
///
/// The buffer is filtered to the part of the spectrum that can reach the array's bandpass
/// (widened by a guard band for Doppler) and to the transmitter's bandwidth, then
/// oversampled by `OVERSAMPLING`. Because the buffer is cyclic, both steps are exact.
pub(crate) struct ConditionedBuffer {
    source: Arc<[Complex32]>,
    frequency: f64,
    sample_rate: f64,
    bandwidth: Option<f64>,
    pub(crate) samples: Vec<Complex32>,
}

impl ConditionedBuffer {
    /// Conditions the buffer of a transmitter for reception by an array.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration receiving the transmitter.
    /// - `transmitter`: Transmitter whose buffer is conditioned.
    ///
    /// # Returns
    /// The filtered and oversampled buffer.
    ///
    /// # Panics
    /// Panics if the oversampled buffer would exceed 2^30 samples.
    pub(crate) fn new(array: &Array, transmitter: &Transmitter) -> Self {
        let len = transmitter.buffer.len();
        let oversampled_len = len * OVERSAMPLING;
        assert!(
            oversampled_len <= 1 << 30,
            "transmit buffer of {len} samples is too long"
        );

        let mut planner = FftPlanner::<f64>::new();
        let mut spectrum = transmitter
            .buffer
            .iter()
            .map(|s| Complex64::new(s.re as f64, s.im as f64))
            .collect::<Vec<_>>();
        planner.plan_fft_forward(len).process(&mut spectrum);

        let guard = DOPPLER_GUARD * (array.downmix_frequency + array.bandpass[1]);
        let mut oversampled = vec![Complex64::ZERO; oversampled_len];
        for (k, value) in spectrum.iter().enumerate() {
            let bin = signed_bin(k, len);
            let frequency = bin as f64 * transmitter.sample_rate / len as f64;
            let received = frequency + transmitter.frequency - array.downmix_frequency;
            let in_band =
                received >= array.bandpass[0] - guard && received <= array.bandpass[1] + guard;
            let in_bandwidth = transmitter
                .bandwidth
                .is_none_or(|b| frequency.abs() <= b / 2.0);
            if in_band && in_bandwidth {
                oversampled[bin.rem_euclid(oversampled_len as i64) as usize] = *value;
            }
        }
        planner
            .plan_fft_inverse(oversampled_len)
            .process(&mut oversampled);

        ConditionedBuffer {
            source: transmitter.buffer.clone(),
            frequency: transmitter.frequency,
            sample_rate: transmitter.sample_rate,
            bandwidth: transmitter.bandwidth,
            samples: oversampled
                .iter()
                .map(|s| Complex32::new((s.re / len as f64) as _, (s.im / len as f64) as _))
                .collect(),
        }
    }

    /// Checks whether this buffer was conditioned from the current state of a transmitter.
    ///
    /// # Arguments
    /// - `transmitter`: Transmitter to compare with.
    ///
    /// # Returns
    /// `true` if the transmitter's buffer and the parameters used for conditioning are unchanged.
    pub(crate) fn matches(&self, transmitter: &Transmitter) -> bool {
        Arc::ptr_eq(&self.source, &transmitter.buffer)
            && self.frequency == transmitter.frequency
            && self.sample_rate == transmitter.sample_rate
            && self.bandwidth == transmitter.bandwidth
    }

    /// Interpolates the cyclic buffer using 4-point Lagrange interpolation.
    ///
    /// # Arguments
    /// - `position`: Position in oversampled samples, in `[0, len]`.
    ///
    /// # Returns
    /// The interpolated sample.
    fn interpolate(&self, position: f64) -> Complex64 {
        let len = self.samples.len();
        let index = position.floor();
        let u = position - index;
        let index = index as usize + len;
        let weights = lagrange_weights(u);
        (0..4)
            .map(|i| {
                let s = self.samples[(index + i - 1) % len];
                Complex64::new(s.re as f64, s.im as f64) * weights[i]
            })
            .sum()
    }
}

/// Computes the weights of 4-point Lagrange interpolation.
///
/// # Arguments
/// - `u`: Fractional position between the middle two points, in `[0, 1)`.
///
/// # Returns
/// The weights of the four points.
fn lagrange_weights(u: f64) -> [f64; 4] {
    [
        -u * (u - 1.0) * (u - 2.0) / 6.0,
        (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0,
        -(u + 1.0) * u * (u - 2.0) / 2.0,
        (u + 1.0) * u * (u - 1.0) / 6.0,
    ]
}

/// Computes the signed frequency index of a DFT bin.
///
/// Unlike `fft_bin_frequency`, this supports DFTs of odd length.
///
/// # Arguments
/// - `k`: Bin index.
/// - `len`: Length of the DFT.
///
/// # Returns
/// The frequency index, in `(-len / 2, len / 2]` for odd and `[-len / 2, len / 2)` for even lengths.
fn signed_bin(k: usize, len: usize) -> i64 {
    if k <= (len - 1) / 2 {
        k as i64
    } else {
        k as i64 - len as i64
    }
}

/// Solves the light-time equation `τ = |receiver - p(time - τ)| / c`.
///
/// # Arguments
/// - `calibrator`: The emitting calibrator, with position `p(t)`.
/// - `receiver`: Position of the receiver.
/// - `time`: Time at which the signal is received (s).
///
/// # Returns
/// The delay `τ` (s) between emission and reception.
pub(crate) fn light_time(calibrator: &Calibrator, receiver: Vec3, time: f64) -> f64 {
    let mut delay = (receiver - calibrator.position_at(time)).norm() / C;
    for _ in 0..LIGHT_TIME_ITERATIONS {
        delay = (receiver - calibrator.position_at(time - delay)).norm() / C;
    }
    delay
}

/// Models the delay of a signal from `calibrator` to `receiver` around `time`.
///
/// # Arguments
/// - `calibrator`: The emitting calibrator.
/// - `receiver`: Position of the receiver.
/// - `time`: Center of the modeled interval (s).
/// - `half_duration`: Half the length of the modeled interval (s).
///
/// # Returns
/// The delay `τ0` at `time`, and the coefficients (lowest order first) of the cubic
/// polynomial approximating `τ(time + x * half_duration) - τ0` for `x ∈ [-1, 1]`.
/// The polynomial interpolates the exact delay at the Chebyshev nodes.
pub(crate) fn delay_model(
    calibrator: &Calibrator,
    receiver: Vec3,
    time: f64,
    half_duration: f64,
) -> (f64, [f64; 4]) {
    let nodes: [f64; 4] =
        std::array::from_fn(|j| (std::f64::consts::PI * (2 * j + 1) as f64 / 8.0).cos());
    let delay0 = light_time(calibrator, receiver, time);
    let delays = nodes.map(|x| light_time(calibrator, receiver, time + x * half_duration) - delay0);
    (delay0, fit_cubic(nodes, delays))
}

/// Calibrator constants used during sample evaluation.
pub(crate) struct CalibratorSignal {
    /// Carrier frequency (Hz).
    pub(crate) frequency: f64,
    /// Rate at which oversampled buffer samples are played out (Hz).
    pub(crate) oversampled_rate: f64,
    pub(crate) buffer: Arc<ConditionedBuffer>,
}

/// Antenna independent part of a calibrator sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CommonSample {
    /// Integer part of the oversampled buffer position `(t - t_start) * rate`.
    pub(crate) buffer_index: u32,
    /// Fractional part of the oversampled buffer position.
    pub(crate) buffer_frac: f64,
    /// Baseband carrier phase `(f_c - f_LO) * t` in cycles, in `[0, 1)`.
    pub(crate) carrier: f64,
}

/// Delay model of one antenna-calibrator pair over one segment of the synthesis window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    /// Integer part of the oversampled buffer delay `τ0 * rate`.
    pub(crate) buffer_index: u32,
    /// Fractional part of the oversampled buffer delay.
    pub(crate) buffer_frac: f64,
    /// Carrier phase delay `f_c * τ0` in cycles, in `[0, 1)`.
    pub(crate) carrier: f64,
    /// Received amplitude, including the antenna gain.
    pub(crate) amplitude: f64,
    /// Cubic polynomial `τ(x) - τ0` (s) in the normalized segment coordinate `x ∈ [-1, 1]`.
    pub(crate) delay: [f64; 4],
}

/// All calibrator signals of one synthesis window.
///
/// Calibrators are evaluated in the time domain, by playing their transmit buffer
/// against the time-varying light-time delay to each antenna, which includes carrier
/// and code Doppler. The runtimes transform the resulting signal to the frequency domain
/// and add it to the synthesized spectrum, so calibrators pass through the same bandpass
/// as the sky.
///
/// All geometry is computed in f64. Large quantities (buffer positions and carrier
/// phases) are reduced modulo the buffer length or one cycle, so that the GPU runtime
/// only needs to add small remainders in f32.
///
/// Sample `m` of the synthesis window is received at
/// `start_time + (m - offset) / sample_frequency`, such that the first output sample
/// is received at `start_time`.
pub(crate) struct CalibratorWindow {
    pub(crate) num_bins: usize,
    /// Number of samples before the first output sample.
    pub(crate) offset: usize,
    pub(crate) segment_len: usize,
    pub(crate) num_segments: usize,
    pub(crate) calibrators: Vec<CalibratorSignal>,
    /// Indexed by `calibrator * num_bins + m`.
    pub(crate) common: Vec<CommonSample>,
    /// Indexed by `(antenna * num_calibrators + calibrator) * num_segments + segment`.
    pub(crate) segments: Vec<Segment>,
}

impl CalibratorWindow {
    /// Precomputes the calibrator signals of a synthesis window.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration.
    /// - `calibrators`: Calibrators to simulate.
    /// - `buffers`: Conditioned buffers of the calibrators' transmitters.
    /// - `num_bins`: Number of samples in the synthesis window.
    /// - `start_time`: Time at which the first output sample is received (s).
    ///
    /// # Returns
    /// The precomputed calibrator signals.
    ///
    /// # Panics
    /// Panics if a calibrator coincides with an antenna.
    pub(crate) fn new(
        array: &Array,
        calibrators: &[Calibrator],
        buffers: &[Arc<ConditionedBuffer>],
        num_bins: usize,
        start_time: f64,
    ) -> Self {
        let sample_frequency = array.sample_frequency;
        let offset = (num_bins - array.sample_window_size) / 2;
        // time of sample m relative to start_time
        let sample_time = |m: f64| (m - offset as f64) / sample_frequency;
        let segment_len = SEGMENT_LEN.min(num_bins);
        let num_segments = num_bins.div_ceil(segment_len);
        let segment_half_len = segment_half_len(segment_len);

        let signals = calibrators
            .iter()
            .zip(buffers)
            .map(|(calibrator, buffer)| CalibratorSignal {
                frequency: calibrator.transmitter.frequency,
                oversampled_rate: calibrator.transmitter.sample_rate * OVERSAMPLING as f64,
                buffer: buffer.clone(),
            })
            .collect::<Vec<_>>();

        let mut common = Vec::with_capacity(calibrators.len() * num_bins);
        for (calibrator, signal) in calibrators.iter().zip(&signals) {
            let len = signal.buffer.samples.len() as f64;
            let baseband_frequency = signal.frequency - array.downmix_frequency;
            let index0 = ((start_time - calibrator.transmitter.start_time)
                * signal.oversampled_rate)
                .rem_euclid(len);
            let carrier0 = (baseband_frequency * start_time).rem_euclid(1.0);
            common.extend((0..num_bins).map(|m| {
                let dt = sample_time(m as f64);
                let (buffer_index, buffer_frac) = split(index0 + dt * signal.oversampled_rate, len);
                let (_, carrier) = split(carrier0 + dt * baseband_frequency, 1.0);
                CommonSample {
                    buffer_index,
                    buffer_frac,
                    carrier,
                }
            }));
        }

        let mut segments =
            Vec::with_capacity(array.antenna_positions.len() * calibrators.len() * num_segments);
        for &antenna in &array.antenna_positions {
            for (calibrator, signal) in calibrators.iter().zip(&signals) {
                let len = signal.buffer.samples.len() as f64;
                let transmitter = &calibrator.transmitter;
                for segment in 0..num_segments {
                    let center = segment_center(segment, segment_len);
                    let time = start_time + sample_time(center);
                    let (delay0, delay) = delay_model(
                        calibrator,
                        antenna,
                        time,
                        segment_half_len / sample_frequency,
                    );

                    let direction = calibrator.position_at(time - delay0) - antenna;
                    let distance = direction.norm();
                    assert!(distance > 0.0, "calibrator coincides with an antenna");
                    let direction_z = direction.z / distance;
                    let gain = direction_z * direction_z;
                    let amplitude = (gain * transmitter.power / (2.0 * TAU)).sqrt() / distance;

                    let (buffer_index, buffer_frac) = split(delay0 * signal.oversampled_rate, len);
                    let (_, carrier) = split(signal.frequency * delay0, 1.0);
                    segments.push(Segment {
                        buffer_index,
                        buffer_frac,
                        carrier,
                        amplitude,
                        delay,
                    });
                }
            }
        }

        CalibratorWindow {
            num_bins,
            offset,
            segment_len,
            num_segments,
            calibrators: signals,
            common,
            segments,
        }
    }

    /// Evaluates the calibrator signal received by an antenna over the synthesis window.
    ///
    /// # Arguments
    /// - `antenna`: Index of the antenna in the array.
    ///
    /// # Returns
    /// The signal summed over all calibrators and tapered by `taper`.
    pub(crate) fn antenna_signal(&self, antenna: usize) -> Vec<Complex64> {
        let mut signal = vec![Complex64::ZERO; self.num_bins];
        for calibrator in 0..self.calibrators.len() {
            for (m, value) in signal.iter_mut().enumerate() {
                *value += self.sample(antenna, calibrator, m);
            }
        }
        for (m, value) in signal.iter_mut().enumerate() {
            *value *= taper(m, self.offset, self.num_bins);
        }
        signal
    }

    /// Evaluates a sample of a calibrator signal as received by an antenna.
    ///
    /// # Arguments
    /// - `antenna`: Index of the antenna in the array.
    /// - `calibrator`: Index of the calibrator.
    /// - `m`: Index of the sample in the synthesis window.
    ///
    /// # Returns
    /// The received sample, without taper.
    pub(crate) fn sample(&self, antenna: usize, calibrator: usize, m: usize) -> Complex64 {
        let signal = &self.calibrators[calibrator];
        let common = self.common[calibrator * self.num_bins + m];
        let segment_idx = m / self.segment_len;
        let segment = self.segments
            [(antenna * self.calibrators.len() + calibrator) * self.num_segments + segment_idx];

        let x = (m as f64 - segment_center(segment_idx, self.segment_len))
            / segment_half_len(self.segment_len);
        let [d0, d1, d2, d3] = segment.delay;
        let delay = d0 + x * (d1 + x * (d2 + x * d3));

        let len = signal.buffer.samples.len() as f64;
        let position = (common.buffer_index as f64 - segment.buffer_index as f64
            + common.buffer_frac
            - segment.buffer_frac
            - signal.oversampled_rate * delay)
            .rem_euclid(len);
        let value = signal.buffer.interpolate(position);

        let phase = common.carrier - segment.carrier - signal.frequency * delay;
        value * Complex64::from_polar(segment.amplitude, TAU * phase)
    }
}

/// Computes the taper applied to calibrator signals.
///
/// The calibrator signal is not periodic over the synthesis window. Tapering it to zero
/// with a raised cosine over the samples that are discarded from the output removes the
/// discontinuity at the window edges, which would otherwise ring through the bandpass
/// filter into the output samples. The output samples themselves have weight one.
///
/// # Arguments
/// - `m`: Index of the sample in the synthesis window.
/// - `offset`: Number of samples before the first output sample.
/// - `num_bins`: Number of samples in the synthesis window.
///
/// # Returns
/// The weight of the sample, in `(0, 1]`.
pub(crate) fn taper(m: usize, offset: usize, num_bins: usize) -> f64 {
    let distance = m.min(num_bins - 1 - m);
    if distance >= offset {
        1.0
    } else {
        let u = (distance as f64 + 0.5) / offset as f64;
        0.5 - 0.5 * (std::f64::consts::PI * u).cos()
    }
}

/// Computes the center of a segment of the synthesis window.
///
/// # Arguments
/// - `segment`: Index of the segment.
/// - `segment_len`: Number of samples per segment.
///
/// # Returns
/// The center of the segment, in samples.
pub(crate) fn segment_center(segment: usize, segment_len: usize) -> f64 {
    (segment * segment_len) as f64 + (segment_len - 1) as f64 / 2.0
}

/// Computes the distance from the center of a segment to its outermost samples.
///
/// # Arguments
/// - `segment_len`: Number of samples per segment.
///
/// # Returns
/// Half the length of the segment, in samples.
pub(crate) fn segment_half_len(segment_len: usize) -> f64 {
    ((segment_len - 1) as f64 / 2.0).max(0.5)
}

/// Reduces a value modulo a modulus and splits it into an integer and a fractional part.
///
/// # Arguments
/// - `value`: Value to reduce.
/// - `modulus`: Positive modulus.
///
/// # Returns
/// The integer and fractional part of the reduced value.
fn split(value: f64, modulus: f64) -> (u32, f64) {
    let mut reduced = value.rem_euclid(modulus);
    if reduced >= modulus {
        // rem_euclid can round up to the modulus for tiny negative values
        reduced = 0.0;
    }
    let index = reduced.floor();
    (index as u32, reduced - index)
}

/// Fits the cubic polynomial through four points.
///
/// # Arguments
/// - `x`: Distinct x coordinates of the points.
/// - `y`: y coordinates of the points.
///
/// # Returns
/// The coefficients, lowest order first.
fn fit_cubic(x: [f64; 4], y: [f64; 4]) -> [f64; 4] {
    // Gaussian elimination with partial pivoting on the Vandermonde system
    let mut a: [[f64; 5]; 4] =
        std::array::from_fn(|i| [1.0, x[i], x[i] * x[i], x[i] * x[i] * x[i], y[i]]);
    for col in 0..4 {
        let pivot = (col..4)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        for row in col + 1..4 {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (value, pivot) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *value -= factor * pivot;
            }
        }
    }
    let mut coefficients = [0.0; 4];
    for row in (0..4).rev() {
        let sum: f64 = (row + 1..4).map(|k| a[row][k] * coefficients[k]).sum();
        coefficients[row] = (a[row][4] - sum) / a[row][row];
    }
    coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::{Simulation, Source};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const SAMPLE_FREQUENCY: f64 = 120e6;
    const DOWNMIX_FREQUENCY: f64 = 45e6;

    fn test_array(antenna_positions: Vec<Vec3>, sample_window_size: usize) -> Array {
        Array {
            sample_frequency: SAMPLE_FREQUENCY,
            downmix_frequency: DOWNMIX_FREQUENCY,
            bandpass: [5e6, 55e6],
            sample_window_size,
            system_noise_intensity: 0.0,
            antenna_positions,
        }
    }

    /// Simulates one window on the CPU runtime.
    fn simulate(
        array: &Array,
        sources: Vec<Source>,
        calibrators: Vec<Calibrator>,
        frequency_resolution: usize,
        time: f64,
    ) -> Vec<Vec<Complex64>> {
        let mut simulation =
            Simulation::new("cpu".into(), array.clone(), frequency_resolution, Some(1));
        simulation.set_sources(sources);
        simulation.set_calibrators(calibrators);
        simulation.start(Some(time));
        to_complex64(simulation.finish())
    }

    fn to_complex64(samples: Vec<Vec<Complex32>>) -> Vec<Vec<Complex64>> {
        samples
            .into_iter()
            .map(|s| {
                s.into_iter()
                    .map(|v| Complex64::new(v.re as _, v.im as _))
                    .collect()
            })
            .collect()
    }

    /// Transmitter of a single tone at `frequency`.
    fn tone(frequency: f64, power: f64) -> Transmitter {
        Transmitter::new(frequency, 1e6, power, vec![Complex32::ONE], None, 0.0)
    }

    fn static_calibrator(position: Vec3, transmitter: Transmitter) -> Calibrator {
        Calibrator::new(position, transmitter, None, None, 0.0)
    }

    fn power(samples: &[Complex64]) -> f64 {
        samples.iter().map(|s| s.norm_sqr()).sum::<f64>() / samples.len() as f64
    }

    fn relative_rms_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).norm_sqr())
            .sum::<f64>();
        (error / expected.iter().map(|e| e.norm_sqr()).sum::<f64>()).sqrt()
    }

    /// Wraps a phase to `(-π, π]`.
    fn wrap(phase: f64) -> f64 {
        Complex64::from_polar(1.0, phase).arg()
    }

    #[test]
    fn light_time_static() {
        let position = Vec3::new(3e3, -4e3, 12e3);
        let calibrator = static_calibrator(position, tone(75e6, 1.0));
        let delay = light_time(&calibrator, Vec3::default(), 5.0);
        assert!((delay - position.norm() / C).abs() < 1e-18);
    }

    #[test]
    fn light_time_receding() {
        // x(t) = x0 + v t, so c τ = x0 + v (t - τ)
        let (x0, v, t) = (1e5, 1600.0, 2.0);
        let calibrator = Calibrator::new(
            Vec3::new(x0, 0.0, 0.0),
            tone(75e6, 1.0),
            Some(Vec3::new(v, 0.0, 0.0)),
            None,
            0.0,
        );
        let delay = light_time(&calibrator, Vec3::default(), t);
        assert!((delay - (x0 + v * t) / (C + v)).abs() < 1e-18);
    }

    #[test]
    fn delay_polynomial_matches_light_time() {
        // low lunar orbit like dynamics
        let calibrator = Calibrator::new(
            Vec3::new(-2e4, 1e4, 1e5),
            tone(75e6, 1.0),
            Some(Vec3::new(1600.0, 200.0, -100.0)),
            Some(Vec3::new(-2.0, 0.5, -25.6)),
            0.0,
        );
        let receiver = Vec3::new(50.0, -30.0, 1.0);
        // segments of 4096 samples at 120 MHz and at 1 MHz
        for half_duration in [2048.0 / 120e6, 2048.0 / 1e6] {
            let time = 0.5;
            let (delay0, [d0, d1, d2, d3]) =
                delay_model(&calibrator, receiver, time, half_duration);
            let max_error = (0..=100)
                .map(|i| {
                    let x = i as f64 / 50.0 - 1.0;
                    let modeled = delay0 + d0 + x * (d1 + x * (d2 + x * d3));
                    let exact = light_time(&calibrator, receiver, time + x * half_duration);
                    (modeled - exact).abs()
                })
                .fold(0.0, f64::max);
            println!("max delay polynomial error: {max_error:e} s");
            assert!(max_error < 1e-15);
        }
    }

    /// A distant static calibrator has the same geometric phase between antennas as a sky
    /// source in the same direction, including its sign and the carrier phase.
    #[test]
    fn distant_calibrator_matches_source() {
        let antennas = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(7.0, -3.0, 0.0)];
        let baseline = antennas[1] - antennas[0];
        let array = test_array(antennas, 4096);
        let direction = Vec3::new(0.3, 0.2, 1.0).normalized();
        let bin = 1024; // 30 MHz baseband
        let baseband_frequency = bin as f64 * SAMPLE_FREQUENCY / 4096.0;
        let expected =
            wrap(TAU * (baseband_frequency + DOWNMIX_FREQUENCY) * direction.dot(baseline) / C);

        let source = Source::new(direction, 75e6, 1.0, 0.0);
        let samples = simulate(&array, vec![source], vec![], 4, 0.0);
        let spectra = samples
            .into_iter()
            .map(|mut s| {
                FftPlanner::new().plan_fft_forward(s.len()).process(&mut s);
                s
            })
            .collect::<Vec<_>>();
        let source_phase = (spectra[1][bin] * spectra[0][bin].conj()).arg();

        let calibrator = static_calibrator(
            direction * 1e9,
            tone(DOWNMIX_FREQUENCY + baseband_frequency, 1.0),
        );
        let samples = simulate(&array, vec![], vec![calibrator], 4, 0.0);
        let calibrator_phase = samples[1]
            .iter()
            .zip(&samples[0])
            .map(|(a, b)| a * b.conj())
            .sum::<Complex64>()
            .arg();

        println!("expected {expected}, source {source_phase}, calibrator {calibrator_phase}");
        assert!(wrap(source_phase - expected).abs() < 0.02);
        assert!(wrap(calibrator_phase - expected).abs() < 1e-3);
    }

    #[test]
    fn tone_is_doppler_shifted() {
        let array = test_array(vec![Vec3::default()], 4096);
        let carrier = DOWNMIX_FREQUENCY + 20e6;
        // approaching the antenna at 1.6 km/s
        let velocity = -1600.0;
        let calibrator = Calibrator::new(
            Vec3::new(0.0, 0.0, 1e5),
            tone(carrier, 1.0),
            Some(Vec3::new(0.0, 0.0, velocity)),
            None,
            0.0,
        );
        let samples = &simulate(&array, vec![], vec![calibrator], 4, 0.0)[0];
        let phase_step = samples
            .windows(2)
            .map(|w| w[1] * w[0].conj())
            .sum::<Complex64>()
            .arg();
        let measured = phase_step * SAMPLE_FREQUENCY / TAU + DOWNMIX_FREQUENCY;
        let expected = carrier / (1.0 + velocity / C);
        println!(
            "doppler shift {} Hz, error {} Hz",
            expected - carrier,
            measured - expected
        );
        assert!((measured - expected).abs() < 1.0);
    }

    /// Compares a moving calibrator with an arbitrary band-limited buffer against a direct
    /// evaluation of the signal model, using exact light-time and exact (trigonometric)
    /// interpolation of the cyclic buffer. Covers delay, carrier and code Doppler,
    /// and the interpolation.
    #[test]
    fn moving_calibrator_matches_signal_model() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let len = 997;
        let tx_sample_rate = 37e6;
        let mut spectrum = (0..len)
            .map(|k| {
                let frequency = signed_bin(k, len) as f64 * tx_sample_rate / len as f64;
                if frequency.abs() <= 12e6 {
                    Complex64::from_polar(rng.random_range(0.5..1.0), rng.random_range(0.0..TAU))
                } else {
                    Complex64::ZERO
                }
            })
            .collect::<Vec<_>>();
        FftPlanner::new()
            .plan_fft_inverse(len)
            .process(&mut spectrum);
        let rms = (power(&spectrum)).sqrt();
        let buffer = spectrum
            .iter()
            .map(|s| Complex32::new((s.re / rms) as _, (s.im / rms) as _))
            .collect::<Vec<_>>();

        let carrier = DOWNMIX_FREQUENCY + 30e6;
        let (power_w, start_time, time) = (10.0, 0.1, 0.37);
        let transmitter = Transmitter::new(
            carrier,
            tx_sample_rate,
            power_w,
            buffer.clone(),
            None,
            start_time,
        );
        let calibrator = Calibrator::new(
            Vec3::new(2e4, -1e4, 9e4),
            transmitter,
            Some(Vec3::new(1200.0, -800.0, 300.0)),
            Some(Vec3::new(3.0, 1.0, -20.0)),
            0.0,
        );
        let antenna = Vec3::new(40.0, -25.0, 1.0);
        let array = test_array(vec![antenna], 1024);
        let samples = &simulate(&array, vec![], vec![calibrator.clone()], 4, time)[0];

        let mut buffer_spectrum = buffer
            .iter()
            .map(|s| Complex64::new(s.re as _, s.im as _))
            .collect::<Vec<_>>();
        FftPlanner::new()
            .plan_fft_forward(len)
            .process(&mut buffer_spectrum);
        let expected = (0..array.sample_window_size)
            .map(|n| {
                let t = time + n as f64 / SAMPLE_FREQUENCY;
                let delay = light_time(&calibrator, antenna, t);
                let x = (t - delay - start_time) * tx_sample_rate;
                let value = buffer_spectrum
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        v * Complex64::from_polar(
                            1.0,
                            TAU * signed_bin(k, len) as f64 * x / len as f64,
                        )
                    })
                    .sum::<Complex64>()
                    / len as f64;
                let direction = calibrator.position_at(t - delay) - antenna;
                let distance = direction.norm();
                let gain = (direction.z / distance).powi(2);
                let amplitude = (gain * power_w / (2.0 * TAU)).sqrt() / distance;
                let phase = (carrier - DOWNMIX_FREQUENCY) * t - carrier * delay;
                value * Complex64::from_polar(amplitude, TAU * phase)
            })
            .collect::<Vec<_>>();

        let error = relative_rms_error(samples, &expected);
        println!("relative rms error {:.1} dB", 20.0 * error.log10());

        assert!(error < 1e-4);
    }

    #[test]
    fn consecutive_windows_are_contiguous() {
        let transmitter = Transmitter::new(
            DOWNMIX_FREQUENCY + 25e6,
            20e6,
            1.0,
            (0..311)
                .map(|i| Complex32::from_polar(1.0, (i * i) as f32 * 0.37))
                .collect(),
            Some(16e6),
            0.0,
        );
        let calibrator = Calibrator::new(
            Vec3::new(1e4, 2e4, 8e4),
            transmitter,
            Some(Vec3::new(-900.0, 1300.0, 50.0)),
            Some(Vec3::new(0.0, 0.0, -25.0)),
            0.0,
        );
        let antenna = Vec3::new(12.0, 7.0, 0.0);
        let time = 1.25;

        let short = test_array(vec![antenna], 1024);
        let mut simulation = Simulation::new("cpu".into(), short, 4, Some(1));
        simulation.set_calibrators(vec![calibrator.clone()]);
        simulation.start(Some(time));
        let mut joined = to_complex64(simulation.finish()).remove(0);
        simulation.start(None);
        joined.extend(to_complex64(simulation.finish()).remove(0));

        let long = test_array(vec![antenna], 2048);
        let single = &simulate(&long, vec![], vec![calibrator], 4, time)[0];

        let error = relative_rms_error(&joined, single);
        println!("relative rms error {:.1} dB", 20.0 * error.log10());
        assert!(error < 1e-3);
    }

    /// For a buffer that is white over the receiver sample rate, the received power equals
    /// that of the previous calibrator implementation with intensity equal to `power`:
    /// `G * P / (4π d²) * (in-band bins / bins)`.
    #[test]
    fn white_buffer_power() {
        let sample_window_size = 4096;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut buffer = (0..sample_window_size)
            .map(|_| Complex64::from_polar(1.0, rng.random_range(0.0..TAU)))
            .collect::<Vec<_>>();
        FftPlanner::new()
            .plan_fft_inverse(sample_window_size)
            .process(&mut buffer);
        let rms = power(&buffer).sqrt();
        let buffer = buffer
            .iter()
            .map(|s| Complex32::new((s.re / rms) as _, (s.im / rms) as _))
            .collect();

        let (power_w, distance) = (10.0, 1e3);
        let transmitter = Transmitter::new(
            DOWNMIX_FREQUENCY,
            SAMPLE_FREQUENCY,
            power_w,
            buffer,
            None,
            0.0,
        );
        let array = test_array(vec![Vec3::default()], sample_window_size);
        let calibrator = static_calibrator(Vec3::new(0.0, 0.0, distance), transmitter);
        // one bin per output sample, so the buffer is periodic over the synthesis window
        let samples = &simulate(&array, vec![], vec![calibrator], 1, 0.0)[0];

        let bins = sample_window_size as f64;
        let in_band_bins = (array.bandpass[1] * bins / SAMPLE_FREQUENCY).floor()
            - (array.bandpass[0] * bins / SAMPLE_FREQUENCY).ceil()
            + 1.0;
        let expected = power_w / (2.0 * TAU * distance * distance) * in_band_bins / bins;
        let measured = power(samples);
        println!("power ratio {}", measured / expected);
        assert!((measured / expected - 1.0).abs() < 1e-3);
    }

    /// Tones well within the bandpass are received in full and tones well outside it are
    /// removed. Near the band edges, the taper of the calibrator signal widens the edge of
    /// the bandpass to about `sample_frequency / offset` (20 kHz here).
    #[test]
    fn band_edges() {
        let sample_window_size = 4096;
        let frequency_resolution = 4;
        let array = test_array(vec![Vec3::default()], sample_window_size);
        let bins = (sample_window_size * frequency_resolution) as f64;
        let bin_spacing = SAMPLE_FREQUENCY / bins;
        let last_bin = (array.bandpass[1] / bin_spacing).floor();
        let position = Vec3::new(0.0, 0.0, 1e5);
        let amplitude2 = 1.0 / (2.0 * TAU * position.norm2());

        // received power relative to the transmitted tone
        let received_power = |bin: f64, velocity: f64| {
            let calibrator = Calibrator::new(
                position,
                tone(DOWNMIX_FREQUENCY + bin * bin_spacing, 1.0),
                Some(Vec3::new(0.0, 0.0, velocity)),
                None,
                0.0,
            );
            power(&simulate(&array, vec![], vec![calibrator], frequency_resolution, 0.0)[0])
                / amplitude2
        };

        for velocity in [0.0, -1600.0] {
            let in_band = received_power(last_bin - 10.0, velocity);
            // within the guard band of the prefilter, so only removed by the bandpass
            let just_out_of_band = received_power(last_bin + 1.0, velocity);
            let out_of_band = received_power(last_bin + 10.0, velocity);
            println!(
                "velocity {velocity}: in band {in_band}, just out of band {just_out_of_band:e}, out of band {out_of_band:e}"
            );
            assert!((in_band - 1.0).abs() < 1e-3);
            assert!(just_out_of_band < 0.1);
            assert!(out_of_band < 1e-12);
        }
    }
}
