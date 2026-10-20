use std::thread;

use num_complex::{Complex32, Complex64};

use super::{
    Array, Beam, Orientation, Source, Transmitter, Vec3,
    beam::{Pattern, cdot, cdot_real, cnorm},
    utils::fft_bin_frequency,
};

/// Maximum number of frequency nodes of a beam table.
const MAX_FREQUENCY_NODES: usize = 512;
/// Number of frequency nodes when the response varies with frequency, but only through
/// analytic beams.
const ANALYTIC_FREQUENCY_NODES: usize = 65;

/// Uniform grid of physical frequencies at which beam responses are tabulated.
///
/// A response that does not depend on frequency has a single node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrequencyNodes {
    /// Frequency of the first node (Hz).
    pub(crate) start: f64,
    /// Spacing of the nodes (Hz), 0 for a single node.
    pub(crate) step: f64,
    pub(crate) count: usize,
}

impl FrequencyNodes {
    /// Creates nodes spanning a band.
    ///
    /// # Arguments
    /// - `band`: Physical frequency range (Hz).
    /// - `beams`: Beams that contribute to the response.
    /// - `flat`: Whether the response is independent of frequency.
    fn new(band: [f64; 2], beams: &[&Beam], flat: bool) -> Self {
        if flat {
            return FrequencyNodes {
                start: 0.5 * (band[0] + band[1]),
                step: 0.0,
                count: 1,
            };
        }
        let width = band[1] - band[0];
        // twice as fine as the finest tabulated beam
        let spacing = beams
            .iter()
            .filter_map(|b| b.as_tabulated())
            .filter_map(|t| {
                t.frequencies
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .reduce(f64::min)
            })
            .reduce(f64::min);
        let count = match spacing {
            Some(s) => ((2.0 * width / s).ceil() as usize + 1).clamp(2, MAX_FREQUENCY_NODES),
            None => ANALYTIC_FREQUENCY_NODES,
        };
        FrequencyNodes {
            start: band[0],
            step: width / (count - 1) as f64,
            count,
        }
    }

    /// Frequency (Hz) of node `k`.
    pub(crate) fn frequency(&self, k: usize) -> f64 {
        self.start + k as f64 * self.step
    }

    /// Finds the node below a frequency and the weight of the node above it, clamped to
    /// the nodes.
    pub(crate) fn locate(&self, frequency: f64) -> (usize, f64) {
        if self.count == 1 {
            return (0, 0.0);
        }
        let u = ((frequency - self.start) / self.step).clamp(0.0, (self.count - 1) as f64);
        let k = (u.floor() as usize).min(self.count - 2);
        (k, u - k as f64)
    }

    /// Linearly interpolates values tabulated at the nodes.
    pub(crate) fn interpolate<T>(&self, values: &[T], frequency: f64) -> T
    where
        T: Copy + std::ops::Mul<f64, Output = T> + std::ops::Add<Output = T>,
    {
        let (k, w) = self.locate(frequency);
        if self.count == 1 {
            values[0]
        } else {
            values[k] * (1.0 - w) + values[k + 1] * w
        }
    }
}

/// Antennas sharing a beam and orientation, which therefore respond identically.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReceiverGroup {
    /// Index into the array's beams.
    pub(crate) beam: usize,
    pub(crate) orientation: Orientation,
}

/// Receive response of an array: its antenna groups, reference area and noise.
///
/// The response of a receiver is expressed relative to the reference area `A_ref(f)`,
/// the largest peak effective area of the array's beams, so that `|h|² = A_e / A_ref`.
pub(crate) struct ArrayResponse {
    beams: Vec<Beam>,
    used_beams: Vec<usize>,
    pub(crate) groups: Vec<ReceiverGroup>,
    /// Group of every antenna.
    pub(crate) antenna_groups: Vec<u32>,
    pub(crate) nodes: FrequencyNodes,
    horizon: Option<Vec3>,
    /// Amplitude (√Jy) of the receiver noise per bin of the synthesis window.
    pub(crate) noise_amplitudes: Vec<f64>,
}

impl ArrayResponse {
    /// Prepares the response of an array.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration.
    /// - `num_bins`: Number of bins of the synthesis window.
    pub(crate) fn new(array: &Array, num_bins: usize) -> Self {
        // beams that are the same share a group
        let canonical = |b: usize| {
            (0..=b)
                .find(|&c| array.beams[c].same(&array.beams[b]))
                .unwrap()
        };
        let mut groups: Vec<ReceiverGroup> = Vec::new();
        let antenna_groups = (0..array.antenna_positions.len())
            .map(|a| {
                let beam = canonical(array.antenna_beam(a));
                let orientation = array.antenna_orientation(a);
                let group = groups
                    .iter()
                    .position(|g| g.beam == beam && g.orientation == orientation)
                    .unwrap_or_else(|| {
                        groups.push(ReceiverGroup { beam, orientation });
                        groups.len() - 1
                    });
                group as u32
            })
            .collect();
        let mut used_beams = groups.iter().map(|g| g.beam).collect::<Vec<_>>();
        used_beams.sort_unstable();
        used_beams.dedup();

        // the response relative to the reference area is independent of frequency for a
        // single analytic beam
        let flat = used_beams.len() == 1 && array.beams[used_beams[0]].as_tabulated().is_none();
        let beams = used_beams
            .iter()
            .map(|&b| &array.beams[b])
            .collect::<Vec<_>>();
        let nodes = FrequencyNodes::new(array.physical_band(), &beams, flat);

        let mut response = ArrayResponse {
            beams: array.beams.clone(),
            used_beams,
            groups,
            antenna_groups,
            nodes,
            horizon: array.horizon.map(|h| h.normalized()),
            noise_amplitudes: Vec::new(),
        };
        response.noise_amplitudes = (0..num_bins)
            .map(|bin| {
                let frequency = fft_bin_frequency(num_bins, array.sample_frequency, bin)
                    + array.downmix_frequency;
                if frequency > 0.0 {
                    (array.system_noise_intensity(frequency) / response.reference_area(frequency))
                        .sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        response
    }

    /// The reference area `A_ref` (m²) at a frequency.
    pub(crate) fn reference_area(&self, frequency: f64) -> f64 {
        self.used_beams
            .iter()
            .map(|&b| self.beams[b].peak_effective_area(frequency))
            .fold(0.0, f64::max)
    }

    /// Number of independent field components of a source the array receives: 1 if no
    /// receiver has polarization, else 2.
    pub(crate) fn source_polarizations(&self) -> usize {
        if (0..self.groups.len()).all(|g| self.is_scalar(g)) {
            1
        } else {
            2
        }
    }

    /// Whether the receivers of a group have no polarization.
    pub(crate) fn is_scalar(&self, group: usize) -> bool {
        self.beams[self.groups[group].beam].is_scalar()
    }

    /// The beams used by the array.
    pub(crate) fn used_beams(&self) -> impl Iterator<Item = &Beam> {
        self.used_beams.iter().map(|&b| &self.beams[b])
    }

    /// Receive response of a group to a wave arriving from `direction`, in the world frame.
    ///
    /// # Arguments
    /// - `group`: Index of the receiver group.
    /// - `direction`: Unit vector pointing from the antenna to the emitter.
    /// - `frequency`: Physical frequency (Hz).
    ///
    /// # Returns
    /// The receive vector `h` with `|h|² = A_e / A_ref`, or its magnitude for beams without
    /// polarization. A wave with field `E` is received as `h · E`.
    pub(crate) fn receive(&self, group: usize, direction: Vec3, frequency: f64) -> Pattern {
        let ReceiverGroup { beam, orientation } = self.groups[group];
        let beam = &self.beams[beam];
        let below_horizon = self.horizon.is_some_and(|up| direction.dot(up) < 0.0);
        let scale = if below_horizon {
            0.0
        } else {
            (beam.area_scale(frequency) / self.reference_area(frequency)).sqrt()
        };
        match beam.pattern(orientation.to_local(direction), frequency) {
            Pattern::Scalar(s) => Pattern::Scalar(s * scale),
            Pattern::Vector(v) => Pattern::Vector(orientation.to_world_c(v).map(|c| c * scale)),
        }
    }

    /// Complex response of the link between a transmitter and a receiver, relative to an
    /// isotropic transmitter and a receiver with the reference area, both polarization
    /// matched.
    ///
    /// # Arguments
    /// - `antenna`: Index of the receiving antenna.
    /// - `transmitter`: The transmitter.
    /// - `attitude`: Orientation of the transmitter's beam.
    /// - `direction`: Unit vector pointing from the antenna to the transmitter.
    /// - `frequency`: Physical frequency (Hz).
    pub(crate) fn link(
        &self,
        antenna: usize,
        transmitter: &Transmitter,
        attitude: Orientation,
        direction: Vec3,
        frequency: f64,
    ) -> Complex64 {
        let receive = self.receive(self.antenna_groups[antenna] as usize, direction, frequency);
        // the field radiated towards the antenna
        let transmit = transmitter.beam.as_ref().map(|beam| {
            match beam.pattern(attitude.to_local(direction * -1.0), frequency) {
                Pattern::Vector(v) => attitude.to_world_c(v),
                Pattern::Scalar(_) => unreachable!("transmitter beams have a polarization"),
            }
        });
        match (receive, transmit) {
            (Pattern::Vector(h), Some(e)) => cdot(h, e),
            (Pattern::Vector(h), None) => cnorm(h).into(),
            (Pattern::Scalar(s), Some(e)) => (s * cnorm(e)).into(),
            (Pattern::Scalar(s), None) => s.into(),
        }
    }
}

/// Unit vectors pointing north (increasing declination) and east (increasing right
/// ascension) on the sky at a direction.
pub(crate) fn sky_basis(direction: Vec3) -> (Vec3, Vec3) {
    let ra = direction.y.atan2(direction.x);
    let dec = direction.z.clamp(-1.0, 1.0).asin();
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();
    let north = Vec3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
    let east = Vec3::new(-sin_ra, cos_ra, 0.0);
    (north, east)
}

/// Factors that turn two independent unit phasors `z1` and `z2` into a field with the
/// coherency of given Stokes parameters.
///
/// The field is `E_north = l11 z1`, `E_east = l21 z1 + l22 z2`, the Cholesky factorisation
/// of the coherency `½ [[I + Q, U + iV], [U - iV, I - Q]]`.
///
/// # Returns
/// `(l11, l21, l22)`.
pub(crate) fn field_factors([i, q, u, v]: [f64; 4]) -> (f64, Complex64, f64) {
    let b11 = 0.5 * (i + q);
    let b22 = 0.5 * (i - q);
    let b21 = Complex64::new(0.5 * u, -0.5 * v);
    let l11 = b11.max(0.0).sqrt();
    let l21 = if l11 > 0.0 {
        b21 / l11
    } else {
        Complex64::ZERO
    };
    let l22 = (b22 - l21.norm_sqr()).max(0.0).sqrt();
    (l11, l21, l22)
}

/// Response of every receiver group to every source, tabulated at the frequency nodes.
///
/// The response of a group to a source is its Jones row `(J_north, J_east)`, with which a
/// field `(E_north, E_east)` is received as `J_north E_north + J_east E_east`. For groups
/// without polarization, `J_north` is the real receive amplitude and `J_east` is zero.
pub(crate) struct SkyTable {
    pub(crate) num_groups: usize,
    pub(crate) nodes: FrequencyNodes,
    /// Indexed by `(source * num_groups + group) * nodes.count + node`.
    pub(crate) jones: Vec<[Complex32; 2]>,
}

impl SkyTable {
    /// Tabulates the response of an array to a sky model.
    pub(crate) fn new(response: &ArrayResponse, sources: &[Source]) -> Self {
        let num_groups = response.groups.len();
        let nodes = response.nodes;
        let per_source = num_groups * nodes.count;
        let mut jones = vec![[Complex32::ZERO; 2]; sources.len() * per_source];

        let threads = thread::available_parallelism().map_or(1, |n| n.get());
        let sources_per_thread = sources.len().div_ceil(threads).max(1);
        thread::scope(|scope| {
            for (sources, jones) in sources
                .chunks(sources_per_thread)
                .zip(jones.chunks_mut(sources_per_thread * per_source))
            {
                scope.spawn(move || {
                    for (source, jones) in sources.iter().zip(jones.chunks_mut(per_source)) {
                        let (north, east) = sky_basis(source.direction);
                        for (group, jones) in jones.chunks_mut(nodes.count).enumerate() {
                            for (k, value) in jones.iter_mut().enumerate() {
                                let frequency = nodes.frequency(k);
                                let [a, b] =
                                    match response.receive(group, source.direction, frequency) {
                                        Pattern::Scalar(s) => [s.into(), Complex64::ZERO],
                                        Pattern::Vector(h) => {
                                            [cdot_real(h, north), cdot_real(h, east)]
                                        }
                                    };
                                *value = [a, b].map(|c| Complex32::new(c.re as f32, c.im as f32));
                            }
                        }
                    }
                });
            }
        });

        SkyTable {
            num_groups,
            nodes,
            jones,
        }
    }

    /// Jones row of a group for a source at a location `(k, w)` between the nodes, from
    /// `FrequencyNodes::locate`.
    pub(crate) fn jones_at(
        &self,
        source: usize,
        group: usize,
        (k, w): (usize, f64),
    ) -> [Complex32; 2] {
        let base = (source * self.num_groups + group) * self.nodes.count;
        let lower = self.jones[base + k];
        if self.nodes.count == 1 {
            return lower;
        }
        let upper = self.jones[base + k + 1];
        let w = w as f32;
        [0, 1].map(|i| lower[i] * (1.0 - w) + upper[i] * w)
    }
}

/// Frequency nodes of the calibrator links of an array.
///
/// # Arguments
/// - `array`: Antenna array configuration.
/// - `response`: Receive response of the array.
/// - `transmitters`: Transmitters of the calibrators.
pub(crate) fn link_nodes<'a>(
    array: &Array,
    response: &'a ArrayResponse,
    transmitters: impl Iterator<Item = &'a Transmitter> + Clone,
) -> FrequencyNodes {
    let flat = response.nodes.count == 1
        && transmitters
            .clone()
            .all(|t| t.beam.as_ref().is_none_or(|b| b.as_tabulated().is_none()));
    let beams = response
        .used_beams()
        .chain(transmitters.filter_map(|t| t.beam.as_ref()))
        .collect::<Vec<_>>();
    FrequencyNodes::new(array.physical_band(), &beams, flat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::Simulation;
    use crate::library::beam::tests::tabulate;
    use crate::library::model::NoiseTemperature;

    #[test]
    fn field_factors_reproduce_coherency() {
        for stokes in [
            [2.0, 0.0, 0.0, 0.0],
            [2.0, 0.5, -0.3, 0.2],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ] {
            let (l11, l21, l22) = field_factors(stokes);
            let [i, q, u, v] = stokes;
            // <E E^H> with independent unit phasors
            let b11 = l11 * l11;
            let b22 = l21.norm_sqr() + l22 * l22;
            let b12 = l11 * l21.conj();
            assert!((b11 - 0.5 * (i + q)).abs() < 1e-12);
            assert!((b22 - 0.5 * (i - q)).abs() < 1e-12);
            assert!((b12 - Complex64::new(0.5 * u, 0.5 * v)).norm() < 1e-12);
        }
    }

    #[test]
    fn sky_basis_is_right_handed() {
        for direction in [
            Vec3::from_ra_dec(0.3, 0.4),
            Vec3::from_ra_dec(2.0, -1.0),
            Vec3::from_ra_dec(4.0, 0.0),
        ] {
            let (north, east) = sky_basis(direction);
            assert!(north.dot(direction).abs() < 1e-12 && east.dot(direction).abs() < 1e-12);
            // north, east and the propagation direction (towards the observer)
            assert!((north.cross(east) - direction * -1.0).norm() < 1e-12);
        }
        // north points towards the celestial pole
        let (north, _) = sky_basis(Vec3::from_ra_dec(1.0, 0.0));
        assert!((north - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
    }

    fn test_array(beams: Vec<Beam>, orientations: Vec<Orientation>) -> Array {
        let n = orientations.len();
        Array {
            sample_frequency: 120e6,
            downmix_frequency: 0.0,
            bandpass: [20e6, 50e6],
            sample_window_size: 4096,
            system_noise_temperature: NoiseTemperature::Constant(0.0),
            antenna_positions: vec![Vec3::default(); n],
            beams,
            antenna_beams: Vec::new(),
            antenna_orientations: orientations,
            horizon: None,
        }
    }

    /// Simulates one window on the CPU runtime.
    fn simulate(array: &Array, sources: Vec<Source>) -> Vec<Vec<Complex64>> {
        let mut simulation = Simulation::new("cpu".into(), array.clone(), 4, Some(7));
        simulation.set_sources(sources);
        simulation.start(Some(0.0));
        simulation
            .finish()
            .into_iter()
            .map(|s| {
                s.iter()
                    .map(|v| Complex64::new(v.re as _, v.im as _))
                    .collect()
            })
            .collect()
    }

    /// Mean of `a b*` over the samples.
    fn correlate(a: &[Complex64], b: &[Complex64]) -> Complex64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| a * b.conj())
            .sum::<Complex64>()
            / a.len() as f64
    }

    fn flat_source(direction: Vec3, intensity: f64, p: f64, chi: f64, v: f64) -> Source {
        Source::new(direction, 35e6, intensity, 0.0, 0.0, 0.0, p, chi, v, 0.0)
    }

    /// An unpolarized source with flux density `S`, flat over the bandpass of width `B`, is
    /// received by the legacy beam at the zenith with a mean power of `S B / (2 f_s)`.
    #[test]
    fn legacy_beam_receives_half_the_flux_density() {
        let array = Array {
            sample_window_size: 1 << 15,
            ..test_array(vec![Beam::cos_squared()], vec![Orientation::identity()])
        };
        // a single window of random phases scatters the power by about 1%
        let zenith = Vec3::new(0.0, 0.0, 1.0);
        let samples = simulate(&array, vec![flat_source(zenith, 8.0, 0.0, 0.0, 0.0)]);
        let measured = correlate(&samples[0], &samples[0]).re;
        let expected = 0.5 * 8.0 * 30e6 / 120e6;
        println!("measured {measured}, expected {expected}");
        assert!((measured / expected - 1.0).abs() < 0.03);

        // and with gain cos²θ off zenith
        let direction = Vec3::new(0.6, 0.0, 0.8);
        let samples = simulate(&array, vec![flat_source(direction, 8.0, 0.0, 0.0, 0.0)]);
        let measured = correlate(&samples[0], &samples[0]).re;
        println!("measured {measured}, expected {}", 0.64 * expected);
        assert!((measured / (0.64 * expected) - 1.0).abs() < 0.03);
    }

    /// The correlations of the X and Y receivers of a dual-polarized station recover the
    /// Stokes parameters of a source at the zenith, where north is -x and east is +y.
    #[test]
    fn dual_polarized_station_recovers_stokes() {
        let beam = tabulate(&Beam::short_dipole(), &[10e6, 60e6], 2.0, 2.0);
        let array = Array::dual_polarized(
            vec![Vec3::default()],
            120e6,
            0.0,
            20e6,
            50e6,
            1 << 14,
            NoiseTemperature::Constant(0.0),
            Some(beam),
            None,
            None,
        );
        let (p, chi, v) = (0.6, 0.4, -0.3);
        let zenith = Vec3::new(0.0, 0.0, 1.0);
        let samples = simulate(&array, vec![flat_source(zenith, 1.0, p, chi, v)]);
        let xx = correlate(&samples[0], &samples[0]).re;
        let yy = correlate(&samples[1], &samples[1]).re;
        let xy = correlate(&samples[0], &samples[1]);
        let i = xx + yy;
        let (q, u, v_measured) = ((xx - yy) / i, -2.0 * xy.re / i, -2.0 * xy.im / i);
        println!("q {q}, u {u}, v {v_measured}");
        assert!((q - p * (2.0 * chi).cos()).abs() < 0.03);
        assert!((u - p * (2.0 * chi).sin()).abs() < 0.03);
        assert!((v_measured - v).abs() < 0.03);
        // the dipole has its peak at the zenith, so receives half the flux density per polarization
        assert!((i / (30e6 / 120e6) - 1.0).abs() < 0.02);
    }

    #[test]
    fn horizon_blocks_sources() {
        let mut array = test_array(vec![Beam::cos_squared()], vec![Orientation::identity()]);
        let below = Vec3::new(0.0, 0.6, -0.8);
        let unmasked = simulate(&array, vec![flat_source(below, 1.0, 0.0, 0.0, 0.0)]);
        assert!(correlate(&unmasked[0], &unmasked[0]).re > 0.01);
        array.horizon = Some(Vec3::new(0.0, 0.0, 1.0));
        let masked = simulate(&array, vec![flat_source(below, 1.0, 0.0, 0.0, 0.0)]);
        assert_eq!(correlate(&masked[0], &masked[0]).re, 0.0);
    }

    /// Rotating an antenna and the direction of an incoming wave together rotates the
    /// receive vector with them.
    #[test]
    fn receive_vector_rotates_with_orientation() {
        let beam = tabulate(&Beam::short_dipole(), &[10e6, 60e6], 2.0, 2.0);
        let rotation = Orientation::new(Vec3::new(0.3, -0.5, 0.8), Vec3::new(1.0, 0.2, 0.0));
        let array = test_array(vec![beam], vec![Orientation::identity(), rotation]);
        let response = ArrayResponse::new(&array, 16);
        for i in 0..20 {
            let direction = Vec3::from_ra_dec(i as f64 * 0.31, 0.7 - i as f64 * 0.07);
            let (Pattern::Vector(h), Pattern::Vector(rotated)) = (
                response.receive(0, direction, 35e6),
                response.receive(1, rotation.to_world(direction), 35e6),
            ) else {
                unreachable!()
            };
            let expected = rotation.to_world_c(h);
            assert!(cnorm([0, 1, 2].map(|k| rotated[k] - expected[k])) < 1e-9);
        }
    }

    #[test]
    fn nodes_interpolate_linearly() {
        let nodes = FrequencyNodes {
            start: 10.0,
            step: 2.0,
            count: 3,
        };
        let values = [1.0, 3.0, 7.0];
        assert_eq!(nodes.interpolate(&values, 9.0), 1.0);
        assert_eq!(nodes.interpolate(&values, 11.0), 2.0);
        assert_eq!(nodes.interpolate(&values, 13.0), 5.0);
        assert_eq!(nodes.interpolate(&values, 20.0), 7.0);
    }
}
