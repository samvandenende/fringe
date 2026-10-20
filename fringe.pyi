from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Simulation:
    """Simulation for antenna signal generation."""

    def __init__(
        self,
        runtime: str,
        array: Array,
        frequency_resolution: int = ...,
        rng_seed: int | None = ...,
    ) -> None:
        """
        Creates a new simulation.

        # Arguments
        - `runtime`: Type of execution backend. Must be `"cpu"` or `"gpu"`.
        - `array`: Antenna array configuration.
        - `frequency_resolution` - FFT oversampling factor.
        - `rng_seed`: Optional RNG seed for reproducible phase generation.

        # Returns
        A simulation ready for configuration.
        """
        ...
    def set_sources(self, sources: list[Source]) -> None:
        """
        Set or update the source list used in the simulation.

        These sources will be used on the next call to `Simulation::start`

        The response of every receiver to every source is tabulated here, so calling this
        takes a while for large sky models with tabulated beams.

        # Arguments
        - `sources`: List of `Source` objects defining the sky model.
        """
        ...
    def set_calibrators(self, calibrators: list[Calibrator]) -> None:
        """
        Set or update the calibrators used in the simulation.

        These calibrators will be used on the next call to `Simulation::start`.
        Transmit buffers that did not change since the previous call are reused
        without being processed again.

        # Arguments
        - `calibrators`: List of `Calibrator` objects.
        """
        ...
    def time(self) -> float:
        """
        Returns the time (s) at which the next sample window starts if
        `Simulation::start` is called without a time.
        """
        ...
    def start(self, time: float | None = None) -> None:
        """
        Start simulation of a batch of time-domain signals.

        The simulation work is dispatched to the configured runtime.
        `Simulation::finish` must be called to obtain the results before
        a next call to `Simulation::start`.

        # Arguments
        - `time`: Time (s) at which the first sample of the window is received.
          If not given, the window directly follows the previous one.

        # Panics
        Panics if time is not finite.
        """
        ...
    def finish(self) -> npt.NDArray[np.complex64]:
        """
        Collects simulation results from the runtime.

        Must be called after `Simulation::start`.

        # Returns
        A 2D array of single precision complex-valued antenna samples, of shape
        (antennas, samples). Cast it to `np.complex128` if computations on it need double
        precision.
        """
        ...

class Vec3:
    """3D Cartesian vector."""

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Creates a new 3D vector.

        Args:
            x: X component.
            y: Y component.
            z: Z component.
        """
        ...

    def __repr__(self) -> str: ...
    def __add__(self, rhs: "Vec3") -> "Vec3": ...
    def __sub__(self, rhs: "Vec3") -> "Vec3": ...
    def __mul__(self, rhs: float) -> "Vec3": ...
    def __rmul__(self, rhs: float) -> "Vec3": ...
    def __truediv__(self, rhs: float) -> "Vec3": ...
    def add_inplace(self, other: "Vec3") -> None:
        """
        In-place vector addition.

        Args:
            other: Vector to add.
        """
        ...

    def sub_inplace(self, other: "Vec3") -> None:
        """
        In-place vector subtraction.

        Args:
            other: Vector to subtract.
        """
        ...

    def scale(self, s: float) -> None:
        """
        Scales the vector by a scalar.

        Args:
            s: Scale factor.
        """
        ...

    def normalize(self) -> None:
        """
        Normalizes the vector in-place to unit length.

        If the vector has zero magnitude, it remains unchanged.
        """
        ...

    def dot(self, other: "Vec3") -> float:
        """
        Computes the dot product with another vector.

        Args:
            other: Right-hand-side vector.

        Returns:
            The dot product.
        """
        ...

    def cross(self, other: "Vec3") -> "Vec3":
        """
        Computes the cross product with another vector.

        Args:
            other: Right-hand-side vector.

        Returns:
            The cross product.
        """
        ...

    def norm2(self) -> float:
        """
        Returns the squared Euclidean norm (square magnitude).
        """
        ...

    def norm(self) -> float:
        """
        Returns the Euclidean norm (magnitude).
        """
        ...

    def normalized(self) -> "Vec3":
        """
        Returns a normalized copy of the vector.

        If the vector has zero magnitude, it is returned unchanged.
        """
        ...

    def as_tuple(self) -> tuple[float, float, float]:
        """
        Converts the vector into a tuple.
        """
        ...

    @staticmethod
    def from_ra_dec(ra: float, dec: float):
        """
        Constructs a unit vector from spherical coordinates (RA, Dec).

        # Arguments:
        * ra — Right ascension in radians
        * dec — Declination in radians
        """
        ...

    def to_ra_dec(self) -> tuple[float, float]:
        """
        Converts the vector into spherical coordinates (RA, Dec) in radians.

        # Returns:
        Returns (ra, dec) where
          - ra ∈ [0, 2π)
          - dec ∈ [-π/2, π/2]
        """
        ...

class Orientation:
    """
    Orientation of a beam in the world frame.

    The beam's own frame is the frame of the antenna model it was simulated in: `θ` is
    measured from its +z axis and `φ` from its +x axis. The orientation places the model's
    +z axis along `boresight` and its +x axis as close to `reference` as possible.
    """

    def __init__(self, boresight: Vec3, reference: Vec3) -> None:
        """
        Creates an orientation from the direction of the beam's +z axis and a reference
        direction for its +x axis.

        # Panics
        Panics if `boresight` is zero or not finite, or if `reference` is parallel to it.
        """
        ...

    @staticmethod
    def identity() -> Orientation:
        """Returns the orientation in which the beam's frame equals the world frame."""
        ...

    def __repr__(self) -> str: ...
    def boresight(self) -> Vec3:
        """Returns the direction of the beam's +z axis in the world frame."""
        ...

    def reference(self) -> Vec3:
        """Returns the direction of the beam's +x axis in the world frame."""
        ...

    def rotated(self, angle: float) -> Orientation:
        """
        Returns this orientation rotated about its boresight by `angle` (radians), turning
        the +x axis towards the +y axis.
        """
        ...

    def to_local(self, v: Vec3) -> Vec3:
        """Expresses a world frame vector in the beam's frame."""
        ...

    def to_world(self, v: Vec3) -> Vec3:
        """Expresses a beam frame vector in the world frame."""
        ...

class Attitude:
    """Attitude of a calibrator, which orients its transmitter's beam."""

    @staticmethod
    def fixed(orientation: Orientation) -> Attitude:
        """Creates an attitude with a fixed orientation in the world frame."""
        ...

    @staticmethod
    def point_at(target: Vec3, reference: Vec3) -> Attitude:
        """
        Creates an attitude whose boresight tracks a point, such as the array center.

        # Arguments
        - `target`: Point (m) the beam's +z axis points at.
        - `reference`: Direction the beam's +x axis points as close to as possible. If it is
          parallel to the boresight, an arbitrary perpendicular direction is used.

        # Panics
        Panics if `target` or `reference` is not finite.
        """
        ...

    def __repr__(self) -> str: ...
    def orientation_at(self, position: Vec3) -> Orientation:
        """Computes the orientation of the beam of a calibrator at `position`."""
        ...

class Beam:
    """
    Antenna beam pattern, used by receivers and transmitters.

    A beam is defined in its own frame and placed in the world frame by an `Orientation`
    (receivers) or `Attitude` (calibrators). Beams are either analytic or tabulated far
    fields, imported from openEMS or built from arrays, such as far fields exported by
    other antenna simulation software.

    The far field `(E_θ, E_φ)` of a beam is normalized such that `|E_θ|² + |E_φ|²` is its
    directivity, gain or realized gain, or a relative gain for relative beams, whose
    effective area is given separately. The effective area is `λ² / 4π` times the gain.
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        theta: Sequence[float],
        phi: Sequence[float],
        e_theta: Sequence[complex],
        e_phi: Sequence[complex],
        normalization: str = "directivity",
        effective_area: Sequence[float] | None = None,
    ) -> None:
        """
        Creates a tabulated beam from far-field samples.

        # Arguments
        - `frequencies`: Frequencies (Hz).
        - `theta`: Values of `θ` (radians), measured from the beam's +z axis.
        - `phi`: Values of `φ` (radians), measured from the beam's +x axis towards +y.
        - `e_theta`: `E_θ` on the grid, flattened in the order `[frequency][θ][φ]`
          (`numpy.ravel` of an array of shape `(len(frequencies), len(theta), len(phi))`).
        - `e_phi`: `E_φ`, in the same order.
        - `normalization`: What `|E_θ|² + |E_φ|²` represents: `"directivity"`, `"gain"`,
          `"realized_gain"` or `"relative"`.
        - `effective_area`: Effective area (m²) per unit `|E|²` at each frequency, required
          for relative beams.

        The grid may use any `θ`/`φ` convention as long as the directions it represents
        form a regular grid with `φ` around the full circle. The beam is zero outside the
        `θ` range of the grid, so a hemisphere is enough for an antenna over a ground plane.

        # Panics
        Panics if the samples do not form a valid grid.
        """
        ...

    @staticmethod
    def cos_squared() -> Beam:
        """
        Returns the legacy beam: relative gain `cos²θ` with `θ` measured from the beam's
        +z axis, effective area `min(λ²/4, 4 m²)` and no polarization.
        """
        ...

    @staticmethod
    def isotropic(e_theta: complex = 1.0, e_phi: complex = 0.0) -> Beam:
        """
        Returns an isotropic beam with unit gain and a fixed polarization.

        # Arguments
        - `e_theta`, `e_phi`: Polarization in the `θ̂`/`φ̂` basis of the beam's frame,
          normalized to unit length. With the `exp(+iωt)` convention of fringe,
          `(1, -1j)` is right-hand circular for a wave traveling outwards.

        # Panics
        Panics if both components are zero.
        """
        ...

    @staticmethod
    def short_dipole() -> Beam:
        """
        Returns the beam of a short dipole along the beam's +x axis, with directivity 1.5.

        With the identity orientation the dipole is horizontal and the beam's boresight
        points to the zenith.
        """
        ...

    @staticmethod
    def load_openems(
        filepath: str,
        incident_power: Sequence[float] | None = None,
        excitation: Sequence[complex] | None = None,
    ) -> Beam:
        """
        Loads a far field computed by openEMS (`CalcNF2FF`, HDF5 output).

        The field is normalized to directivity using the `Dmax` stored with it, or to
        realized gain if the incident power of the excited port is given.

        The phase of the far field computed by openEMS includes the phase of the excitation
        signal, which differs per frequency (the delay of the excitation pulse), and the
        propagation to the radius of the far field. The latter is always removed. The
        excitation is removed by dividing by its phase if it is given, which keeps the phase
        response (group delay) of the antenna. Otherwise, the phase at each frequency is
        referenced to the field in the direction of the peak gain, which loses it.

        # Arguments
        - `filepath`: Path to the nf2ff HDF5 file.
        - `incident_power`: Incident power (W) of the excited port at each frequency of
          the file (`port.P_inc`), to normalize to realized gain.
        - `excitation`: Complex excitation at each frequency of the file, whose phase is
          removed: the incident voltage of the port (`port.uf_inc`) for the response of
          the antenna into the port's resistance, including its mismatch, which goes with
          `incident_power`.

        # Errors
        Raises an `IOError` if the file cannot be read or does not hold a valid far field.
        """
        ...

    def __repr__(self) -> str: ...
    def kind(self) -> str:
        """Returns the kind of beam: `"cos_squared"`, `"isotropic"`, `"short_dipole"` or `"tabulated"`."""
        ...

    def normalization(self) -> str:
        """
        Returns what `|E|²` represents: `"relative"` for the cos² beam, `"directivity"`
        for the other analytic beams, or the normalization of a tabulated beam.
        """
        ...

    def frequencies(self) -> list[float] | None:
        """Returns the frequencies (Hz) of a tabulated beam, or `None` for an analytic beam."""
        ...

    def gain(self, direction: Vec3, frequency: float) -> float:
        """
        Returns `|E|²` in a direction of the beam's frame: the gain, directivity or
        relative gain, depending on the normalization.

        # Arguments
        - `direction`: Direction in the beam's frame.
        - `frequency`: Frequency (Hz).
        """
        ...

    def field(self, direction: Vec3, frequency: float) -> tuple[complex, complex]:
        """
        Returns the far field `(E_θ, E_φ)` in a direction of the beam's frame.

        For the cos² beam, which has no polarization, this is `(√gain, 0)`.
        """
        ...

    def effective_area(self, direction: Vec3, frequency: float) -> float:
        """
        Returns the effective area (m²) in a direction of the beam's frame.

        # Panics
        Panics if `frequency` is not positive.
        """
        ...

    def diagnostics(self) -> str:
        """
        Returns a summary of the beam, to check an imported beam: its grid, frequencies,
        normalization, and per frequency the peak gain and the integrated gain over 4π,
        which is about 1 for directivity and the radiation efficiency for gain.
        """
        ...

class Array:
    """
    Array model and its signal acquisition parameters.

    This structure defines:
    - array geometry
    - sampling configuration
    - frequency conversion parameters
    - simulation noise characteristics

    Every antenna (receiver) has a beam and an orientation that places the beam in the
    world frame. A receiver is one polarization channel: a dual-polarized antenna is two
    receivers at the same position, see `Array.dual_polarized`.

    The output samples are in √Jy, relative to the reference area `A_ref(f)` of the array,
    the largest peak effective area of its beams: a source of flux density `S` received with
    effective area `A_e` has a mean power of `S * A_e / A_ref`, averaged over the sample
    frequency. A single polarization receives half the flux density of an unpolarized source.
    """

    def __init__(
        self,
        antenna_positions: list[Vec3],
        sample_frequency: float,
        downmix_frequency: float,
        bandpass_fmin: float,
        bandpass_fmax: float,
        sample_window_size: int,
        system_noise_temperature: float | Sequence[tuple[float, float]],
        beams: list[Beam] | None = None,
        antenna_beams: list[int] | None = None,
        antenna_orientations: list[Orientation] | None = None,
        horizon: Vec3 | None = None,
    ) -> None:
        """
        Creates a new antenna array configuration.

        # Arguments
        - `antenna_positions`: Positions of antennas in the array.
        - `sample_frequency`: ADC sampling frequency (Hz).
        - `downmix_frequency`: Frequency used for downconversion (Hz).
        - `bandpass_fmin`: Lower cutoff frequency of the bandpass filter (Hz).
        - `bandpass_fmax`: Upper cutoff frequency of the bandpass filter (Hz).
        - `sample_window_size`: Number of samples per FFT window (must be power of two).
        - `system_noise_temperature`: System noise temperature (K) of every receiver, as a
          number, or as `(frequency (Hz), temperature (K))` pairs with increasing
          frequencies that cover the bandpass, interpolated linearly in frequency. A
          receiver's noise power spectral density is `k_B * T`, which is received as a flux
          density of `k_B * T / A_ref`, with `A_ref` the reference area of the array. The sky
          is simulated from the sources, so `T` should not include the sky's contribution.
        - `beams`: Beams of the antennas, `[Beam.cos_squared()]` if not given.
        - `antenna_beams`: Index into `beams` per antenna, all 0 if not given.
        - `antenna_orientations`: Orientation of the beam per antenna, all the identity
          (boresight at the zenith, +z) if not given.
        - `horizon`: Up direction of the local horizon. If given, nothing is received from
          below the horizon.

        # Panics
        Panics if:
        - sample_frequency is not positive
        - downmix_frequency is negative
        - bandpass bounds are invalid or exceed Nyquist limit
        - sample_window_size is not a power of two
        - system_noise_temperature is negative, or a table that is not increasing in
          frequency or does not cover the bandpass
        - beams is empty, or a tabulated beam does not cover the bandpass
        - antenna_beams or antenna_orientations does not have a value per antenna, or
          antenna_beams refers to a beam that does not exist
        - horizon is zero or not finite
        """
        ...

    @staticmethod
    def dual_polarized(
        antenna_positions: list[Vec3],
        sample_frequency: float,
        downmix_frequency: float,
        bandpass_fmin: float,
        bandpass_fmax: float,
        sample_window_size: int,
        system_noise_temperature: float | Sequence[tuple[float, float]],
        beam: Beam | None = None,
        orientations: list[Orientation] | None = None,
        horizon: Vec3 | None = None,
    ) -> Array:
        """
        Creates an array of dual-polarized antennas.

        Every antenna is two receivers at the same position, with the same beam rotated by
        90° about its boresight: receiver `2 i` is the X polarization of antenna `i`, with
        the antenna's orientation, and receiver `2 i + 1` its Y polarization.

        # Arguments
        - `antenna_positions` up to `system_noise_temperature`: As for `Array(...)`.
        - `beam`: Beam of one polarization, `Beam.short_dipole()` if not given.
        - `orientations`: Orientation of the X polarization per antenna, the identity if
          not given, which with the short dipole puts X along +x and Y along +y.
        - `horizon`: As for `Array(...)`.

        # Panics
        Panics under the same conditions as `Array(...)`.
        """
        ...

    def __repr__(self) -> str: ...
    def sample_frequency(self) -> float:
        """Returns the sampling frequency in Hz."""
        ...

    def downmix_frequency(self) -> float:
        """Returns the downmix frequency in Hz."""
        ...

    def bandpass_fmin(self) -> float:
        """Returns the lower bound of the bandpass filter (f_min) in Hz."""
        ...

    def bandpass_fmax(self) -> float:
        """Returns the upper bound of the bandpass filter (f_max) in Hz."""
        ...

    def sample_window_size(self) -> int:
        """Returns the sample window size."""
        ...

    def system_noise_temperature(self) -> float | list[tuple[float, float]]:
        """
        Returns the system noise temperature as given: a temperature (K), or a list of
        `(frequency (Hz), temperature (K))` pairs.
        """
        ...

    def system_noise_temperature_at(self, frequency: float) -> float:
        """
        Returns the system noise temperature (K) at a frequency (Hz), interpolated in a
        table and held constant beyond its first and last frequency.
        """
        ...

    def antenna_positions(self) -> list[Vec3]:
        """Returns the antenna positions."""
        ...

    def beams(self) -> list[Beam]:
        """Returns the beams of the array."""
        ...

    def antenna_beams(self) -> list[int]:
        """Returns the index into `beams` of every antenna."""
        ...

    def antenna_orientations(self) -> list[Orientation]:
        """Returns the orientation of every antenna."""
        ...

    def horizon(self) -> Vec3 | None:
        """Returns the up direction of the local horizon, if any."""
        ...

    def reference_area(self, frequency: float) -> float:
        """
        Returns the reference area `A_ref` (m²) at a frequency (Hz): the largest peak
        effective area of the beams used by the antennas. Output samples are in √Jy
        relative to this area.

        # Panics
        Panics if `frequency` is not positive.
        """
        ...

class Source:
    """
    Source in the simulated sky model.

    Each source emits a frequency-dependent signal characterized by a reference
    flux density (Jy) and up to third-order spectral indices, and is located at a
    fixed direction vector. The flux density at frequency `f` is

    `I(f) = I0 * exp(a1 * x + a2 * x^2 + a3 * x^3)`, with `x = ln(f / f0)`,

    which is a power law with spectral index `a1` when `a2 = a3 = 0`.

    A source may be polarized. Its Stokes parameters at frequency `f` are

    `Q = p I cos 2χ`, `U = p I sin 2χ`, `V = v I`, with `χ = χ0 + RM λ²`,

    with `p` the linear polarization fraction, `χ0` the polarization angle, `RM` the
    rotation measure and `v` the circular polarization fraction. Polarization is given in
    the basis of north (towards +z, increasing declination) and east (increasing right
    ascension) on the sky, with `χ` measured from north through east. The coherency of the
    field `(E_north, E_east)` is `½ [[I + Q, U + iV], [U - iV, I - Q]]`.
    """

    def __init__(
        self,
        direction: Vec3,
        reference_frequency: float,
        reference_intensity: float,
        spectral_index: float,
        spectral_index_2: float = 0.0,
        spectral_index_3: float = 0.0,
        polarization_fraction: float = 0.0,
        polarization_angle: float = 0.0,
        circular_fraction: float = 0.0,
        rotation_measure: float = 0.0,
    ) -> None:
        """
        Creates a new signal source.

        # Arguments
        - `direction`: Unit vector indicating source direction in space.
        - `reference_frequency`: Frequency (Hz) at which the intensity is defined.
        - `reference_intensity`: Flux density (Jy) at the reference frequency.
        - `spectral_index`: First-order (power-law) spectral index of the source.
        - `spectral_index_2`: Second-order spectral index (curvature), defaults to 0.
        - `spectral_index_3`: Third-order spectral index, defaults to 0.
        - `polarization_fraction`: Linear polarization fraction `p`, defaults to 0.
        - `polarization_angle`: Polarization angle `χ0` (radians) from north through east,
          defaults to 0.
        - `circular_fraction`: Circular polarization fraction `v` (signed), defaults to 0.
        - `rotation_measure`: Rotation measure `RM` (rad/m²), defaults to 0.

        # Panics
        Panics if:
        - reference_frequency is not positive
        - reference_intensity is negative
        - any spectral index is not finite
        - polarization_fraction is negative, or the source is more than fully polarized
        - polarization_angle or rotation_measure is not finite
        """
        ...

    def __repr__(self) -> str: ...
    def direction(self):
        """Returns the source direction as a 3D vector."""
        ...

    def reference_frequency(self) -> float:
        """Returns the reference frequency in Hz used for spectral intensity scaling."""
        ...

    def reference_intensity(self) -> float:
        """Returns the flux density in Jy at the reference frequency."""
        ...

    def spectral_index(self) -> float:
        """Returns the first-order (power-law) spectral index."""
        ...

    def spectral_index_2(self) -> float:
        """Returns the second-order spectral index (curvature)."""
        ...

    def spectral_index_3(self) -> float:
        """Returns the third-order spectral index."""
        ...

    def polarization_fraction(self) -> float:
        """Returns the linear polarization fraction."""
        ...

    def polarization_angle(self) -> float:
        """Returns the polarization angle (radians), from north through east."""
        ...

    def circular_fraction(self) -> float:
        """Returns the circular polarization fraction."""
        ...

    def rotation_measure(self) -> float:
        """Returns the rotation measure (rad/m²)."""
        ...

    def intensity(self, frequency: float) -> float:
        """
        Computes the flux density in Jy at a given frequency (Hz) using the
        log-polynomial spectral model.

        # Panics
        Panics if `frequency <= 0.0`.
        """
        ...

    def stokes(self, frequency: float) -> tuple[float, float, float, float]:
        """
        Computes the Stokes parameters `(I, Q, U, V)` in Jy at a given frequency (Hz).

        # Panics
        Panics if `frequency <= 0.0`.
        """
        ...

class Transmitter:
    """
    Transmitter of a calibrator, configured like a software defined radio.

    The transmitter plays its complex baseband buffer cyclically at `sample_rate`,
    upconverted to the carrier `frequency`. Buffer samples are relative levels: the buffer
    is scaled such that playing it cyclically radiates `power`, whatever its RMS. The
    reconstruction filter is applied after this scaling, so any part of the buffer's
    spectrum outside `bandwidth` reduces the radiated power.

    A transmitter without a beam is isotropic, with `power` its equivalent isotropically
    radiated power, and its polarization matched to every receiver. A transmitter with a
    beam radiates `G_tx * power` towards a receiver, with `G_tx` the gain of the beam in
    that direction, and the received signal includes the polarization mismatch and the
    phase of both beams.

    A calibrator at distance `d` is received with a power flux density of
    `G_tx * P / (4π d²)` (W/m²), times the effective area of the receiver relative to the
    reference area of the array. It is spread over the sample frequency `f_s` of the array,
    so a calibrator whose signal is white over `f_s` is received like a fully polarized sky
    source of `G_tx * P / (4π d² f_s)` W/m²/Hz, i.e. that divided by 1e-26 in Jy.
    """

    def __init__(
        self,
        frequency: float,
        sample_rate: float,
        power: float,
        buffer: Sequence[complex],
        bandwidth: float | None = None,
        start_time: float = 0.0,
        beam: Beam | None = None,
    ) -> None:
        """
        Creates a new transmitter.

        # Arguments
        - `frequency`: Carrier frequency (Hz).
        - `sample_rate`: Rate at which buffer samples are played out (Hz).
        - `power`: Power (W) when playing the buffer: the equivalent isotropically radiated
          power without a beam, or the power fed to the beam, whose gain is applied, with one.
        - `buffer`: Complex baseband samples, transmitted cyclically.
        - `bandwidth`: Optional bandwidth of the reconstruction filter (Hz).
        - `start_time`: Time (s) at which the first buffer sample is transmitted.
        - `beam`: Optional beam of the transmitter, oriented by the calibrator's attitude.
          Outside the frequencies of a tabulated beam, its nearest frequency is used.

        # Panics
        Panics if:
        - frequency is negative
        - sample_rate is not positive
        - power is negative
        - bandwidth is not positive
        - buffer is empty, all zero or contains non-finite samples
        - beam is relative (as `Beam.cos_squared()`), or tabulated without covering frequency
        """
        ...

    def __repr__(self) -> str: ...
    def frequency(self) -> float:
        """Returns the carrier frequency in Hz."""
        ...

    def sample_rate(self) -> float:
        """Returns the rate in Hz at which buffer samples are played out."""
        ...

    def power(self) -> float:
        """
        Returns the power in W: the equivalent isotropically radiated power without a beam,
        or the power fed to the beam with one.
        """
        ...

    def bandwidth(self) -> float | None:
        """Returns the bandwidth of the reconstruction filter in Hz, if any."""
        ...

    def buffer(self) -> list[complex]:
        """Returns the transmitted buffer."""
        ...

    def start_time(self) -> float:
        """Returns the time in s at which the first buffer sample is transmitted."""
        ...

    def set_frequency(self, frequency: float) -> None:
        """
        Sets the carrier frequency in Hz.

        # Panics
        Panics if frequency is negative.
        """
        ...

    def set_sample_rate(self, sample_rate: float) -> None:
        """
        Sets the rate in Hz at which buffer samples are played out.

        # Panics
        Panics if sample_rate is not positive.
        """
        ...

    def set_power(self, power: float) -> None:
        """
        Sets the power in W: the equivalent isotropically radiated power without a beam, or
        the power fed to the beam with one.

        # Panics
        Panics if power is negative.
        """
        ...

    def set_bandwidth(self, bandwidth: float | None = None) -> None:
        """
        Sets the bandwidth of the reconstruction filter in Hz, or removes the filter with `None`.

        # Panics
        Panics if bandwidth is not positive.
        """
        ...

    def set_buffer(self, buffer: Sequence[complex]) -> None:
        """
        Sets the buffer that is transmitted cyclically.

        # Panics
        Panics if buffer is empty, all zero or contains non-finite samples.
        """
        ...

    def set_start_time(self, start_time: float) -> None:
        """
        Sets the time in s at which the first buffer sample is transmitted.

        # Panics
        Panics if start_time is not finite.
        """
        ...

    def beam(self) -> Beam | None:
        """Returns the beam of the transmitter, if any."""
        ...

    def set_beam(self, beam: Beam | None = None) -> None:
        """
        Sets the beam of the transmitter, or makes it isotropic with `None`.

        # Panics
        Panics if beam is relative, or tabulated without covering the carrier frequency.
        """
        ...

class Calibrator:
    """
    Calibrator used to model a known reference emitter (e.g. a satellite).

    The calibrator transmits a deterministic signal from a position that moves
    with constant acceleration:
    `p(t) = position + velocity * (t - epoch) + acceleration * (t - epoch)^2 / 2`.

    Its attitude orients the beam of its transmitter. The beam is evaluated once per
    sample window, for the geometry at the middle of the window.
    """

    def __init__(
        self,
        position: Vec3,
        transmitter: Transmitter,
        velocity: Vec3 | None = None,
        acceleration: Vec3 | None = None,
        epoch: float = 0.0,
        attitude: Attitude | None = None,
    ) -> None:
        """
        Creates a new calibrator.

        # Arguments
        - `position`: Position (m) of the calibrator at `epoch`.
        - `transmitter`: Transmitter of the calibrator.
        - `velocity`: Velocity (m/s) at `epoch`, zero if not given.
        - `acceleration`: Constant acceleration (m/s²), zero if not given.
        - `epoch`: Time (s) at which `position` and `velocity` are valid.
        - `attitude`: Attitude orienting the transmitter's beam, the identity orientation
          if not given.

        # Panics
        Panics if any of the kinematic parameters is not finite.
        """
        ...

    def __repr__(self) -> str: ...
    def position(self) -> Vec3:
        """Returns the calibrator position at `epoch`."""
        ...

    def velocity(self) -> Vec3:
        """Returns the calibrator velocity at `epoch`."""
        ...

    def acceleration(self) -> Vec3:
        """Returns the calibrator acceleration."""
        ...

    def epoch(self) -> float:
        """Returns the time at which `position` and `velocity` are valid."""
        ...

    def transmitter(self) -> Transmitter:
        """Returns the calibrator's transmitter."""
        ...

    def attitude(self) -> Attitude:
        """Returns the calibrator's attitude."""
        ...

    def set_attitude(self, attitude: Attitude) -> None:
        """Replaces the calibrator's attitude."""
        ...

    def set_state(
        self,
        position: Vec3,
        velocity: Vec3 | None = None,
        acceleration: Vec3 | None = None,
        epoch: float = 0.0,
    ) -> None:
        """
        Replaces the kinematic state of the calibrator.

        # Arguments
        - `position`: Position (m) at `epoch`.
        - `velocity`: Velocity (m/s) at `epoch`, zero if not given.
        - `acceleration`: Constant acceleration (m/s²), zero if not given.
        - `epoch`: Time (s) at which `position` and `velocity` are valid.

        # Panics
        Panics if any of the kinematic parameters is not finite.
        """
        ...

    def set_transmitter(self, transmitter: Transmitter) -> None:
        """Replaces the calibrator's transmitter."""
        ...

    def position_at(self, time: float) -> Vec3:
        """
        Computes the calibrator position at a given time.

        # Arguments
        - `time`: Time (s).

        # Returns
        The position (m) at `time`.
        """
        ...

def save_array(array: Array, filepath: str) -> None:
    """
    Saves the array configuration to a file in JSON format.

    Tabulated beams are written next to it in fringe's beam format, named
    `<file stem>.beam<index>.json`.

    Args:
        array: The array to serialize.
        filepath: Destination path where the array will be written.

    Raises:
        OSError: If the file cannot be created or written.
    """
    ...

def load_array(filepath: str) -> Array:
    """
    Loads an array configuration from a JSON file.

    Args:
        filepath: Path to the file containing a serialized Array.

    Returns:
        A reconstructed Array instance.

    Raises:
        OSError: If the file cannot be read or parsed.


    Panics if:
    - antenna_positions is empty
    - sample_frequency is not positive
    - downmix_frequency is negative
    - bandpass bounds are invalid or exceed Nyquist limit
    - sample_window_size is not a power of two
    - system_noise_temperature is invalid, see `Array(...)`
    - the beams or their assignment to antennas are invalid
    """
    ...

def save_sources(sources: list[Source], filepath: str) -> None:
    """
    Saves the list of sources to a file in CSV format.

    Args:
        sources: List of sources to serialize.
        filepath: Destination path where the sources will be written.

    Raises:
        OSError: If the file cannot be created or written.
    """
    ...

def load_sources(filepath: str) -> list[Source]:
    """
    Loads a list of sources from a CSV file.

    The `spectral_index_2`, `spectral_index_3`, `polarization_fraction`,
    `polarization_angle`, `circular_fraction` and `rotation_measure` columns are optional
    and default to 0.

    Args:
        filepath: Path to the file containing serialized Source entries.

    Returns:
        A list of Source objects.

    Raises:
        OSError: If the file cannot be read or parsed.

    Panics if:
    - reference_frequency is not positive
    - reference_intensity is negative
    - any spectral index is not finite
    - the polarization parameters are invalid
    """
    ...

def save_calibrators(calibrators: list[Calibrator], filepath: str) -> None:
    """
    Saves a list of calibrators to a file in JSON format.

    The transmit buffers are written next to it as raw interleaved little-endian
    32-bit float IQ files (`cf32_le`), named `<file stem>.<index>.cf32`.
    Calibrators sharing a buffer share a buffer file. Tabulated transmitter beams are
    written next to it in fringe's beam format, named `<file stem>.beam<index>.json`.

    Args:
        calibrators: The list of calibrators.
        filepath: Destination path where the calibrators will be written.

    Raises:
        OSError: If a file cannot be created or written.
    """
    ...

def load_calibrators(filepath: str) -> list[Calibrator]:
    """
    Loads a list of calibrators from a JSON file.

    Args:
        filepath: Path to the file containing a serialized list of Calibrators.

    Returns:
        A list of Calibrator objects.

    Raises:
        OSError: If a file cannot be read or parsed.

    Panics if, for a Calibrator:
    - a kinematic parameter is not finite
    - a transmitter parameter is invalid
    """
    ...

def save_beam(beam: Beam, filepath: str) -> None:
    """
    Saves a tabulated beam in fringe's native format.

    The beam is written as a JSON header at `filepath`, with its field in a raw interleaved
    little-endian 32-bit float IQ file (`cf32_le`) next to it, named `<file stem>.cf32`,
    holding `E_θ` and `E_φ` of every grid point in the order `[frequency][θ][φ][E_θ, E_φ]`.

    Raises:
        OSError: If the beam is analytic, or if a file cannot be written.
    """
    ...

def load_beam(filepath: str) -> Beam:
    """
    Loads a tabulated beam saved in fringe's native format.

    Raises:
        OSError: If a file cannot be read or does not hold a valid beam.
    """
    ...
