from collections.abc import Sequence

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
    def finish(self) -> list[list[complex]]:
        """
        Collects simulation results from the runtime.

        Must be called after `Simulation::start`.

        # Returns
        A 2D vector of complex-valued antenna samples:
        - Outer dimension: antennas in the array
        - Inner dimension: time-domain samples
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

class Array:
    """
    Array model and its signal acquisition parameters.

    This structure defines:
    - array geometry
    - sampling configuration
    - frequency conversion parameters
    - simulation noise characteristics
    """

    def __init__(
        self,
        antenna_positions: list[Vec3],
        sample_frequency: float,
        downmix_frequency: float,
        bandpass_fmin: float,
        bandpass_fmax: float,
        sample_window_size: int,
        system_noise_intensity: float,
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
        - `system_noise_intensity`: System noise intensity.

        # Panics
        Panics if:
        - sample_frequency is not positive
        - downmix_frequency is negative
        - bandpass bounds are invalid or exceed Nyquist limit
        - sample_window_size is not a power of two
        - system_noise_intensity is negative
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

    def system_noise_intensity(self) -> float:
        """Returns the system noise intensity."""
        ...

class Source:
    """
    Source in the simulated sky model.

    Each source emits a frequency-dependent signal characterized by a reference
    intensity and spectral index, and is located at a fixed direction vector.
    """

    def __init__(
        self,
        direction: Vec3,
        reference_frequency: float,
        reference_intensity: float,
        spectral_index: float,
    ) -> None:
        """
        Creates a new signal source.

        # Arguments
        - `direction`: Unit vector indicating source direction in space.
        - `reference_frequency`: Frequency at which intensity is defined.
        - `reference_intensity`: Signal strength at the reference frequency.
        - `spectral_index`: Power-law spectral index of the source.

        # Panics
        Panics if:
        - reference_frequency is not positive
        - reference_intensity is negative
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
        """Returns the reference intensity at the reference frequency."""
        ...

    def spectral_index(self) -> float:
        """Returns the spectral index used in the power-law intensity model."""
        ...

    def intensity(self, frequency: float) -> float:
        """
        Computes the intensity at a given frequency using a power-law model.

        # Panics
        Panics if `frequency <= 0.0`.
        """
        ...

class Transmitter:
    """
    Transmitter of a calibrator, configured like a software defined radio.

    The transmitter plays its complex baseband buffer cyclically at `sample_rate`,
    upconverted to the carrier `frequency`. Buffer samples are relative levels, like
    the full scale of a DAC: `power` is the radiated power for a buffer with unit RMS.
    """

    def __init__(
        self,
        frequency: float,
        sample_rate: float,
        power: float,
        buffer: Sequence[complex],
        bandwidth: float | None = None,
        start_time: float = 0.0,
    ) -> None:
        """
        Creates a new transmitter.

        # Arguments
        - `frequency`: Carrier frequency (Hz).
        - `sample_rate`: Rate at which buffer samples are played out (Hz).
        - `power`: Equivalent isotropically radiated power (W) for a buffer with unit RMS.
        - `buffer`: Complex baseband samples, transmitted cyclically.
        - `bandwidth`: Optional bandwidth of the reconstruction filter (Hz).
        - `start_time`: Time (s) at which the first buffer sample is transmitted.

        # Panics
        Panics if:
        - frequency is negative
        - sample_rate is not positive
        - power is negative
        - bandwidth is not positive
        - buffer is empty or contains non-finite samples
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
        """Returns the radiated power in W for a buffer with unit RMS."""
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
        Sets the radiated power in W for a buffer with unit RMS.

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
        Panics if buffer is empty or contains non-finite samples.
        """
        ...

    def set_start_time(self, start_time: float) -> None:
        """
        Sets the time in s at which the first buffer sample is transmitted.

        # Panics
        Panics if start_time is not finite.
        """
        ...

class Calibrator:
    """
    Calibrator used to model a known reference emitter (e.g. a satellite).

    The calibrator transmits a deterministic signal from a position that moves
    with constant acceleration:
    `p(t) = position + velocity * (t - epoch) + acceleration * (t - epoch)^2 / 2`.
    """

    def __init__(
        self,
        position: Vec3,
        transmitter: Transmitter,
        velocity: Vec3 | None = None,
        acceleration: Vec3 | None = None,
        epoch: float = 0.0,
    ) -> None:
        """
        Creates a new calibrator.

        # Arguments
        - `position`: Position (m) of the calibrator at `epoch`.
        - `transmitter`: Transmitter of the calibrator.
        - `velocity`: Velocity (m/s) at `epoch`, zero if not given.
        - `acceleration`: Constant acceleration (m/s²), zero if not given.
        - `epoch`: Time (s) at which `position` and `velocity` are valid.

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
    - system_noise_intensity is negative
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

    Args:
        filepath: Path to the file containing serialized Source entries.

    Returns:
        A list of Source objects.

    Raises:
        OSError: If the file cannot be read or parsed.

    Panics if:
    - reference_frequency is not positive
    - reference_intensity is negative
    """
    ...

def save_calibrators(calibrators: list[Calibrator], filepath: str) -> None:
    """
    Saves a list of calibrators to a file in JSON format.

    The transmit buffers are written next to it as raw interleaved little-endian
    32-bit float IQ files (`cf32_le`), named `<file stem>.<index>.cf32`.
    Calibrators sharing a buffer share a buffer file.

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
