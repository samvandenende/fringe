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
    def set_calibrator(self, calibrator: Calibrator) -> None:
        """
        Set or update the calibrator source.

        This calibrator will be used on the next call to `Simulation::start`

        # Arguments
        - `calibrator`: `Calibrator` object with position and intensity.
        """
        ...
    def start(self) -> None:
        """
        Start simulation of a batch of time-domain signals.

        The simulation work is dispatched to the configured runtime.
        `Simulation::finish` must be called to obtain the results before
        a next call to `Simulation::start`.
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
    def calibrator_frequency_domain_signal(self) -> list[complex]:
        """
        todo
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

class Calibrator:
    """
    Calibrator used to model a known reference emitter (e.g. a satellite).

    The calibrator acts as a deterministic signal source which can
    be used for system calibration.
    """

    def __init__(self, position: Vec3, intensity: float) -> None:
        """
        Creates a new calibrator.

        # Arguments
        - `position`: Position of the calibration source.
        - `intensity`: Signal intensity of the calibrator.

        # Panics
        Panics if intensity is negative.
        """
        ...

    def __repr__(self) -> str: ...
    def position(self) -> Vec3:
        """Returns the calibrator position."""
        ...

    def intensity(self) -> float:
        """Returns the calibrator intensity."""
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

def save_calibrator(calibrator: Calibrator, filepath: str) -> None:
    """
    Saves the calibrator to a file in JSON format.

    Args:
        calibrator: The calibrator to serialize.
        filepath: Destination path where the calibrator will be written.

    Raises:
        OSError: If the file cannot be created or written.
    """
    ...

def load_calibrator(filepath: str) -> Calibrator:
    """
    Loads a calibrator from a JSON file.

    Args:
        filepath: Path to the file containing a serialized Calibrator.

    Returns:
        A reconstructed Calibrator instance.

    Raises:
        OSError: If the file cannot be read or parsed.

    Panics if:
    - intensity is negative
    """
