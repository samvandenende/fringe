import json

import matplotlib.pyplot as plt
import numpy as np

import fringe as fr

"""
Simulates a calibrator satellite in low lunar orbit transmitting a periodic
pseudorandom signal, and estimates its time of arrival (ToA) at every antenna by
correlating the received samples with the known transmit buffer.
"""

ARRAY_FILE = "example_array.json"

RUNTIME = "gpu"
FREQUENCY_RESOLUTION = 4
RNG_SEED = 42
N_WINDOWS = 5
C = 299792458.0

# Calibrator, similar to the accompanying thesis
BUFFER_LEN = 16384
POWER = 10.0  # W
POSITION = fr.Vec3(2e4, -1e4, 1e5)  # m
VELOCITY = fr.Vec3(1600.0, 0.0, 0.0)  # m/s
ACCELERATION = fr.Vec3(0.0, 0.0, -1.47)  # m/s², centripetal acceleration of the orbit

# Noise intensity in the same units as the calibrator power, chosen for a
# signal-to-noise ratio of about -4 dB per sample.
SYSTEM_NOISE_INTENSITY = 3e-9


with open(ARRAY_FILE) as f:
    array_json = json.load(f)
antenna_positions = [fr.Vec3(p["x"], p["y"], p["z"]) for p in array_json["antenna_positions"]]


def make_array(system_noise_intensity):
    return fr.Array(
        antenna_positions,
        array_json["sample_frequency"],
        array_json["downmix_frequency"],
        array_json["bandpass"][0],
        array_json["bandpass"][1],
        array_json["sample_window_size"],
        system_noise_intensity,
    )


array = make_array(SYSTEM_NOISE_INTENSITY)
sample_frequency = array.sample_frequency()
sample_window_size = array.sample_window_size()


def make_buffer():
    """Periodic pseudorandom signal with a flat spectrum over the array's bandpass."""
    rng = np.random.default_rng(RNG_SEED)
    frequencies = np.fft.fftfreq(BUFFER_LEN, 1 / sample_frequency) + array.downmix_frequency()
    in_band = (frequencies >= array.bandpass_fmin()) & (frequencies <= array.bandpass_fmax())
    spectrum = np.where(in_band, np.exp(2j * np.pi * rng.random(BUFFER_LEN)), 0.0)
    buffer = np.fft.ifft(spectrum)
    return (buffer / np.sqrt(np.mean(np.abs(buffer) ** 2))).astype(np.complex64)


buffer = make_buffer()
# transmitting at the array's sample rate and downmix frequency means that the
# received baseband signal is a delayed copy of the buffer
transmitter = fr.Transmitter(
    frequency=array.downmix_frequency(),
    sample_rate=sample_frequency,
    power=POWER,
    buffer=buffer,
)
calibrator = fr.Calibrator(POSITION, transmitter, VELOCITY, ACCELERATION)


def light_time(receiver, time):
    delay = 0.0
    for _ in range(5):
        p = calibrator.position_at(time - delay)
        delay = np.linalg.norm(np.subtract(receiver.as_tuple(), p.as_tuple())) / C
    return delay


def estimate_toa(samples, start_sample, upsampling=8, chunk_size=64):
    """
    Estimates the delay of the buffer in each antenna's samples, modulo the buffer period,
    by circular cross-correlation. The correlation is interpolated by zero-padding its
    spectrum, followed by quadratic interpolation around the peak.
    """
    indices = (start_sample + np.arange(sample_window_size)) % BUFFER_LEN
    buffer_spectrum = np.conj(np.fft.fft(buffer))
    upsampled_len = BUFFER_LEN * upsampling
    half = BUFFER_LEN // 2

    delays = []
    for chunk in np.array_split(samples, max(1, len(samples) // chunk_size)):
        received = np.zeros((len(chunk), BUFFER_LEN), dtype=complex)
        received[:, indices] = chunk
        spectrum = np.fft.fft(received) * buffer_spectrum
        padded = np.zeros((len(chunk), upsampled_len), dtype=complex)
        padded[:, :half] = spectrum[:, :half]
        padded[:, -half:] = spectrum[:, half:]
        correlation = np.abs(np.fft.ifft(padded))

        peak = np.argmax(correlation, axis=1)
        rows = np.arange(len(chunk))
        left = correlation[rows, (peak - 1) % upsampled_len]
        center = correlation[rows, peak]
        right = correlation[rows, (peak + 1) % upsampled_len]
        offset = 0.5 * (left - right) / (left - 2 * center + right)
        delays.append((peak + offset) / (upsampling * sample_frequency))
    return np.concatenate(delays)


def cramer_rao_bound():
    """Lower bound on the standard deviation of the ToA of one antenna in one window."""
    # signal and noise power per sample, from simulations of each on their own
    signal_sim = fr.Simulation("cpu", make_array(0.0), FREQUENCY_RESOLUTION, RNG_SEED)
    signal_sim.set_calibrators([calibrator])
    signal_sim.start(0.0)
    signal_power = np.mean(np.abs(np.array(signal_sim.finish())) ** 2)
    noise_sim = fr.Simulation("cpu", array, FREQUENCY_RESOLUTION, RNG_SEED)
    noise_sim.start(0.0)
    noise_power = np.mean(np.abs(np.array(noise_sim.finish())) ** 2)

    # RMS bandwidth of the buffer around its spectral centroid
    power_spectrum = np.abs(np.fft.fft(buffer)) ** 2
    frequencies = np.fft.fftfreq(BUFFER_LEN, 1 / sample_frequency)
    centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    rms_bandwidth2 = np.sum((frequencies - centroid) ** 2 * power_spectrum) / np.sum(power_spectrum)

    snr = signal_power / noise_power
    print(f"SNR per sample: {10 * np.log10(snr):.1f} dB")
    # the noise only occupies the bandpass, so it is not white over the sample rate
    noise_bandwidth = array.bandpass_fmax() - array.bandpass_fmin()
    energy_to_noise_density = sample_window_size * snr * noise_bandwidth / sample_frequency
    return 1 / np.sqrt(8 * np.pi**2 * rms_bandwidth2 * energy_to_noise_density)


def wrap(delay):
    period = BUFFER_LEN / sample_frequency
    return (delay + period / 2) % period - period / 2


def run():
    sim = fr.Simulation(RUNTIME, array, FREQUENCY_RESOLUTION, RNG_SEED)
    sim.set_calibrators([calibrator])

    errors = []
    for window in range(N_WINDOWS):
        start_sample = window * sample_window_size
        # consecutive windows are contiguous in time
        sim.start()
        samples = np.array(sim.finish())
        measured = estimate_toa(samples, start_sample)
        # delay at the center of the window
        time = (start_sample + sample_window_size / 2) / sample_frequency
        expected = np.array([light_time(p, time) for p in antenna_positions])
        error = wrap(measured - expected)
        # only time differences of arrival are observable, so remove the common offset
        errors.append(error - np.mean(error))

    errors = np.array(errors)
    rms = np.sqrt(np.mean(errors**2))
    bound = cramer_rao_bound()
    print(f"RMS ToA error:     {rms * 1e12:.1f} ps ({rms * C * 1e3:.1f} mm)")
    print(f"Cramér-Rao bound:  {bound * 1e12:.1f} ps ({bound * C * 1e3:.1f} mm)")

    side = int(np.sqrt(len(antenna_positions)))
    plt.imshow(np.mean(errors, axis=0).reshape((side, side)) * C * 1e3, cmap="coolwarm")
    plt.colorbar(label="Mean ToA error [mm]")
    plt.title(f"Calibrator ToA error over {N_WINDOWS} windows")
    plt.show()


if __name__ == "__main__":
    run()
