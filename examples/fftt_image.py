import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import fringe as fr

ARRAY_FILE = "example_array.json"
SOURCES_FILE = "example_sky_model.csv"

RUNTIME = "gpu"
FREQUENCY_RESOLUTION = 4
RNG_SEED = 42
N_INTEGRATIONS = 100
IMAGE_FREQ = 30e6


array = fr.load_array(ARRAY_FILE)
sources = fr.load_sources(SOURCES_FILE)

print("=== ARRAY MODEL ===")
print(array)
print("\n=== SKY MODEL ===")
print("Sources([")
for i in range(3):
    print(f"    {sources[i]}")
print("    ...\n]")

sample_window_size = array.sample_window_size()
sample_frequency = array.sample_frequency()

sim = fr.Simulation(RUNTIME, array, FREQUENCY_RESOLUTION, RNG_SEED)
sim.set_sources(sources)


def sim_result_to_fftt_image(result):
    spectra = np.fft.fft(
        result, axis=1, norm="ortho"
    )  # experiment with window functions
    bin = round(IMAGE_FREQ * sample_window_size / sample_frequency)
    e_field = spectra[:, bin]
    image = np.fft.fftshift(np.fft.ifft2(e_field.reshape((32, 32))))
    return np.abs(image) ** 2


def run():
    images = []
    sim.start()
    for _ in tqdm(range(N_INTEGRATIONS - 1)):
        result = sim.finish()
        sim.start()
        images.append(sim_result_to_fftt_image(result))
    result = sim.finish()
    images.append(sim_result_to_fftt_image(result))

    image = np.zeros_like(images[0])
    for img in images:
        image += img / N_INTEGRATIONS

    plt.imshow(image, cmap="plasma")
    plt.colorbar(label="Brightness")
    plt.title(f"Sky image at {IMAGE_FREQ / 1e6}MHz")
    plt.show()


if __name__ == "__main__":
    run()
