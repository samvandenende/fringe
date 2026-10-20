import json
import os

import numpy as np

import fringe as fr

"""
Simulates a dual-polarized array observing a polarized source, and a calibrator with a
directive transmitter pointing at the array. The Stokes parameters of the source are
recovered from the correlations of the two polarizations of every antenna.

The antenna beam is read from an openEMS far field (nf2ff HDF5 file) if one is given in
OPENEMS_FILE. Otherwise a beam is tabulated from numpy arrays, the way the far field of
any other antenna simulation software can be imported.
"""

ARRAY_FILE = "example_array.json"
OPENEMS_FILE = None  # e.g. "nf2ff.h5"

RUNTIME = "cpu"
FREQUENCY_RESOLUTION = 4
RNG_SEED = 42


def tabulated_dipole_over_ground(frequencies):
    """
    Far field of a horizontal half-wave dipole along x, a quarter wavelength above an
    infinite ground plane at 30 MHz, over the upper hemisphere, normalized to directivity.
    """
    theta = np.radians(np.arange(0.0, 90.0 + 1e-9, 2.0))
    phi = np.radians(np.arange(0.0, 360.0, 2.0))
    t, p = np.meshgrid(theta, phi, indexing="ij")
    e_theta, e_phi = [], []
    for f in frequencies:
        k_h = 2 * np.pi * f / 299792458.0 * (299792458.0 / 30e6 / 4)
        # direction cosine along the dipole
        cos_x = np.sin(t) * np.cos(p)
        sin_x = np.sqrt(np.maximum(1 - cos_x**2, 1e-12))
        element = np.cos(np.pi / 2 * cos_x) / sin_x
        ground = 2j * np.sin(k_h * np.cos(t))
        # unit vector of the field: the dipole axis perpendicular to the direction
        e_t = np.cos(t) * np.cos(p) / sin_x
        e_p = -np.sin(p) / sin_x
        amplitude = element * ground
        e_theta.append(amplitude * e_t)
        e_phi.append(amplitude * e_p)
    e_theta, e_phi = np.array(e_theta), np.array(e_phi)
    # normalize to directivity, integrating |E|² over the hemisphere
    weights = np.sin(theta)[None, :, None] * np.radians(2.0) ** 2
    power = np.sum((np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2) * weights, axis=(1, 2))
    scale = np.sqrt(4 * np.pi / power)[:, None, None]
    return fr.Beam(
        list(frequencies),
        theta.tolist(),
        phi.tolist(),
        (e_theta * scale).ravel().tolist(),
        (e_phi * scale).ravel().tolist(),
        "directivity",
    )


with open(ARRAY_FILE) as f:
    array_json = json.load(f)
positions = [fr.Vec3(p["x"], p["y"], p["z"]) for p in array_json["antenna_positions"]]

if OPENEMS_FILE and os.path.exists(OPENEMS_FILE):
    beam = fr.Beam.load_openems(OPENEMS_FILE)
else:
    beam = tabulated_dipole_over_ground(np.arange(10e6, 61e6, 5e6))
print(beam.diagnostics())

# every antenna is an X and a Y receiver; nothing is received from below the horizon
array = fr.Array.dual_polarized(
    positions,
    array_json["sample_frequency"],
    array_json["downmix_frequency"],
    array_json["bandpass"][0],
    array_json["bandpass"][1],
    array_json["sample_window_size"],
    0.0,
    beam=beam,
    horizon=fr.Vec3(0.0, 0.0, 1.0),
)

# a polarized source at the zenith, where north is -x and east is +y
p, chi, v = 0.5, np.radians(30.0), 0.1
source = fr.Source(
    fr.Vec3(0.0, 0.0, 1.0),
    30e6,
    100.0,
    0.0,
    polarization_fraction=p,
    polarization_angle=chi,
    circular_fraction=v,
)

sim = fr.Simulation(RUNTIME, array, FREQUENCY_RESOLUTION, RNG_SEED)
sim.set_sources([source])
sim.start(0.0)
samples = np.array(sim.finish())
x, y = samples[0::2], samples[1::2]
xx = np.mean(np.abs(x) ** 2)
yy = np.mean(np.abs(y) ** 2)
xy = np.mean(x * np.conj(y))
i = xx + yy
print("Stokes parameters of the source, relative to I:")
print(f"  Q: {(xx - yy) / i:+.3f} (expected {p * np.cos(2 * chi):+.3f})")
print(f"  U: {-2 * xy.real / i:+.3f} (expected {p * np.sin(2 * chi):+.3f})")
print(f"  V: {-2 * xy.imag / i:+.3f} (expected {v:+.3f})")

# a calibrator with a short dipole pointing at the array center, received by the X and Y
# receivers of the first antenna
transmitter = fr.Transmitter(
    frequency=30e6,
    sample_rate=1e6,
    power=1.0,
    buffer=[1.0],
    beam=fr.Beam.short_dipole(),
)
calibrator = fr.Calibrator(
    fr.Vec3(2e4, 0.0, 1e5),
    transmitter,
    attitude=fr.Attitude.point_at(fr.Vec3(0.0, 0.0, 0.0), fr.Vec3(1.0, 0.0, 0.0)),
)
sim.set_sources([])
sim.set_calibrators([calibrator])
sim.start(0.0)
samples = np.array(sim.finish())
print("Calibrator power received by antenna 0 (X, Y):")
print(f"  {np.mean(np.abs(samples[0]) ** 2):.3e}, {np.mean(np.abs(samples[1]) ** 2):.3e} Jy")
