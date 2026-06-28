import matplotlib.pyplot as plt
import numpy as np

import fringe as fr

"""
Tests to see if the different runtimes provided by Fringe
produce identical output, upto floating point rounding errors.
"""


def show_results(samples_cpu, samples_gpu, fs=1.0, window="hann"):
    cpu = np.asarray(samples_cpu)
    gpu = np.asarray(samples_gpu)

    if cpu.shape != gpu.shape:
        raise ValueError(f"Shape mismatch: CPU {cpu.shape}, GPU {gpu.shape}")

    n = cpu.size
    t = np.arange(n)

    # ============================================================
    # 1. COMPLEX RELATIVE ERROR (core metric)
    # ============================================================
    err = gpu - cpu

    # avoid division blow-ups
    eps = 1e-15
    rel_err_complex = np.abs(err) / (np.abs(cpu) + eps)

    # RMS EVM (standard RF metric)
    evm_rms = np.sqrt(np.mean(np.abs(err) ** 2)) / (
        np.sqrt(np.mean(np.abs(cpu) ** 2)) + eps
    )
    evm_db = 20 * np.log10(evm_rms + eps)

    # per-sample normalized error statistics
    mean_rel = np.mean(rel_err_complex)
    p95_rel = np.percentile(rel_err_complex, 95)
    max_rel = np.max(rel_err_complex)

    # ============================================================
    # 2. TIME DOMAIN (normalized view)
    # ============================================================
    cpu_mag = np.abs(cpu)
    gpu_mag = np.abs(gpu)

    rel_mag_err = np.abs(gpu_mag - cpu_mag) / (cpu_mag + eps)

    # phase error (wrapped small-angle approximation)
    phase_err = np.angle(gpu * np.conj(cpu))

    # ============================================================
    # 3. FFT DOMAIN (fractional spectral error)
    # ============================================================
    if window == "hann":
        w = np.hanning(n)
    else:
        w = np.ones(n)

    cpu_f = cpu * w
    gpu_f = gpu * w

    cpu_fft = np.fft.fftshift(np.fft.fft(cpu_f))
    gpu_fft = np.fft.fftshift(np.fft.fft(gpu_f))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))

    cpu_spec = np.abs(cpu_fft) / n
    gpu_spec = np.abs(gpu_fft) / n

    spec_rel_err = np.abs(gpu_spec - cpu_spec) / (cpu_spec + eps)

    # log for visibility
    spec_rel_err_db = 20 * np.log10(spec_rel_err + eps)

    # ============================================================
    # 4. PLOTS
    # ============================================================
    _fig = plt.figure(figsize=(15, 10))

    # ---- Relative magnitude error ----
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, rel_mag_err, c="tab:blue")
    ax1.set_title("Relative magnitude error |Δx| / |x_cpu|")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Fractional error")
    ax1.grid(True, alpha=0.3)

    # ---- Phase error ----
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, phase_err, c="tab:green")
    ax2.set_title("Phase error (GPU ⊖ CPU)")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Radians")
    ax2.grid(True, alpha=0.3)

    # ---- EVM summary + distribution ----
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(rel_err_complex, bins=80, alpha=0.8, color="tab:red")
    ax3.set_title("Distribution of normalized complex error")
    ax3.set_xlabel("|x_gpu - x_cpu| / |x_cpu|")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    # annotate EVM
    ax3.text(
        0.65,
        0.85,
        f"EVM RMS: {evm_rms:.4e}\nEVM (dB): {evm_db:.2f} dB\nMean: {mean_rel:.2e}",
        transform=ax3.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # ---- Spectral relative error ----
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(freq, spec_rel_err_db, c="tab:orange")
    ax4.set_title("Spectral relative error (dB)")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.set_ylabel("Relative error [dB]")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 5. PRINT SUMMARY
    # ============================================================
    print("=== Normalized comparison (CPU vs GPU) ===")
    print(f"EVM RMS:        {evm_rms:.6e}")
    print(f"EVM (dB):       {evm_db:.2f} dB")
    print(f"Mean rel error: {mean_rel:.6e}")
    print(f"95% rel error:  {p95_rel:.6e}")
    print(f"Max rel error:  {max_rel:.6e}")


def test_system_noise(
    system_noise_intensity=1.0,
    sample_frequency=120e6,
    downmix_frequency=45e6,
    bandpass_fmin=5e6,
    bandpass_fmax=55e6,
    sample_window_size=2**12,
    frequency_resolution=4,
):
    antenna_position = fr.Vec3(0.0, 0.0, 0.0)
    array = fr.Array(
        [antenna_position],
        sample_frequency,
        downmix_frequency,
        bandpass_fmin,
        bandpass_fmax,
        sample_window_size,
        system_noise_intensity,
    )

    sim_cpu = fr.Simulation("cpu", array, frequency_resolution, 42)
    sim_gpu = fr.Simulation("gpu", array, frequency_resolution, 42)

    sim_cpu.start()
    sim_gpu.start()
    samples_cpu = np.array(sim_cpu.finish()[0])
    samples_gpu = np.array(sim_gpu.finish()[0])

    show_results(samples_cpu, samples_gpu)


def test_source(
    source_reference_frequency=75e6,
    source_reference_intensity=1.0,
    source_spectral_index=0.0,
    sample_frequency=120e6,
    downmix_frequency=45e6,
    bandpass_fmin=5e6,
    bandpass_fmax=55e6,
    sample_window_size=2**12,
    frequency_resolution=4,
):
    antenna_position = fr.Vec3(0.0, 0.0, 0.0)
    array = fr.Array(
        [antenna_position],
        sample_frequency,
        downmix_frequency,
        bandpass_fmin,
        bandpass_fmax,
        sample_window_size,
        0.0,
    )

    sim_cpu = fr.Simulation("cpu", array, frequency_resolution, 42)
    sim_gpu = fr.Simulation("gpu", array, frequency_resolution, 42)

    source = fr.Source(
        fr.Vec3(0.0, 0.0, 1.0),
        source_reference_frequency,
        source_reference_intensity,
        source_spectral_index,
    )
    sim_cpu.set_sources([source])
    sim_gpu.set_sources([source])

    sim_cpu.start()
    sim_gpu.start()
    samples_cpu = np.array(sim_cpu.finish()[0])
    samples_gpu = np.array(sim_gpu.finish()[0])

    show_results(samples_cpu, samples_gpu)


def test_many_sources(
    sample_frequency=120e6,
    downmix_frequency=0.0,
    bandpass_fmin=15e6,
    bandpass_fmax=45e6,
    sample_window_size=2**12,
    frequency_resolution=4,
):
    antenna_position = fr.Vec3(0.0, 0.0, 0.0)
    array = fr.Array(
        [antenna_position],
        sample_frequency,
        downmix_frequency,
        bandpass_fmin,
        bandpass_fmax,
        sample_window_size,
        0.0,
    )

    sim_cpu = fr.Simulation("cpu", array, frequency_resolution, 42)
    sim_gpu = fr.Simulation("gpu", array, frequency_resolution, 42)

    sources = fr.load_sources("../examples/example_sky_model.csv")
    sim_cpu.set_sources(sources)
    sim_gpu.set_sources(sources)

    sim_cpu.start()
    sim_gpu.start()
    samples_cpu = np.array(sim_cpu.finish()[0])
    samples_gpu = np.array(sim_gpu.finish()[0])

    show_results(samples_cpu, samples_gpu)


print("=== System noise only ===")
test_system_noise()
print("")
print("=== One source only ===")
test_source()
print("")
print("=== Many sources ===")
test_many_sources()
