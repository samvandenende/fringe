const LIGHT_SPEED: f32 = 299.792458; // light speed in megameters/s

struct Params {
    sample_freq_mhz: f32,
    downmix_freq_mhz: f32,
    // first bin and number of bins of the bandpass
    band_start: u32,
    band_len: u32,
    num_bins: u32,
    sources_tile_size: u32,
    num_receivers: u32,
    receiver_noise_i: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// random phase per receiver and bin of the bandpass
@group(0) @binding(1) var<storage, read> noise_phase: array<f32>;
@group(0) @binding(2) var<storage, read_write> spectrum: array<vec2<f32>>;

fn fft_bin_frequency(bin: u32) -> f32 {
    let halfN: u32 = params.num_bins / 2u;

    if (bin < halfN) {
        return f32(bin) * params.sample_freq_mhz / f32(params.num_bins);
    } else {
        return f32(i32(bin) - i32(params.num_bins)) * params.sample_freq_mhz / f32(params.num_bins);
    }
}

// Initializes the spectra with the system noise of the receivers, zero outside the bandpass.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    let receiver_idx = id.y;

    if (bin >= params.num_bins) {
        return;
    }

    let idx = receiver_idx * params.num_bins + bin;
    let j = bin - params.band_start;
    if (bin < params.band_start || j >= params.band_len) {
        spectrum[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let phys_freq = fft_bin_frequency(bin) + params.downmix_freq_mhz;
    let lambda = LIGHT_SPEED / phys_freq;
    let eff_aperture = min(4.0, lambda * lambda / 4.0);
    let noise_ampl = sqrt(params.receiver_noise_i / eff_aperture);
    let phase = noise_phase[receiver_idx * params.band_len + j];
    spectrum[idx] = vec2<f32>(cos(phase), sin(phase)) * noise_ampl;
}
