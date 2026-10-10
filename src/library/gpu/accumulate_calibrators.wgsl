struct Params {
    sample_freq_mhz: f32,
    bandpass_fmin_mhz: f32,
    bandpass_fmax_mhz: f32,
    num_bins: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> signal_spectrum: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> spectrum: array<vec2<f32>>;

fn fft_bin_frequency(bin: u32) -> f32 {
    let halfN: u32 = params.num_bins / 2u;

    if (bin < halfN) {
        return f32(bin) * params.sample_freq_mhz / f32(params.num_bins);
    } else {
        return f32(i32(bin) - i32(params.num_bins)) * params.sample_freq_mhz / f32(params.num_bins);
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    let receiver_idx = id.y;

    if (bin >= params.num_bins) {
        return;
    }

    // same bandpass as the sky in compute_spectra.wgsl
    let bin_freq = fft_bin_frequency(bin);
    if (bin_freq < params.bandpass_fmin_mhz || bin_freq > params.bandpass_fmax_mhz) {
        return;
    }

    let idx = receiver_idx * params.num_bins + bin;
    spectrum[idx] += signal_spectrum[idx] * params.scale;
}
