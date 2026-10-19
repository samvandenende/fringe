struct Params {
    sample_freq_mhz: f32,
    // first bin and number of bins of the bandpass
    band_start: u32,
    band_len: u32,
    num_bins: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> signal_spectrum: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> spectrum: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    let receiver_idx = id.y;

    if (bin >= params.num_bins) {
        return;
    }

    // same bandpass as the sky in compute_noise.wgsl
    if (bin < params.band_start || bin - params.band_start >= params.band_len) {
        return;
    }

    let idx = receiver_idx * params.num_bins + bin;
    spectrum[idx] += signal_spectrum[idx] * params.scale;
}
