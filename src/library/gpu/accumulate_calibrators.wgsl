struct Params {
    sample_freq_mhz: f32,
    // first bin and number of bins of the bandpass
    band_start: u32,
    band_len: u32,
    num_bins: u32,
    scale: f32,
    downmix_freq_mhz: f32,
    // frequency nodes of the link response, none if link_nodes is 0
    link_f0_mhz: f32,
    link_df_mhz: f32,
    link_nodes: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> signal_spectrum: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> spectrum: array<vec2<f32>>;
// link response per receiver and frequency node
@group(0) @binding(3) var<storage, read> links: array<vec2<f32>>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

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

    // same bandpass as the sky in compute_noise.wgsl
    if (bin < params.band_start || bin - params.band_start >= params.band_len) {
        return;
    }
    let bin_freq = fft_bin_frequency(bin);

    let idx = receiver_idx * params.num_bins + bin;
    var value = signal_spectrum[idx] * params.scale;
    if (params.link_nodes > 0u) {
        // filter by the link response of the calibrator, interpolated between the nodes
        let u = clamp((bin_freq + params.downmix_freq_mhz - params.link_f0_mhz) / params.link_df_mhz, 0.0, f32(params.link_nodes - 1u));
        let k = min(u32(floor(u)), params.link_nodes - 2u);
        let base = receiver_idx * params.link_nodes;
        let link = mix(links[base + k], links[base + k + 1u], u - f32(k));
        value = complex_mul(value, link);
    }
    spectrum[idx] += value;
}
