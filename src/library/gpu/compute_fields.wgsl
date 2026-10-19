// BLOCK in compute_spectra.wgsl
const BLOCK: u32 = 16u;

struct Source {
    dir: vec4<f32>,
    f0: f32,
    i0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

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
@group(0) @binding(1) var<storage, read> sources: array<Source>;
// random phase per source and bin of the bandpass
@group(0) @binding(2) var<storage, read> random_phase: array<f32>;
// field per source and bin of the bandpass, without the geometric delay, padded to a
// multiple of BLOCK bins
@group(0) @binding(3) var<storage, read_write> fields: array<vec2<f32>>;

fn fft_bin_frequency(bin: u32) -> f32 {
    let halfN: u32 = params.num_bins / 2u;

    if (bin < halfN) {
        return f32(bin) * params.sample_freq_mhz / f32(params.num_bins);
    } else {
        return f32(i32(bin) - i32(params.num_bins)) * params.sample_freq_mhz / f32(params.num_bins);
    }
}

// Computes the field of every source of the tile, received with a cos²(θ) gain.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let s = id.y;

    if (j >= params.band_len) {
        return;
    }

    let phys_freq = fft_bin_frequency(params.band_start + j) + params.downmix_freq_mhz;
    let src = sources[s];

    let gain = src.dir.z * src.dir.z; // cos²(θ) gain model
    let x = log(phys_freq / src.f0);
    let ampl = gain * src.i0 * exp(x * (src.a1 + x * (src.a2 + x * src.a3)));

    let phase = random_phase[s * params.band_len + j];
    let stride = (params.band_len + BLOCK - 1u) / BLOCK * BLOCK;
    fields[s * stride + j] = vec2<f32>(cos(phase), sin(phase)) * sqrt(ampl);
}
