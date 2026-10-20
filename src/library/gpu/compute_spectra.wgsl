const LIGHT_SPEED: f32 = 299.792458; // light speed in megameters/s
const TAU: f32 = 6.283185307179586;
// set in the group of receivers whose beam has no polarization
const SCALAR_GROUP: u32 = 0x80000000u;
// number of consecutive bins of the bandpass evaluated by one invocation, written out in main
const BLOCK: u32 = 16u;

struct Receiver {
    x: f32,
    y: f32,
    z: f32,
    // index of the receiver group, with SCALAR_GROUP set for beams without polarization
    group: u32,
};

struct Source {
    dir: vec4<f32>,
    f0: f32,
    i0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    pol_frac: f32,
    pol_angle: f32,
    circ_frac: f32,
    rm: f32,
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
    num_groups: u32,
    // frequency nodes of the beam table
    beam_f0_mhz: f32,
    beam_df_mhz: f32,
    beam_nodes: u32,
    // number of random phases per source and bin
    polarizations: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(2) var<storage, read> sources: array<Source>;
// received field per source, group and bin of the bandpass, without the geometric delay
@group(0) @binding(3) var<storage, read> fields: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> spectrum: array<vec2<f32>>;

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

// Adds the fields of the sources of a tile, delayed geometrically, to a block of BLOCK
// consecutive bins of a receiver's spectrum.
//
// The invocations of a workgroup evaluate the same bins of different receivers, which mostly
// read the same fields. The geometric phase is advanced from bin to bin by a rotation. The
// block is written out without loops, so that the fields of a source are loaded together.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let receiver_idx = id.x;
    let first = id.y * BLOCK;

    if (receiver_idx >= params.num_receivers || first >= params.band_len) {
        return;
    }

    let rx = receivers[receiver_idx];
    let group = rx.group & ~SCALAR_GROUP;
    let position = vec3<f32>(rx.x, rx.y, rx.z);
    let count = min(BLOCK, params.band_len - first);
    let idx = receiver_idx * params.num_bins + params.band_start + first;

    var sum: array<vec2<f32>, BLOCK>;
    for (var b = 0u; b < count; b++) {
        sum[b] = spectrum[idx + b];
    }

    let first_freq = fft_bin_frequency(params.band_start + first) + params.downmix_freq_mhz;
    let bin_width = params.sample_freq_mhz / f32(params.num_bins);
    // the fields of a source and group are padded to a multiple of BLOCK bins
    let stride = (params.band_len + BLOCK - 1u) / BLOCK * (BLOCK / 2u);

    for (var s = 0u; s < params.sources_tile_size; s++) {
        // geometric delay in µs
        let delay = dot(sources[s].dir.xyz, position) / LIGHT_SPEED;
        let phase = TAU * fract(first_freq * delay);
        let step_phase = TAU * fract(bin_width * delay);
        let step = vec2<f32>(cos(step_phase), sin(step_phase));

        let base = (s * params.num_groups + group) * stride + first / 2u;
        let f0 = fields[base];
        let f1 = fields[base + 1u];
        let f2 = fields[base + 2u];
        let f3 = fields[base + 3u];
        let f4 = fields[base + 4u];
        let f5 = fields[base + 5u];
        let f6 = fields[base + 6u];
        let f7 = fields[base + 7u];

        var rotation = vec2<f32>(cos(phase), sin(phase));
        sum[0] += complex_mul(f0.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[1] += complex_mul(f0.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[2] += complex_mul(f1.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[3] += complex_mul(f1.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[4] += complex_mul(f2.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[5] += complex_mul(f2.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[6] += complex_mul(f3.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[7] += complex_mul(f3.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[8] += complex_mul(f4.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[9] += complex_mul(f4.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[10] += complex_mul(f5.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[11] += complex_mul(f5.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[12] += complex_mul(f6.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[13] += complex_mul(f6.zw, rotation);
        rotation = complex_mul(rotation, step);
        sum[14] += complex_mul(f7.xy, rotation);
        rotation = complex_mul(rotation, step);
        sum[15] += complex_mul(f7.zw, rotation);
    }

    for (var b = 0u; b < count; b++) {
        spectrum[idx + b] = sum[b];
    }
}
