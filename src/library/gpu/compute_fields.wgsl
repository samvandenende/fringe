const LIGHT_SPEED: f32 = 299.792458; // light speed in megameters/s

struct Source {
    dir: vec4<f32>,
    f0: f32,
    i0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    // linear polarization fraction, angle (rad), circular fraction and rotation measure (rad/m²)
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
@group(0) @binding(1) var<storage, read> sources: array<Source>;
// random phases per source and bin of the bandpass, for the north and east polarization, or
// only one if no receiver has polarization
@group(0) @binding(2) var<storage, read> random_phase: array<f32>;
// Jones row (north re, north im, east re, east im) per source, group and frequency node
@group(0) @binding(3) var<storage, read> beam: array<vec4<f32>>;
// 1 for the groups whose beam has no polarization
@group(0) @binding(4) var<storage, read> scalar_group: array<u32>;
// received field per source, group and bin of the bandpass, without the geometric delay,
// padded to a multiple of BLOCK bins
@group(0) @binding(5) var<storage, read_write> fields: array<vec2<f32>>;

// BLOCK in compute_spectra.wgsl
const BLOCK: u32 = 16u;

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

// Computes the field of every source of the tile, received through the Jones row of every group.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    let s = id.y;
    let group = id.z;

    if (j >= params.band_len) {
        return;
    }

    let phys_freq = fft_bin_frequency(params.band_start + j) + params.downmix_freq_mhz;
    let lambda = LIGHT_SPEED / phys_freq;

    // beam table nodes bracketing the frequency
    var k0 = 0u;
    var k1 = 0u;
    var w = 0.0;
    if (params.beam_nodes > 1u) {
        let u = clamp((phys_freq - params.beam_f0_mhz) / params.beam_df_mhz, 0.0, f32(params.beam_nodes - 1u));
        k0 = min(u32(floor(u)), params.beam_nodes - 2u);
        k1 = k0 + 1u;
        w = u - f32(k0);
    }

    let src = sources[s];

    let x = log(phys_freq / src.f0);
    let intensity = src.i0 * exp(x * (src.a1 + x * (src.a2 + x * src.a3)));

    let base = (s * params.num_groups + group) * params.beam_nodes;
    let jones = mix(beam[base + k0], beam[base + k1], w);
    let j_north = jones.xy;
    let j_east = jones.zw;

    let phase_idx = (s * params.band_len + j) * params.polarizations;
    let phase1 = random_phase[phase_idx];

    var y: vec2<f32>;
    if (scalar_group[group] != 0u) {
        // beams without polarization receive half the flux density
        y = vec2<f32>(cos(phase1), sin(phase1)) * (j_north.x * sqrt(0.5 * intensity));
    } else {
        let z1 = vec2<f32>(cos(phase1), sin(phase1));
        let phase2 = random_phase[phase_idx + 1u];
        let z2 = vec2<f32>(cos(phase2), sin(phase2));
        if (src.pol_frac == 0.0 && src.circ_frac == 0.0) {
            y = (complex_mul(j_north, z1) + complex_mul(j_east, z2)) * sqrt(0.5 * intensity);
        } else {
            // field (E_north, E_east) = (l11 z1, l21 z1 + l22 z2), with the coherency of the Stokes parameters
            let chi = src.pol_angle + src.rm * lambda * lambda;
            let p = src.pol_frac * intensity;
            let q = p * cos(2.0 * chi);
            let u = p * sin(2.0 * chi);
            let v = src.circ_frac * intensity;
            let b11 = 0.5 * (intensity + q);
            let b22 = 0.5 * (intensity - q);
            let b21 = vec2<f32>(0.5 * u, -0.5 * v);
            let l11 = sqrt(max(b11, 0.0));
            var l21 = vec2<f32>(0.0, 0.0);
            if (l11 > 0.0) {
                l21 = b21 / l11;
            }
            let l22 = sqrt(max(b22 - dot(l21, l21), 0.0));
            y = complex_mul(j_north * l11 + complex_mul(j_east, l21), z1) + complex_mul(j_east, z2) * l22;
        }
    }

    let stride = (params.band_len + BLOCK - 1u) / BLOCK * BLOCK;
    fields[(s * params.num_groups + group) * stride + j] = y;
}
