const TAU: f32 = 6.283185307179586;

struct Params {
    n_s: u32,
    stage: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;


fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn twiddle(k: u32, N: u32) -> vec2<f32> {
    let angle = TAU * f32(k) / f32(N);
    return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let spectrum = gid.y;
    let n_s = params.n_s;

    if (i >= n_s / 2) {
        return;
    }

    let base = spectrum * n_s;

    let L = 1u << params.stage;

    let a = input[base + i];
    let b = input[base + i + n_s / 2];

    let j = i % L;
    let tw = twiddle(j, L * 2);
    let b_tw = complex_mul(b, tw);

    let block_idx = i / L;
    let idx_out = block_idx * (L * 2) + j;

    output[base + idx_out] = a + b_tw;
    output[base + idx_out + L] = a - b_tw;
}
