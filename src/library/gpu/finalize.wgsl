struct Params {
    num_bins: u32,
    window_size: u32,
    offset: u32,
    norm: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;

// Selects the middle `window_size` samples of each receiver's synthesis window and normalizes them.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let receiver_idx = id.y;

    if (n >= params.window_size) {
        return;
    }

    output[receiver_idx * params.window_size + n] =
        samples[receiver_idx * params.num_bins + params.offset + n] / params.norm;
}
