const TAU: f32 = 6.283185307179586;

struct Calibrator {
    frequency_hz: f32,
    oversampled_rate_hz: f32,
    buffer_len: u32,
    buffer_offset: u32,
};

struct CommonSample {
    buffer_index: u32,
    buffer_frac: f32,
    carrier: f32,
};

struct Segment {
    buffer_index: u32,
    buffer_frac: f32,
    carrier: f32,
    amplitude: f32,
    delay_s: vec4<f32>,
};

struct Params {
    num_bins: u32,
    num_calibrators: u32,
    num_segments: u32,
    segment_len: u32,
    segment_half_len: f32,
    // number of samples before the first output sample
    offset: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> calibrators: array<Calibrator>;
@group(0) @binding(2) var<storage, read> common_samples: array<CommonSample>;
@group(0) @binding(3) var<storage, read> segments: array<Segment>;
@group(0) @binding(4) var<storage, read> buffers: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> signal: array<vec2<f32>>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// raised cosine taper over the samples that are discarded from the output, see `calibrator::taper`
fn taper(m: u32) -> f32 {
    let distance = min(m, params.num_bins - 1u - m);
    if (distance >= params.offset) {
        return 1.0;
    }
    let u = (f32(distance) + 0.5) / f32(params.offset);
    return 0.5 - 0.5 * cos(TAU / 2.0 * u);
}

fn buffer_sample(calibrator: Calibrator, index: i32) -> vec2<f32> {
    let len = i32(calibrator.buffer_len);
    let wrapped = ((index % len) + len) % len;
    return buffers[calibrator.buffer_offset + u32(wrapped)];
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let m = id.x;
    let receiver_idx = id.y;

    if (m >= params.num_bins) {
        return;
    }

    // normalized position within the segment, in [-1, 1]
    let segment_idx = m / params.segment_len;
    let local = f32(m - segment_idx * params.segment_len);
    let x = (local - f32(params.segment_len - 1u) / 2.0) / params.segment_half_len;

    var sum = vec2<f32>(0.0, 0.0);
    for (var c = 0u; c < params.num_calibrators; c++) {
        let calibrator = calibrators[c];
        let sample = common_samples[c * params.num_bins + m];
        let segment = segments[(receiver_idx * params.num_calibrators + c) * params.num_segments + segment_idx];

        let d = segment.delay_s;
        let delay = d.x + x * (d.y + x * (d.z + x * d.w));

        // the large parts of the buffer position are kept as integers, only small remainders are added in f32
        let frac = sample.buffer_frac - segment.buffer_frac - calibrator.oversampled_rate_hz * delay;
        let frac_floor = floor(frac);
        let u = frac - frac_floor;
        let index = i32(sample.buffer_index) - i32(segment.buffer_index) + i32(frac_floor);

        // 4-point Lagrange interpolation
        let w0 = -u * (u - 1.0) * (u - 2.0) / 6.0;
        let w1 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0;
        let w2 = -(u + 1.0) * u * (u - 2.0) / 2.0;
        let w3 = (u + 1.0) * u * (u - 1.0) / 6.0;
        let value = w0 * buffer_sample(calibrator, index - 1)
            + w1 * buffer_sample(calibrator, index)
            + w2 * buffer_sample(calibrator, index + 1)
            + w3 * buffer_sample(calibrator, index + 2);

        let phase = TAU * fract(sample.carrier - segment.carrier - calibrator.frequency_hz * delay);
        sum += complex_mul(value, vec2<f32>(cos(phase), sin(phase))) * segment.amplitude;
    }

    signal[receiver_idx * params.num_bins + m] = sum * taper(m);
}
