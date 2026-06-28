const LIGHT_SPEED: f32 = 299.792458; // light speed in megameters/s
const TAU: f32 = 6.283185307179586;

struct Receiver {
    x: f32,
    y: f32,
    z: f32,
    cal_dst: f32,
    cal_time_delay: f32,
    cal_dir_z: f32,
    _p0: u32,
    _p1: u32,
};

struct Source {
    dir: vec4<f32>,
    f0: f32,
    i0: f32,
    a: f32,
    _pad: f32,
};

struct Params {
    sample_freq_mhz: f32,
    downmix_freq_mhz: f32,
    bandpass_fmin_mhz: f32,
    bandpass_fmax_mhz: f32,
    receiver_noise_i: f32,
    num_bins: u32,
    cal_i0: f32,
    sources_tile_size: u32,
    source_offset: u32,
};

@group(0) @binding(0) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(1) var<storage, read> sources: array<Source>;
@group(0) @binding(2) var<storage, read> random_phase: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> spectrum: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> kahan_buffer: array<vec2<f32>>;

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

    let idx = receiver_idx * params.num_bins + bin;

    let bin_freq = fft_bin_frequency(bin);
    if (bin_freq < params.bandpass_fmin_mhz || bin_freq > params.bandpass_fmax_mhz) {
        // bin frequency is filtered out
        spectrum[idx] = vec2<f32>(0.0, 0.0);
        return;
    }
    let phys_freq = bin_freq + params.downmix_freq_mhz;

    let rx = receivers[receiver_idx];

    var sum = spectrum[idx];
    var c = kahan_buffer[idx];

    if (params.source_offset == 0) {
        let noise_phase = sum.x;
        let cal_phase = sum.y;
        spectrum[idx] = vec2<f32>(0.0, 0.0);

        // system noise contribution
        let lambda = LIGHT_SPEED / phys_freq;
        let eff_aperture = min(4.0, lambda*lambda/4.0);
        let noise_ampl = sqrt(params.receiver_noise_i / eff_aperture);
        sum = vec2<f32>(cos(noise_phase), sin(noise_phase)) * noise_ampl;

        // calibrator signal contribution
        let gain = rx.cal_dir_z * rx.cal_dir_z;
        let cal_ampl = sqrt(gain * params.cal_i0) / rx.cal_dst / sqrt(2.0 * TAU);
        let delta_phase = TAU * bin_freq * rx.cal_time_delay;
        let new_cal_phase = cal_phase + delta_phase;
        let cal_contrib = vec2<f32>(cos(new_cal_phase), sin(new_cal_phase)) * cal_ampl;

        let u = sum + cal_contrib;
        c = (u - sum) - cal_contrib;
        sum = u;
    }

    for (var s = 0u; s < params.sources_tile_size; s++) {
        let src = sources[s];

        let gain = src.dir.z * src.dir.z; // cos^2(theta) gain model
        let ampl = gain * src.i0 * pow(phys_freq / src.f0, src.a);

        let phase = random_phase[s * params.num_bins + bin]
            + TAU * phys_freq * dot(src.dir, vec4<f32>(rx.x, rx.y, rx.z, 0.0)) / LIGHT_SPEED;

        let y = vec2<f32>(cos(phase), sin(phase)) * sqrt(ampl);

        // Kahan summation, better numerical accuracy when accumulating
        let t = y - c;
        let u = sum + t;
        c = (u - sum) - t;
        sum = u;
    }

    spectrum[idx] = sum;
    kahan_buffer[idx] = c;
}
