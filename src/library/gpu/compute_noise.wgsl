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
// receiver noise amplitude per bin
@group(0) @binding(1) var<storage, read> noise_amplitude: array<f32>;
// random phase per receiver and bin of the bandpass
@group(0) @binding(2) var<storage, read> noise_phase: array<f32>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<vec2<f32>>;

// Initializes the spectra with the system noise of the receivers, zero outside the bandpass.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    let receiver_idx = id.y;

    if (bin >= params.num_bins) {
        return;
    }

    let idx = receiver_idx * params.num_bins + bin;
    let j = bin - params.band_start;
    if (bin < params.band_start || j >= params.band_len) {
        spectrum[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let phase = noise_phase[receiver_idx * params.band_len + j];
    spectrum[idx] = vec2<f32>(cos(phase), sin(phase)) * noise_amplitude[bin];
}
