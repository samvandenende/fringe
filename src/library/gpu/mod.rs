#![allow(non_snake_case)] // ingore non-snake-case for units in variable names

use crate::library::normalize_and_truncate;

use super::{Array, Calibrator, Phases, Source, Vec3};
use num_complex::Complex32;
use std::num::NonZero;

const GPU_TILE_SIZE: u32 = 1024;
const WORKGROUP_SIZE_X: u32 = 256;
/// Upper bound on the GPU memory used by the large per-batch buffers
/// (two spectrum buffers, two readback buffers and the phases tile).
const GPU_MEMORY_BUDGET: u64 = 1 << 32;
/// Number of large buffers that share `GPU_MEMORY_BUDGET`.
const NUM_LARGE_BUFFERS: u64 = 5;

/// How the simulation is split up to fit within the device limits.
///
/// Receivers are independent of each other, so they are processed in batches.
/// Sources are accumulated in tiles, of which the random phases are uploaded one at a time.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BatchPlan {
    num_spectrum_bins: usize,
    receivers_per_batch: usize,
    sources_per_tile: u32,
}

impl BatchPlan {
    /// Derives the largest batch and tile sizes that fit within the device limits and memory budget.
    ///
    /// # Panics
    /// Panics if not even a single spectrum fits on the device.
    fn new(
        limits: &wgpu::Limits,
        num_receivers: usize,
        num_spectrum_bins: usize,
        memory_budget: u64,
    ) -> Self {
        let max_buffer_size = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            .min(memory_budget / NUM_LARGE_BUFFERS);
        let max_workgroups = limits.max_compute_workgroups_per_dimension as usize;

        let spectrum_size = (num_spectrum_bins * size_of::<Complex32>()) as u64;
        let phases_size = (num_spectrum_bins * size_of::<f32>()) as u64;

        assert!(
            num_spectrum_bins.div_ceil(WORKGROUP_SIZE_X as usize) <= max_workgroups,
            "sample window size * frequency resolution ({num_spectrum_bins}) exceeds what the GPU can dispatch"
        );
        assert!(
            spectrum_size <= max_buffer_size,
            "a single spectrum of {num_spectrum_bins} bins ({spectrum_size} bytes) does not fit in a GPU buffer of at most {max_buffer_size} bytes; reduce the sample window size or frequency resolution"
        );

        let receivers_per_batch = ((max_buffer_size / spectrum_size) as usize)
            .min(max_workgroups)
            .min(num_receivers);
        let sources_per_tile = (max_buffer_size / phases_size).min(GPU_TILE_SIZE as u64) as u32;

        BatchPlan {
            num_spectrum_bins,
            receivers_per_batch,
            sources_per_tile,
        }
    }

    fn spectra_buffer_size(&self) -> u64 {
        (self.receivers_per_batch * self.num_spectrum_bins * size_of::<Complex32>()) as u64
    }

    fn phases_tile_buffer_size(&self) -> u64 {
        (self.sources_per_tile as usize * self.num_spectrum_bins * size_of::<f32>()) as u64
    }
}

/// A batch of receivers that has been submitted, but not yet read back.
struct PendingBatch {
    readback_idx: usize,
    num_receivers: usize,
    submission: wgpu::SubmissionIndex,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Gpu {
    x: f32,
    y: f32,
    z: f32,
    _p: u32,
}

impl From<Vec3> for Vec3Gpu {
    fn from(value: Vec3) -> Self {
        Vec3Gpu {
            x: value.x as _,
            y: value.y as _,
            z: value.z as _,
            _p: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SourceGpu {
    direction: Vec3Gpu,
    reference_frequency_MHz: f32,
    reference_intensity: f32,
    spectral_index: f32,
    _p: u32,
}

impl From<Source> for SourceGpu {
    fn from(value: Source) -> Self {
        SourceGpu {
            direction: value.direction.into(),
            reference_frequency_MHz: (value.reference_frequency / 1e6) as f32,
            reference_intensity: value.reference_intensity as _,
            spectral_index: value.spectral_index as _,
            _p: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReceiverGpu {
    x: f32,
    y: f32,
    z: f32,
    calibrator_distance: f32,
    calibrator_time_delay_μs: f32,
    calibrator_direction_z: f32,
    _p: [u32; 2],
}

fn receivers(array: &Array, calibrator: &Calibrator) -> Vec<ReceiverGpu> {
    const LIGHT_SPEED_MMS: f64 = 299.7924580; // in megameters per second

    array
        .antenna_positions
        .iter()
        .map(|p| {
            let p_diff = *p - calibrator.position;
            let calibrator_distance = p_diff.norm();
            let calibrator_time_delay_μs = calibrator_distance / LIGHT_SPEED_MMS;
            let calibrator_direction_z = -p_diff.z / calibrator_distance;
            ReceiverGpu {
                x: p.x as f32,
                y: p.y as f32,
                z: p.z as f32,
                calibrator_distance: calibrator_distance as f32,
                calibrator_time_delay_μs: calibrator_time_delay_μs as f32,
                calibrator_direction_z: calibrator_direction_z as f32,
                _p: [0; _],
            }
        })
        .collect()
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeSpectraParams {
    array_sample_frequency_MHz: f32,
    array_downmix_frequency_MHz: f32,
    array_bandpass_fmin_MHz: f32,
    array_bandpass_fmax_MHz: f32,
    array_system_noise_intensity: f32,
    spectrum_synthesis_window_size: u32,
    calibrator_intensity: f32,
    sources_tile_size: u32,
    source_offset: u32,
    _p: [u32; 3], // padding for 16-byte allignment
}

impl ComputeSpectraParams {
    fn new(array: &Array, calibrator: &Calibrator, num_spectrum_bins: usize) -> Self {
        ComputeSpectraParams {
            array_sample_frequency_MHz: (array.sample_frequency / 1e6) as f32,
            array_downmix_frequency_MHz: (array.downmix_frequency / 1e6) as f32,
            array_bandpass_fmin_MHz: (array.bandpass[0] / 1e6) as f32,
            array_bandpass_fmax_MHz: (array.bandpass[1] / 1e6) as f32,
            array_system_noise_intensity: array.system_noise_intensity as _,
            spectrum_synthesis_window_size: num_spectrum_bins as _,
            calibrator_intensity: calibrator.intensity as _,
            sources_tile_size: 0,
            source_offset: 0,
            _p: [0; _],
        }
    }

    fn update(&mut self, sources_tile_size: u32, source_offset: u32) {
        self.sources_tile_size = sources_tile_size;
        self.source_offset = source_offset;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeIfftParams {
    n_s: u32,
    stage: u32,
    _p: [u32; 2],
}

impl ComputeIfftParams {
    pub fn new(n_s: u32, stage: u32) -> Self {
        ComputeIfftParams {
            n_s,
            stage,
            _p: [0; _],
        }
    }
}

/// GPU-accelerated runtime for parallel generation of simulated antenna data.
pub(crate) struct Runtime {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    receiver_buf: wgpu::Buffer,
    sources_tile_buf: wgpu::Buffer,
    phases_tile_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    spectra_buf1: wgpu::Buffer,
    spectra_buf2: wgpu::Buffer,
    // double buffered, so a batch can be read back while the next one is computed
    readback_bufs: [wgpu::Buffer; 2],
    compute_spectra_bindgroup: wgpu::BindGroup,
    compute_spectra_pipeline: wgpu::ComputePipeline,
    compute_ifft_bgl: wgpu::BindGroupLayout,
    compute_ifft_pipeline: wgpu::ComputePipeline,
    plan: BatchPlan,
    sample_window_size: usize,
    pending: Option<PendingBatch>,
    samples: Vec<Vec<Complex32>>,
}

impl Runtime {
    /// Creates a new runtime instance.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration used to size internal resources.
    /// - `frequency_resolution` - FFT oversampling factor.
    ///
    /// # Returns
    /// A newly initialized `Runtime`.
    pub(crate) fn new(array: &Array, frequency_resolution: usize) -> Self {
        Self::with_memory_budget(array, frequency_resolution, GPU_MEMORY_BUDGET)
    }

    /// Creates a new runtime instance that uses at most roughly `memory_budget` bytes of GPU memory.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration used to size internal resources.
    /// - `frequency_resolution` - FFT oversampling factor.
    /// - `memory_budget` - Upper bound on the size of the large GPU buffers, in bytes.
    ///
    /// # Returns
    /// A newly initialized `Runtime`.
    pub(crate) fn with_memory_budget(
        array: &Array,
        frequency_resolution: usize,
        memory_budget: u64,
    ) -> Self {
        let (_instance, _adapter, device, queue) = pollster::block_on(init_gpu());

        let plan = BatchPlan::new(
            &device.limits(),
            array.antenna_positions.len(),
            array.sample_window_size * frequency_resolution,
            memory_budget,
        );

        let receiver_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Receivers"),
            size: (plan.receivers_per_batch * size_of::<ReceiverGpu>()) as _,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_tile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sources tile"),
            size: (plan.sources_per_tile as usize * size_of::<SourceGpu>()) as _,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let phases_tile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phases tile"),
            size: plan.phases_tile_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params"),
            size: size_of::<ComputeSpectraParams>().max(size_of::<ComputeIfftParams>()) as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spectra_buf1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectra 1"),
            size: plan.spectra_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectra_buf2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectra 2"),
            size: plan.spectra_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_bufs = [0, 1].map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback"),
                size: plan.spectra_buffer_size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let compute_spectra_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BGL"),
                entries: &[
                    // receivers
                    storage_entry(0),
                    // sources
                    storage_entry(1),
                    // random phase
                    storage_entry(2),
                    // params
                    uniform_entry(3),
                    // output spectrum
                    storage_rw_entry(4),
                    // kahan summation buffer
                    storage_rw_entry(5),
                ],
            });

        let compute_spectra_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &compute_spectra_bgl,
            entries: &[
                receiver_buf.as_entire_binding(),
                sources_tile_buf.as_entire_binding(),
                phases_tile_buf.as_entire_binding(),
                params_buf.as_entire_binding(),
                spectra_buf1.as_entire_binding(),
                spectra_buf2.as_entire_binding(),
            ]
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect::<Vec<_>>(),
        });

        let compute_spectra_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_spectra.wgsl"));

        let compute_spectra_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Spectrum Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&compute_spectra_bgl)],
                        immediate_size: 0,
                    }),
                ),
                module: &compute_spectra_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let compute_ifft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BGL"),
            entries: &[
                // params
                uniform_entry(0),
                // spectrum / samples ping pong buffer
                storage_entry(1),
                // spectrum / samples ping pong buffer
                storage_rw_entry(2),
            ],
        });

        let compute_ifft_shader =
            device.create_shader_module(wgpu::include_wgsl!("compute_ifft.wgsl"));

        let compute_ifft_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("IFFT Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&compute_ifft_bgl)],
                        immediate_size: 0,
                    }),
                ),
                module: &compute_ifft_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            _instance,
            _adapter,
            device,
            queue,
            receiver_buf,
            sources_tile_buf,
            phases_tile_buf,
            params_buf,
            spectra_buf1,
            spectra_buf2,
            readback_bufs,
            compute_spectra_bindgroup,
            compute_spectra_pipeline,
            compute_ifft_bgl,
            compute_ifft_pipeline,
            plan,
            sample_window_size: array.sample_window_size,
            pending: None,
            samples: Vec::new(),
        }
    }

    /// Start the simulation
    ///
    /// Executes the GPU compute pipeline for spectrum synthesis and inverse FFT.
    ///
    /// Receivers are processed in batches that fit within the device limits.
    /// For each batch, this method:
    /// - Uploads receiver geometry to the GPU
    /// - Uploads system noise + calibration phases
    /// - Processes sources in tiled batches
    /// - Runs spectrum synthesis compute shader per tile
    /// - Executes iterative inverse FFT stages (ping-pong buffering)
    /// - Reads back the previous batch while the current one is being computed
    ///
    /// The last batch is left running and is read back by `Runtime::finish`.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrator`: Optional external calibrator.
    /// - `phases`: Precomputed phase information for system noise, sources,
    ///   and calibration signals.
    pub(crate) fn start(
        &mut self,
        array: &Array,
        sources: &[Source],
        calibrator: Option<&Calibrator>,
        phases: &Phases,
    ) {
        let null_calibrator = Calibrator::new(Vec3::new(0.0, 0.0, 1.0), 0.0);
        let calibrator = calibrator.unwrap_or(&null_calibrator);

        let receivers = receivers(array, calibrator);
        let num_spectrum_bins = self.plan.num_spectrum_bins;

        // discard the results of a previous run that was never finished
        if let Some(stale) = self.pending.take() {
            self.read_back(stale);
        }
        self.samples = Vec::with_capacity(receivers.len());
        let mut params = ComputeSpectraParams::new(array, calibrator, num_spectrum_bins);

        for (batch_idx, receivers_batch) in
            receivers.chunks(self.plan.receivers_per_batch).enumerate()
        {
            let receiver_offset = batch_idx * self.plan.receivers_per_batch;
            let readback_idx = batch_idx % 2;
            let submission = self.submit_batch(
                receivers_batch,
                receiver_offset,
                sources,
                phases,
                &mut params,
                readback_idx,
            );

            // read back the previous batch while the GPU works on this one
            if let Some(previous) = self.pending.take() {
                self.read_back(previous);
            }
            self.pending = Some(PendingBatch {
                readback_idx,
                num_receivers: receivers_batch.len(),
                submission,
            });
        }
    }

    /// Encodes and submits the spectrum synthesis and inverse FFT of one batch of receivers.
    ///
    /// # Returns
    /// The index of the submission that copies the batch's samples to the readback buffer.
    fn submit_batch(
        &self,
        receivers: &[ReceiverGpu],
        receiver_offset: usize,
        sources: &[Source],
        phases: &Phases,
        params: &mut ComputeSpectraParams,
        readback_idx: usize,
    ) -> wgpu::SubmissionIndex {
        let num_spectrum_bins = self.plan.num_spectrum_bins;
        let tile_size_max = self.plan.sources_per_tile;

        self.queue
            .write_buffer(&self.receiver_buf, 0, bytemuck::cast_slice(receivers));

        let bins_range = receiver_offset * num_spectrum_bins
            ..(receiver_offset + receivers.len()) * num_spectrum_bins;
        let system_noise_and_cal_signal_phases: Vec<Complex32> = phases
            .calibrator_signal
            .iter()
            .cycle()
            .zip(&phases.system_noise[bins_range])
            .map(|(i, r)| Complex32::new(*r, *i))
            .collect();
        self.queue.write_buffer(
            &self.spectra_buf1,
            0,
            bytemuck::cast_slice(&system_noise_and_cal_signal_phases),
        );

        let num_tiles = (sources.len() as u32).div_ceil(tile_size_max).max(1);
        for tile in 0..num_tiles {
            let source_offset = tile * tile_size_max;
            let tile_size = (sources.len() as u32 - source_offset).min(tile_size_max);
            let phase_offset = source_offset as usize * num_spectrum_bins;
            // upload source tile
            let sources_slice = sources
                [source_offset as usize..(source_offset + tile_size) as usize]
                .iter()
                .map(|source| source.clone().into())
                .collect::<Vec<SourceGpu>>();
            self.queue.write_buffer(
                &self.sources_tile_buf,
                0,
                bytemuck::cast_slice(&sources_slice),
            );
            // upload source phase tile
            let phase_len = tile_size as usize * num_spectrum_bins;
            let phases_slice = &phases.sources[phase_offset..phase_offset + phase_len];
            self.queue
                .write_buffer(&self.phases_tile_buf, 0, bytemuck::cast_slice(phases_slice));
            // upload params
            params.update(tile_size, source_offset);
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
            // encode and submit work
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.compute_spectra_pipeline);
                pass.set_bind_group(0, &self.compute_spectra_bindgroup, &[]);

                let wg_x = (num_spectrum_bins as u32).div_ceil(WORKGROUP_SIZE_X);
                pass.dispatch_workgroups(wg_x, receivers.len() as _, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        let log_n = num_spectrum_bins.trailing_zeros();
        let readback_size = (receivers.len() * num_spectrum_bins * size_of::<Complex32>()) as u64;
        let mut ping_buf = &self.spectra_buf1;
        let mut pong_buf = &self.spectra_buf2;
        let mut submission = None;
        for stage in 0..log_n {
            let params = ComputeIfftParams::new(num_spectrum_bins as u32, stage);
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let ifft_stage_bindgroup = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IFFT stage bind group"),
                layout: &self.compute_ifft_bgl,
                entries: &[
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params_buf,
                        offset: 0,
                        size: NonZero::new(size_of::<ComputeIfftParams>() as _),
                    }),
                    ping_buf.as_entire_binding(),
                    pong_buf.as_entire_binding(),
                ]
                .iter()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.clone(),
                })
                .collect::<Vec<_>>(),
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.compute_ifft_pipeline);
                pass.set_bind_group(0, &ifft_stage_bindgroup, &[]);
                pass.dispatch_workgroups(
                    (num_spectrum_bins as u32 / 2).div_ceil(WORKGROUP_SIZE_X),
                    receivers.len() as _,
                    1,
                );
            }

            if stage == log_n - 1 {
                encoder.copy_buffer_to_buffer(
                    pong_buf,
                    0,
                    &self.readback_bufs[readback_idx],
                    0,
                    readback_size,
                );
            }

            submission = Some(self.queue.submit(Some(encoder.finish())));
            std::mem::swap(&mut ping_buf, &mut pong_buf);
        }

        submission.expect("spectrum must have at least 2 bins")
    }

    /// Waits for a submitted batch to complete and appends its samples to `self.samples`.
    ///
    /// Converts raw complex buffers into structured samples and
    /// normalizes results by √N (FFT scaling correction).
    fn read_back(&mut self, batch: PendingBatch) {
        let num_spectrum_bins = self.plan.num_spectrum_bins;
        let readback_buf = &self.readback_bufs[batch.readback_idx];
        let readback_size =
            (batch.num_receivers * num_spectrum_bins * size_of::<Complex32>()) as u64;

        let slice = readback_buf.slice(..readback_size);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("Failed to map readback buffer")
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(batch.submission),
                timeout: None,
            })
            .expect("Failed to poll");

        {
            let data = slice.get_mapped_range();
            let batch_samples: &[Complex32] = bytemuck::cast_slice(&data);
            self.samples.extend(
                batch_samples
                    .chunks(num_spectrum_bins)
                    .map(|samples| normalize_and_truncate(samples, self.sample_window_size)),
            );
        }

        readback_buf.unmap();
    }

    /// Finish the running simulation.
    ///
    /// Waits for the last batch to complete and reads it back.
    ///
    /// # Returns
    /// Simulated time-domain antenna data.
    pub(crate) fn finish(&mut self) -> Vec<Vec<Complex32>> {
        if let Some(batch) = self.pending.take() {
            self.read_back(batch);
        }
        std::mem::take(&mut self.samples)
    }
}

async fn init_gpu() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .unwrap();

    (instance, adapter, device, queue)
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_array(num_antennas: usize, sample_window_size: usize) -> Array {
        Array {
            sample_frequency: 120e6,
            downmix_frequency: 45e6,
            bandpass: [5e6, 55e6],
            sample_window_size,
            system_noise_intensity: 1.0,
            antenna_positions: (0..num_antennas)
                .map(|i| Vec3::new(i as f64 * 3.0, (i % 7) as f64 * 5.0, 0.0))
                .collect(),
        }
    }

    fn test_sources(num_sources: usize) -> Vec<Source> {
        (0..num_sources)
            .map(|i| {
                let direction = Vec3::from_ra_dec(i as f64 * 0.1, 0.5 + i as f64 * 0.01);
                Source::new(direction, 75e6, 1.0 + i as f64 * 0.01, -0.7)
            })
            .collect()
    }

    fn simulate(
        runtime: &mut Runtime,
        array: &Array,
        sources: &[Source],
        frequency_resolution: usize,
    ) -> Vec<Vec<Complex32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let phases = Phases::new(&mut rng, array, sources.len(), frequency_resolution);
        let calibrator = Calibrator::new(Vec3::new(10.0, 20.0, 30.0), 1.0);
        runtime.start(array, sources, Some(&calibrator), &phases);
        runtime.finish()
    }

    #[test]
    fn batch_plan_respects_default_limits() {
        let limits = wgpu::Limits::default();
        let num_spectrum_bins = (1 << 14) * 4;
        let plan = BatchPlan::new(&limits, 512, num_spectrum_bins, u64::MAX);

        assert!(plan.spectra_buffer_size() <= limits.max_storage_buffer_binding_size);
        assert!(plan.phases_tile_buffer_size() <= limits.max_storage_buffer_binding_size);
        assert_eq!(plan.receivers_per_batch, 256);
        assert_eq!(plan.sources_per_tile, 512);
    }

    #[test]
    fn batch_plan_caps_dispatch_size() {
        let plan = BatchPlan::new(&wgpu::Limits::default(), 1_000_000, 16, u64::MAX);
        assert_eq!(plan.receivers_per_batch, 65535);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn batch_plan_rejects_oversized_spectrum() {
        BatchPlan::new(&wgpu::Limits::default(), 1, 1 << 20, 1 << 20);
    }

    #[test]
    fn batched_output_matches_unbatched() {
        let frequency_resolution = 4;
        let array = test_array(37, 1 << 8);
        let sources = test_sources(300);
        let num_spectrum_bins = (array.sample_window_size * frequency_resolution) as u64;

        let mut unbatched = Runtime::new(&array, frequency_resolution);
        assert_eq!(unbatched.plan.receivers_per_batch, 37);

        // room for 5 receivers per batch and 10 sources per tile
        let budget = NUM_LARGE_BUFFERS * num_spectrum_bins * 8 * 5;
        let mut batched = Runtime::with_memory_budget(&array, frequency_resolution, budget);
        assert_eq!(batched.plan.receivers_per_batch, 5);
        assert_eq!(batched.plan.sources_per_tile, 10);

        let expected = simulate(&mut unbatched, &array, &sources, frequency_resolution);
        let actual = simulate(&mut batched, &array, &sources, frequency_resolution);
        assert_eq!(actual.len(), 37);
        assert_eq!(actual, expected);

        // the runtime can be reused
        assert_eq!(
            simulate(&mut batched, &array, &sources, frequency_resolution),
            expected
        );
    }

    #[test]
    fn large_array_does_not_exceed_limits() {
        let frequency_resolution = 4;
        let array = test_array(512, 1 << 14);
        let sources = test_sources(4);
        let mut runtime = Runtime::new(&array, frequency_resolution);
        let samples = simulate(&mut runtime, &array, &sources, frequency_resolution);
        assert_eq!(samples.len(), 512);
        assert!(samples.iter().all(|s| s.len() == 1 << 14));
        assert!(samples.iter().flatten().all(|s| s.is_finite()));
    }
}
