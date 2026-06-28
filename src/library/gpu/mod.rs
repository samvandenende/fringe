#![allow(non_snake_case)] // ingore non-snake-case for units in variable names

use crate::library::normalize_and_truncate;

use super::{Array, Calibrator, Phases, Source, Vec3};
use num_complex::Complex32;
use std::num::NonZero;

const GPU_TILE_SIZE: u32 = 1024;
const WORKGROUP_SIZE_X: u32 = 256;

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
    fn new(array: &Array, calibrator: &Calibrator, frequency_resolution: usize) -> Self {
        ComputeSpectraParams {
            array_sample_frequency_MHz: (array.sample_frequency / 1e6) as f32,
            array_downmix_frequency_MHz: (array.downmix_frequency / 1e6) as f32,
            array_bandpass_fmin_MHz: (array.bandpass[0] / 1e6) as f32,
            array_bandpass_fmax_MHz: (array.bandpass[1] / 1e6) as f32,
            array_system_noise_intensity: array.system_noise_intensity as _,
            spectrum_synthesis_window_size: (array.sample_window_size * frequency_resolution) as _,
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
    readback_buf: wgpu::Buffer,
    compute_spectra_bindgroup: wgpu::BindGroup,
    compute_spectra_pipeline: wgpu::ComputePipeline,
    compute_ifft_bgl: wgpu::BindGroupLayout,
    compute_ifft_pipeline: wgpu::ComputePipeline,
    frequency_resolution: usize,
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
        let (_instance, _adapter, device, queue) = pollster::block_on(init_gpu());

        let receiver_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Receivers"),
            size: (array.antenna_positions.len() * size_of::<ReceiverGpu>()) as _,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sources_tile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sources tile"),
            size: (GPU_TILE_SIZE * std::mem::size_of::<Source>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let phases_tile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Phases tile"),
            size: (GPU_TILE_SIZE as usize * array.sample_window_size * frequency_resolution * 4)
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params"),
            size: size_of::<ComputeSpectraParams>().max(size_of::<ComputeIfftParams>()) as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_bins =
            array.antenna_positions.len() * array.sample_window_size * frequency_resolution;
        let spectra_buf1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectra 1"),
            size: (total_bins * 2 /* floats per complex */ * 4/* size of float */) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectra_buf2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectra 2"),
            size: (total_bins * 2 /* floats per complex */ * 4/* size of float */) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: (total_bins * 2 /* floats per complex */ * 4/* size of float */) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            readback_buf,
            compute_spectra_bindgroup,
            compute_spectra_pipeline,
            compute_ifft_bgl,
            compute_ifft_pipeline,
            frequency_resolution,
        }
    }

    /// Start the simulation
    ///
    /// Executes the GPU compute pipeline for spectrum synthesis and inverse FFT.
    ///
    /// This method:
    /// - Uploads receiver geometry to the GPU
    /// - Uploads system noise + calibration phases
    /// - Processes sources in tiled batches
    /// - Runs spectrum synthesis compute shader per tile
    /// - Executes iterative inverse FFT stages (ping-pong buffering)
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrator`: Optional external calibrator.
    /// - `phases`: Precomputed phase information for system noise, sources,
    ///   and calibration signals.
    pub(crate) fn start(
        &self,
        array: &Array,
        sources: &[Source],
        calibrator: Option<&Calibrator>,
        phases: &Phases,
    ) {
        let null_calibrator = Calibrator::new(Vec3::new(0.0, 0.0, 1.0), 0.0);
        let calibrator = calibrator.unwrap_or(&null_calibrator);

        let receivers = receivers(array, calibrator);
        self.queue
            .write_buffer(&self.receiver_buf, 0, bytemuck::cast_slice(&receivers));

        let system_noise_and_cal_signal_phases: Vec<Complex32> = phases
            .calibrator_signal
            .iter()
            .cycle()
            .zip(phases.system_noise.clone())
            .map(|(i, r)| Complex32::new(r, *i))
            .collect();
        self.queue.write_buffer(
            &self.spectra_buf1,
            0,
            bytemuck::cast_slice(&system_noise_and_cal_signal_phases),
        );

        let mut params = ComputeSpectraParams::new(array, calibrator, self.frequency_resolution);
        let num_spectrum_bins = params.spectrum_synthesis_window_size;
        let num_tiles = (sources.len() as u32).div_ceil(GPU_TILE_SIZE).max(1);
        for tile in 0..num_tiles {
            let source_offset = tile * GPU_TILE_SIZE;
            let tile_size = (sources.len() as u32 - source_offset).min(GPU_TILE_SIZE);
            let phase_offset = source_offset as usize * num_spectrum_bins as usize;
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
            let phase_len = tile_size as usize * num_spectrum_bins as usize;
            let phases_slice = &phases.sources[phase_offset..phase_offset + phase_len];
            self.queue
                .write_buffer(&self.phases_tile_buf, 0, bytemuck::cast_slice(phases_slice));
            // upload params
            params.update(tile_size, source_offset);
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
            // encode and submit work
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.compute_spectra_pipeline);
                pass.set_bind_group(0, &self.compute_spectra_bindgroup, &[]);

                let wg_x = num_spectrum_bins.div_ceil(WORKGROUP_SIZE_X);
                pass.dispatch_workgroups(wg_x, receivers.len() as _, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        let log_n = num_spectrum_bins.trailing_zeros();
        let mut ping_buf = &self.spectra_buf1;
        let mut pong_buf = &self.spectra_buf2;
        for stage in 0..log_n {
            let params = ComputeIfftParams::new(num_spectrum_bins, stage);
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
                    (num_spectrum_bins / 2).div_ceil(WORKGROUP_SIZE_X),
                    receivers.len() as _,
                    1,
                );
            }

            if stage == log_n - 1 {
                encoder.copy_buffer_to_buffer(
                    pong_buf,
                    0,
                    &self.readback_buf,
                    0,
                    self.readback_buf.size(),
                );
            }

            self.queue.submit(Some(encoder.finish()));
            std::mem::swap(&mut ping_buf, &mut pong_buf);
        }
    }

    /// Finish the running simulation.
    ///
    /// This method:
    /// - Maps the GPU readback buffer
    /// - Waits for GPU completion
    /// - Converts raw complex buffers into structured samples
    /// - Normalizes results by √N (FFT scaling correction)
    ///
    /// # Returns
    /// Simulated time-domain antenna data.
    pub(crate) fn finish(&self) -> Vec<Vec<Complex32>> {
        let n_receivers = self.receiver_buf.size() as usize / size_of::<ReceiverGpu>();
        let spectrum_size = self.readback_buf.size() as usize / size_of::<[f32; 2]>() / n_receivers;
        let sample_window_size = spectrum_size / self.frequency_resolution;

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("Failed to poll");

        let samples = {
            let data = slice.get_mapped_range();
            let all_samples: &[Complex32] = bytemuck::cast_slice(&data);
            all_samples
                .chunks(spectrum_size)
                .map(|samples| normalize_and_truncate(samples, sample_window_size))
                .collect()
        };

        self.readback_buf.unmap();

        samples
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
        .request_device(&wgpu::DeviceDescriptor::default())
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
