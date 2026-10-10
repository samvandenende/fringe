#![allow(non_snake_case)] // ingore non-snake-case for units in variable names

use super::{
    Array, Phases, Source, Vec3,
    calibrator::{CalibratorWindow, ConditionedBuffer, segment_half_len},
};
use num_complex::Complex32;
use std::{num::NonZero, sync::Arc};

const GPU_TILE_SIZE: u32 = 1024;
const WORKGROUP_SIZE_X: u32 = 256;
/// Upper bound on the GPU memory used by the large per-batch buffers
/// (three spectrum buffers, two readback buffers and the phases tile).
const GPU_MEMORY_BUDGET: u64 = 1 << 32;
/// Number of large buffers that share `GPU_MEMORY_BUDGET`.
const NUM_LARGE_BUFFERS: u64 = 6;

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
struct ComputeSpectraParams {
    array_sample_frequency_MHz: f32,
    array_downmix_frequency_MHz: f32,
    array_bandpass_fmin_MHz: f32,
    array_bandpass_fmax_MHz: f32,
    array_system_noise_intensity: f32,
    spectrum_synthesis_window_size: u32,
    sources_tile_size: u32,
    source_offset: u32,
}

impl ComputeSpectraParams {
    fn new(array: &Array, num_spectrum_bins: usize) -> Self {
        ComputeSpectraParams {
            array_sample_frequency_MHz: (array.sample_frequency / 1e6) as f32,
            array_downmix_frequency_MHz: (array.downmix_frequency / 1e6) as f32,
            array_bandpass_fmin_MHz: (array.bandpass[0] / 1e6) as f32,
            array_bandpass_fmax_MHz: (array.bandpass[1] / 1e6) as f32,
            array_system_noise_intensity: array.system_noise_intensity as _,
            spectrum_synthesis_window_size: num_spectrum_bins as _,
            sources_tile_size: 0,
            source_offset: 0,
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
    direction: f32,
    _p: u32,
}

impl ComputeIfftParams {
    /// Parameters of one stage of an inverse (`inverse == true`) or forward FFT.
    pub fn new(n_s: u32, stage: u32, inverse: bool) -> Self {
        ComputeIfftParams {
            n_s,
            stage,
            direction: if inverse { 1.0 } else { -1.0 },
            _p: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CalibratorGpu {
    frequency_Hz: f32,
    oversampled_rate_Hz: f32,
    buffer_len: u32,
    buffer_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CommonSampleGpu {
    buffer_index: u32,
    buffer_frac: f32,
    carrier: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SegmentGpu {
    buffer_index: u32,
    buffer_frac: f32,
    carrier: f32,
    amplitude: f32,
    delay_s: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeCalibratorsParams {
    num_bins: u32,
    num_calibrators: u32,
    num_segments: u32,
    segment_len: u32,
    segment_half_len: f32,
    offset: u32,
    _p: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateCalibratorsParams {
    array_sample_frequency_MHz: f32,
    array_bandpass_fmin_MHz: f32,
    array_bandpass_fmax_MHz: f32,
    num_bins: u32,
    scale: f32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FinalizeParams {
    num_bins: u32,
    window_size: u32,
    offset: u32,
    norm: f32,
}

/// GPU buffers holding the calibrator signals of one synthesis window.
struct CalibratorBuffers {
    calibrators: wgpu::Buffer,
    common: wgpu::Buffer,
    segments: wgpu::Buffer,
    transmit: wgpu::Buffer,
    /// Conditioned buffers currently uploaded to `transmit`.
    uploaded: Vec<Arc<ConditionedBuffer>>,
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
    spectra_buf3: wgpu::Buffer,
    // double buffered, so a batch can be read back while the next one is computed
    readback_bufs: [wgpu::Buffer; 2],
    calibrator_bufs: Option<CalibratorBuffers>,
    compute_spectra_bindgroup: wgpu::BindGroup,
    compute_spectra_pipeline: wgpu::ComputePipeline,
    compute_ifft_bgl: wgpu::BindGroupLayout,
    compute_ifft_pipeline: wgpu::ComputePipeline,
    compute_calibrators_bgl: wgpu::BindGroupLayout,
    compute_calibrators_pipeline: wgpu::ComputePipeline,
    accumulate_calibrators_bgl: wgpu::BindGroupLayout,
    accumulate_calibrators_pipeline: wgpu::ComputePipeline,
    finalize_bgl: wgpu::BindGroupLayout,
    finalize_pipeline: wgpu::ComputePipeline,
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
            size: (plan.receivers_per_batch * size_of::<Vec3Gpu>()) as _,
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

        let params_size = [
            size_of::<ComputeSpectraParams>(),
            size_of::<ComputeIfftParams>(),
            size_of::<ComputeCalibratorsParams>(),
            size_of::<AccumulateCalibratorsParams>(),
            size_of::<FinalizeParams>(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params"),
            size: params_size as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let [spectra_buf1, spectra_buf2, spectra_buf3] = ["Spectra 1", "Spectra 2", "Spectra 3"]
            .map(|label| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: plan.spectra_buffer_size(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
        let readback_bufs = [0, 1].map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback"),
                size: (plan.receivers_per_batch * array.sample_window_size * size_of::<Complex32>())
                    as _,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let compute_spectra_bgl = bind_group_layout(
            &device,
            &[
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
        );

        let compute_spectra_bindgroup = bind_group(
            &device,
            &compute_spectra_bgl,
            &[
                receiver_buf.as_entire_binding(),
                sources_tile_buf.as_entire_binding(),
                phases_tile_buf.as_entire_binding(),
                params_buf.as_entire_binding(),
                spectra_buf1.as_entire_binding(),
                spectra_buf2.as_entire_binding(),
            ],
        );

        let compute_spectra_pipeline = compute_pipeline(
            &device,
            &compute_spectra_bgl,
            wgpu::include_wgsl!("compute_spectra.wgsl"),
        );

        let compute_ifft_bgl = bind_group_layout(
            &device,
            &[
                // params
                uniform_entry(0),
                // spectrum / samples ping pong buffer
                storage_entry(1),
                // spectrum / samples ping pong buffer
                storage_rw_entry(2),
            ],
        );
        let compute_ifft_pipeline = compute_pipeline(
            &device,
            &compute_ifft_bgl,
            wgpu::include_wgsl!("compute_ifft.wgsl"),
        );

        let compute_calibrators_bgl = bind_group_layout(
            &device,
            &[
                // params
                uniform_entry(0),
                // calibrators
                storage_entry(1),
                // common samples
                storage_entry(2),
                // segments
                storage_entry(3),
                // conditioned transmit buffers
                storage_entry(4),
                // output signal
                storage_rw_entry(5),
            ],
        );
        let compute_calibrators_pipeline = compute_pipeline(
            &device,
            &compute_calibrators_bgl,
            wgpu::include_wgsl!("compute_calibrators.wgsl"),
        );

        let accumulate_calibrators_bgl = bind_group_layout(
            &device,
            &[
                // params
                uniform_entry(0),
                // calibrator signal spectrum
                storage_entry(1),
                // spectrum
                storage_rw_entry(2),
            ],
        );
        let accumulate_calibrators_pipeline = compute_pipeline(
            &device,
            &accumulate_calibrators_bgl,
            wgpu::include_wgsl!("accumulate_calibrators.wgsl"),
        );

        let finalize_bgl = bind_group_layout(
            &device,
            &[
                // params
                uniform_entry(0),
                // samples of the synthesis window
                storage_entry(1),
                // output samples
                storage_rw_entry(2),
            ],
        );
        let finalize_pipeline =
            compute_pipeline(&device, &finalize_bgl, wgpu::include_wgsl!("finalize.wgsl"));

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
            spectra_buf3,
            readback_bufs,
            calibrator_bufs: None,
            compute_spectra_bindgroup,
            compute_spectra_pipeline,
            compute_ifft_bgl,
            compute_ifft_pipeline,
            compute_calibrators_bgl,
            compute_calibrators_pipeline,
            accumulate_calibrators_bgl,
            accumulate_calibrators_pipeline,
            finalize_bgl,
            finalize_pipeline,
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
    /// - Uploads system noise phases
    /// - Processes sources in tiled batches
    /// - Runs spectrum synthesis compute shader per tile
    /// - Evaluates the calibrator signals, transforms them with a forward FFT
    ///   and adds them to the spectrum within the bandpass
    /// - Executes iterative inverse FFT stages (ping-pong buffering)
    /// - Selects and normalizes the output samples
    /// - Reads back the previous batch while the current one is being computed
    ///
    /// The last batch is left running and is read back by `Runtime::finish`.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration (positions, sampling parameters, etc.).
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrators`: Precomputed calibrator signals, if there are any calibrators.
    /// - `phases`: Precomputed phase information for system noise and sources.
    pub(crate) fn start(
        &mut self,
        array: &Array,
        sources: &[Source],
        calibrators: Option<&Arc<CalibratorWindow>>,
        phases: &Phases,
    ) {
        let receivers = array
            .antenna_positions
            .iter()
            .map(|&p| p.into())
            .collect::<Vec<Vec3Gpu>>();
        let num_spectrum_bins = self.plan.num_spectrum_bins;

        // discard the results of a previous run that was never finished
        if let Some(stale) = self.pending.take() {
            self.read_back(stale);
        }
        self.samples = Vec::with_capacity(receivers.len());
        let mut params = ComputeSpectraParams::new(array, num_spectrum_bins);
        let calibrators_bindgroup = calibrators.map(|window| self.upload_calibrators(window));

        for (batch_idx, receivers_batch) in
            receivers.chunks(self.plan.receivers_per_batch).enumerate()
        {
            let receiver_offset = batch_idx * self.plan.receivers_per_batch;
            let readback_idx = batch_idx % 2;
            let submission = self.submit_batch(
                array,
                receivers_batch,
                receiver_offset,
                sources,
                calibrators.zip(calibrators_bindgroup.as_ref()),
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

    /// Uploads the antenna independent calibrator data of a window, and the
    /// transmit buffers if they changed.
    ///
    /// # Arguments
    /// - `window`: Precomputed calibrator signals.
    ///
    /// # Returns
    /// The bind group of the calibrator compute pass.
    ///
    /// # Panics
    /// Panics if the calibrator data does not fit within the device limits.
    fn upload_calibrators(&mut self, window: &CalibratorWindow) -> wgpu::BindGroup {
        let max_binding_size = self.device.limits().max_storage_buffer_binding_size;
        let num_calibrators = window.calibrators.len();

        let mut buffer_offset = 0;
        let calibrators = window
            .calibrators
            .iter()
            .map(|signal| {
                let calibrator = CalibratorGpu {
                    frequency_Hz: signal.frequency as _,
                    oversampled_rate_Hz: signal.oversampled_rate as _,
                    buffer_len: signal.buffer.samples.len() as _,
                    buffer_offset,
                };
                buffer_offset += calibrator.buffer_len;
                calibrator
            })
            .collect::<Vec<_>>();
        let common = window
            .common
            .iter()
            .map(|c| CommonSampleGpu {
                buffer_index: c.buffer_index,
                buffer_frac: c.buffer_frac as _,
                carrier: c.carrier as _,
            })
            .collect::<Vec<_>>();
        let segments_size = (self.plan.receivers_per_batch
            * num_calibrators
            * window.num_segments
            * size_of::<SegmentGpu>()) as u64;
        let transmit_size = (buffer_offset as usize * size_of::<Complex32>()) as u64;
        let common_size = (common.len() * size_of::<CommonSampleGpu>()) as u64;
        for (size, what) in [
            (segments_size, "calibrator segments"),
            (transmit_size, "calibrator transmit buffers"),
            (common_size, "calibrator samples"),
        ] {
            assert!(
                size <= max_binding_size,
                "{what} ({size} bytes) do not fit in a GPU buffer of at most {max_binding_size} bytes"
            );
        }

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffers = self.calibrator_bufs.get_or_insert_with(|| {
            let empty = |label| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 0,
                    usage,
                    mapped_at_creation: false,
                })
            };
            CalibratorBuffers {
                calibrators: empty("Calibrators"),
                common: empty("Calibrator samples"),
                segments: empty("Calibrator segments"),
                transmit: empty("Calibrator transmit buffers"),
                uploaded: Vec::new(),
            }
        });
        let ensure_size = |buffer: &mut wgpu::Buffer, size: u64| {
            if buffer.size() < size {
                *buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage,
                    mapped_at_creation: false,
                });
                true
            } else {
                false
            }
        };

        ensure_size(
            &mut buffers.calibrators,
            (calibrators.len() * size_of::<CalibratorGpu>()) as u64,
        );
        ensure_size(&mut buffers.common, common_size);
        ensure_size(&mut buffers.segments, segments_size);
        let reallocated = ensure_size(&mut buffers.transmit, transmit_size);

        let unchanged = buffers.uploaded.len() == num_calibrators
            && buffers
                .uploaded
                .iter()
                .zip(&window.calibrators)
                .all(|(a, b)| Arc::ptr_eq(a, &b.buffer));
        if reallocated || !unchanged {
            let transmit = window
                .calibrators
                .iter()
                .flat_map(|signal| signal.buffer.samples.iter().copied())
                .collect::<Vec<_>>();
            self.queue
                .write_buffer(&buffers.transmit, 0, bytemuck::cast_slice(&transmit));
            buffers.uploaded = window
                .calibrators
                .iter()
                .map(|signal| signal.buffer.clone())
                .collect();
        }
        self.queue
            .write_buffer(&buffers.calibrators, 0, bytemuck::cast_slice(&calibrators));
        self.queue
            .write_buffer(&buffers.common, 0, bytemuck::cast_slice(&common));

        bind_group(
            &self.device,
            &self.compute_calibrators_bgl,
            &[
                params_binding::<ComputeCalibratorsParams>(&self.params_buf),
                buffers.calibrators.as_entire_binding(),
                buffers.common.as_entire_binding(),
                buffers.segments.as_entire_binding(),
                buffers.transmit.as_entire_binding(),
                self.spectra_buf2.as_entire_binding(),
            ],
        )
    }

    /// Encodes and submits the spectrum synthesis, calibrator signals, inverse FFT
    /// and output sample selection of one batch of receivers.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration.
    /// - `receivers`: Positions of the receivers in the batch.
    /// - `receiver_offset`: Index of the first receiver of the batch in the array.
    /// - `sources`: List of signal sources contributing to the simulation.
    /// - `calibrators`: Precomputed calibrator signals and the bind group of the calibrator
    ///   compute pass, if there are any calibrators.
    /// - `phases`: Precomputed phase information for system noise and sources.
    /// - `params`: Parameters of the spectrum synthesis.
    /// - `readback_idx`: Index of the readback buffer to copy the samples to.
    ///
    /// # Returns
    /// The index of the submission that copies the batch's samples to the readback buffer.
    #[allow(clippy::too_many_arguments)]
    fn submit_batch(
        &self,
        array: &Array,
        receivers: &[Vec3Gpu],
        receiver_offset: usize,
        sources: &[Source],
        calibrators: Option<(&Arc<CalibratorWindow>, &wgpu::BindGroup)>,
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
        let system_noise_phases: Vec<Complex32> = phases.system_noise[bins_range]
            .iter()
            .map(|r| Complex32::new(*r, 0.0))
            .collect();
        self.queue.write_buffer(
            &self.spectra_buf1,
            0,
            bytemuck::cast_slice(&system_noise_phases),
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
            self.submit_pass(
                &self.compute_spectra_pipeline,
                &self.compute_spectra_bindgroup,
                num_spectrum_bins,
                receivers.len(),
                None,
            );
        }

        if let Some((window, bindgroup)) = calibrators {
            self.submit_calibrators(array, window, bindgroup, receivers.len(), receiver_offset);
        }

        let spectrum = self.submit_fft(
            &self.spectra_buf1,
            &self.spectra_buf2,
            receivers.len(),
            true,
        );
        let output = if std::ptr::eq(spectrum, &self.spectra_buf1) {
            &self.spectra_buf2
        } else {
            &self.spectra_buf1
        };

        let window_size = self.sample_window_size;
        let finalize_params = FinalizeParams {
            num_bins: num_spectrum_bins as _,
            window_size: window_size as _,
            offset: ((num_spectrum_bins - window_size) / 2) as _,
            norm: (num_spectrum_bins as f32).sqrt(),
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&finalize_params));
        let finalize_bindgroup = bind_group(
            &self.device,
            &self.finalize_bgl,
            &[
                params_binding::<FinalizeParams>(&self.params_buf),
                spectrum.as_entire_binding(),
                output.as_entire_binding(),
            ],
        );
        let readback_size = (receivers.len() * window_size * size_of::<Complex32>()) as u64;
        self.submit_pass(
            &self.finalize_pipeline,
            &finalize_bindgroup,
            window_size,
            receivers.len(),
            Some((output, &self.readback_bufs[readback_idx], readback_size)),
        )
    }

    /// Evaluates the calibrator signals of a batch of receivers, transforms them to the
    /// frequency domain and adds them to the spectrum in `spectra_buf1` within the bandpass.
    ///
    /// # Arguments
    /// - `array`: Antenna array configuration.
    /// - `window`: Precomputed calibrator signals.
    /// - `bindgroup`: Bind group of the calibrator compute pass.
    /// - `num_receivers`: Number of receivers in the batch.
    /// - `receiver_offset`: Index of the first receiver of the batch in the array.
    fn submit_calibrators(
        &self,
        array: &Array,
        window: &CalibratorWindow,
        bindgroup: &wgpu::BindGroup,
        num_receivers: usize,
        receiver_offset: usize,
    ) {
        let num_spectrum_bins = self.plan.num_spectrum_bins;
        let segments_per_receiver = window.calibrators.len() * window.num_segments;
        let segments = window.segments[receiver_offset * segments_per_receiver
            ..(receiver_offset + num_receivers) * segments_per_receiver]
            .iter()
            .map(|s| SegmentGpu {
                buffer_index: s.buffer_index,
                buffer_frac: s.buffer_frac as _,
                carrier: s.carrier as _,
                amplitude: s.amplitude as _,
                delay_s: s.delay.map(|d| d as _),
            })
            .collect::<Vec<_>>();
        let buffers = self
            .calibrator_bufs
            .as_ref()
            .expect("calibrators are uploaded");
        self.queue
            .write_buffer(&buffers.segments, 0, bytemuck::cast_slice(&segments));

        let params = ComputeCalibratorsParams {
            num_bins: num_spectrum_bins as _,
            num_calibrators: window.calibrators.len() as _,
            num_segments: window.num_segments as _,
            segment_len: window.segment_len as _,
            segment_half_len: segment_half_len(window.segment_len) as _,
            offset: window.offset as _,
            _p: [0; _],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        self.submit_pass(
            &self.compute_calibrators_pipeline,
            bindgroup,
            num_spectrum_bins,
            num_receivers,
            None,
        );

        let signal_spectrum =
            self.submit_fft(&self.spectra_buf2, &self.spectra_buf3, num_receivers, false);

        let params = AccumulateCalibratorsParams {
            array_sample_frequency_MHz: (array.sample_frequency / 1e6) as f32,
            array_bandpass_fmin_MHz: (array.bandpass[0] / 1e6) as f32,
            array_bandpass_fmax_MHz: (array.bandpass[1] / 1e6) as f32,
            num_bins: num_spectrum_bins as _,
            // undo the scaling of the inverse FFT and the normalization of the output samples
            scale: (num_spectrum_bins as f32).sqrt().recip(),
            _p: [0; _],
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        let accumulate_bindgroup = bind_group(
            &self.device,
            &self.accumulate_calibrators_bgl,
            &[
                params_binding::<AccumulateCalibratorsParams>(&self.params_buf),
                signal_spectrum.as_entire_binding(),
                self.spectra_buf1.as_entire_binding(),
            ],
        );
        self.submit_pass(
            &self.accumulate_calibrators_pipeline,
            &accumulate_bindgroup,
            num_spectrum_bins,
            num_receivers,
            None,
        );
    }

    /// Submits the stages of an inverse or forward FFT.
    ///
    /// # Arguments
    /// - `ping`: Buffer holding the input, overwritten during the FFT.
    /// - `pong`: Scratch buffer.
    /// - `num_receivers`: Number of spectra to transform.
    /// - `inverse`: Whether to compute an inverse instead of a forward FFT.
    ///
    /// # Returns
    /// The buffer holding the result.
    fn submit_fft<'a>(
        &self,
        mut ping: &'a wgpu::Buffer,
        mut pong: &'a wgpu::Buffer,
        num_receivers: usize,
        inverse: bool,
    ) -> &'a wgpu::Buffer {
        let num_spectrum_bins = self.plan.num_spectrum_bins;
        let log_n = num_spectrum_bins.trailing_zeros();
        for stage in 0..log_n {
            let params = ComputeIfftParams::new(num_spectrum_bins as u32, stage, inverse);
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let stage_bindgroup = bind_group(
                &self.device,
                &self.compute_ifft_bgl,
                &[
                    params_binding::<ComputeIfftParams>(&self.params_buf),
                    ping.as_entire_binding(),
                    pong.as_entire_binding(),
                ],
            );
            self.submit_pass(
                &self.compute_ifft_pipeline,
                &stage_bindgroup,
                num_spectrum_bins / 2,
                num_receivers,
                None,
            );
            std::mem::swap(&mut ping, &mut pong);
        }
        ping
    }

    /// Submits a single compute pass, optionally followed by a buffer copy.
    ///
    /// # Arguments
    /// - `pipeline`: Compute pipeline to run.
    /// - `bindgroup`: Bind group of the pipeline.
    /// - `num_x`: Number of bins or samples per receiver.
    /// - `num_receivers`: Number of receivers.
    /// - `copy`: Optional copy `(source, destination, size)` after the pass.
    ///
    /// # Returns
    /// The index of the submission.
    fn submit_pass(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bindgroup: &wgpu::BindGroup,
        num_x: usize,
        num_receivers: usize,
        copy: Option<(&wgpu::Buffer, &wgpu::Buffer, u64)>,
    ) -> wgpu::SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindgroup, &[]);
            pass.dispatch_workgroups(
                (num_x as u32).div_ceil(WORKGROUP_SIZE_X),
                num_receivers as _,
                1,
            );
        }
        if let Some((source, destination, size)) = copy {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
        }
        self.queue.submit(Some(encoder.finish()))
    }

    /// Waits for a submitted batch to complete and appends its samples to `self.samples`.
    fn read_back(&mut self, batch: PendingBatch) {
        let readback_buf = &self.readback_bufs[batch.readback_idx];
        let readback_size =
            (batch.num_receivers * self.sample_window_size * size_of::<Complex32>()) as u64;

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
                    .chunks(self.sample_window_size)
                    .map(|samples| samples.to_vec()),
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

fn bind_group_layout(
    device: &wgpu::Device,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries,
    })
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect::<Vec<_>>(),
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: shader.label,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            }),
        ),
        module: &device.create_shader_module(shader),
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Binds the part of the shared params buffer used by parameters of type `P`.
fn params_binding<P>(params_buf: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: params_buf,
        offset: 0,
        size: NonZero::new(size_of::<P>() as _),
    })
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
    use crate::library::{Calibrator, Transmitter};
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

    /// Two moving calibrators with different transmitters, the second offset by `variant`.
    fn test_calibrators(variant: f64) -> Vec<Calibrator> {
        let noise = Transmitter::new(
            60e6,
            40e6,
            5.0,
            (0..1000)
                .map(|i| Complex32::from_polar(1.0, (i * i) as f32 * 0.21))
                .collect(),
            Some(30e6),
            0.0,
        );
        let tone = Transmitter::new(
            80e6 + variant * 1e6,
            1e6,
            1.0,
            vec![Complex32::ONE],
            None,
            0.0,
        );
        vec![
            Calibrator::new(
                Vec3::new(1e4, 2e4, 1e5),
                noise,
                Some(Vec3::new(1600.0, -300.0, 10.0)),
                Some(Vec3::new(0.0, 0.0, -25.0)),
                0.0,
            ),
            Calibrator::new(
                Vec3::new(-3e4, 5e3 * variant, 2e5),
                tone,
                Some(Vec3::new(-200.0, 1500.0, 0.0)),
                None,
                0.0,
            ),
        ]
    }

    fn calibrator_window(
        array: &Array,
        calibrators: &[Calibrator],
        frequency_resolution: usize,
    ) -> Arc<CalibratorWindow> {
        let buffers = calibrators
            .iter()
            .map(|c| Arc::new(ConditionedBuffer::new(array, &c.transmitter)))
            .collect::<Vec<_>>();
        Arc::new(CalibratorWindow::new(
            array,
            calibrators,
            &buffers,
            array.sample_window_size * frequency_resolution,
            0.75,
        ))
    }

    fn simulate(
        runtime: &mut Runtime,
        array: &Array,
        sources: &[Source],
        frequency_resolution: usize,
    ) -> Vec<Vec<Complex32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let phases = Phases::new(&mut rng, array, sources.len(), frequency_resolution);
        let calibrators = calibrator_window(array, &test_calibrators(0.0), frequency_resolution);
        runtime.start(array, sources, Some(&calibrators), &phases);
        runtime.finish()
    }

    fn relative_rms_error(actual: &[Vec<Complex32>], expected: &[Vec<Complex32>]) -> f64 {
        let (error, total) = actual.iter().flatten().zip(expected.iter().flatten()).fold(
            (0.0, 0.0),
            |(error, total), (a, e)| {
                (
                    error + (a - e).norm_sqr() as f64,
                    total + e.norm_sqr() as f64,
                )
            },
        );
        (error / total).sqrt()
    }

    #[test]
    fn matches_cpu_runtime() {
        let frequency_resolution = 4;
        let array = Array {
            system_noise_intensity: 0.0,
            ..test_array(9, 1 << 10)
        };
        let sources = test_sources(3);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let phases = Phases::new(&mut rng, &array, sources.len(), frequency_resolution);
        let mut gpu = Runtime::new(&array, frequency_resolution);
        let mut cpu = super::super::cpu::Runtime::new(frequency_resolution);

        // the second run changes the transmit buffers, which must be uploaded again
        for variant in [0.0, 1.0] {
            let calibrators =
                calibrator_window(&array, &test_calibrators(variant), frequency_resolution);
            gpu.start(&array, &sources, Some(&calibrators), &phases);
            cpu.start(&array, &sources, Some(&calibrators), &phases);
            let error = relative_rms_error(&gpu.finish(), &cpu.finish());
            println!("relative rms error {:.1} dB", 20.0 * error.log10());
            assert!(error < 1e-4);

            // calibrators only
            gpu.start(&array, &[], Some(&calibrators), &phases);
            cpu.start(&array, &[], Some(&calibrators), &phases);
            let error = relative_rms_error(&gpu.finish(), &cpu.finish());
            println!(
                "calibrators only: relative rms error {:.1} dB",
                20.0 * error.log10()
            );
            assert!(error < 1e-4);
        }
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
