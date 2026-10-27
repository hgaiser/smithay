use super::{Error, VulkanRenderer, buffer::Buffer, capabilities::Capability, shaders::VulkanTexProgram};
use crate::backend::vulkan::{device::Device, image::VulkanImage};

use ash::vk::{
    AccessFlags2, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
    CommandBufferSubmitInfo, CommandBufferUsageFlags, CommandPool as VkCommandPool, DependencyInfo,
    ExportSemaphoreCreateInfo, ExternalSemaphoreHandleTypeFlags, Fence, ImageAspectFlags, ImageLayout,
    ImageMemoryBarrier2, ImageSubresourceRange, MemoryBarrier2, PipelineStageFlags2, QUEUE_FAMILY_IGNORED,
    Semaphore, SemaphoreCreateInfo, SemaphoreGetFdInfoKHR, SemaphoreSubmitInfo, SubmitInfo2,
};
use std::collections::VecDeque;
use std::os::fd::{FromRawFd, OwnedFd};

#[derive(Debug)]
pub struct CommandPool {
    device: Device,
    vk: VkCommandPool,
    pending_buffers: VecDeque<PendingBuffer>,
}

#[derive(Debug)]
struct PendingBuffer {
    seq: u64,
    cmd_buf: CommandBuffer,
    // Keep the resources used by the submission alive until it has completed.
    _images: Vec<VulkanImage>,
    _buffers: Vec<Buffer>,
    /// Semaphores that the submission waited on, destroyed once it is done.
    semaphores: Vec<Semaphore>,
    _programs: Vec<VulkanTexProgram>,
}

impl CommandPool {
    pub(super) fn from_vk(device: &Device, vk: VkCommandPool) -> Self {
        CommandPool {
            device: device.clone(),
            vk,
            pending_buffers: VecDeque::new(),
        }
    }

    pub fn create_and_begin_buffer(&self) -> Result<CommandBuffer, Error> {
        let device = &self.device;

        let allocate_info = CommandBufferAllocateInfo::default()
            .command_pool(self.vk.clone())
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buf = unsafe {
            device
                .vk()
                .allocate_command_buffers(&allocate_info)
                .map_err(Error::CommandBufferError)?[0]
        };

        let begin_info = CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .vk()
                .begin_command_buffer(buf.clone(), &begin_info)
                .map_err(Error::CommandBufferError)?;
        }

        Ok(buf)
    }

    pub fn store_pending_buffer(
        &mut self,
        buf: CommandBuffer,
        seq: u64,
        images: impl IntoIterator<Item = VulkanImage>,
        buffers: impl IntoIterator<Item = Buffer>,
        semaphores: impl IntoIterator<Item = Semaphore>,
        programs: Vec<VulkanTexProgram>,
    ) {
        self.pending_buffers.push_back(PendingBuffer {
            seq,
            cmd_buf: buf,
            _images: images.into_iter().collect(),
            _buffers: buffers.into_iter().collect(),
            semaphores: semaphores.into_iter().collect(),
            _programs: programs,
        });
    }

    pub fn clean_old_buffers(&mut self, seq: u64) {
        let idx = self.pending_buffers.iter().position(|pending| pending.seq > seq);

        // TODO: truncate_front when stable
        let done = (if let Some(idx) = idx {
            self.pending_buffers.drain(..idx)
        } else {
            self.pending_buffers.drain(..)
        })
        .collect::<Vec<PendingBuffer>>();

        {
            let device = &self.device;
            let bufs = done.iter().map(|pending| pending.cmd_buf).collect::<Vec<_>>();
            unsafe {
                if !bufs.is_empty() {
                    device.vk().free_command_buffers(self.vk.clone(), &bufs);
                }
                for semaphore in done.iter().flat_map(|pending| &pending.semaphores) {
                    device.vk().destroy_semaphore(*semaphore, None);
                }
            }
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        {
            let device = &self.device;
            let pending = self.pending_buffers.drain(..).collect::<Vec<_>>();
            let bufs = pending.iter().map(|pending| pending.cmd_buf).collect::<Vec<_>>();

            unsafe {
                let _ = device.vk().device_wait_idle();
                if !bufs.is_empty() {
                    device.vk().free_command_buffers(self.vk, &bufs);
                }
                for semaphore in pending.iter().flat_map(|pending| &pending.semaphores) {
                    device.vk().destroy_semaphore(*semaphore, None);
                }
                device.vk().destroy_command_pool(self.vk.clone(), None);
            }
        }
    }
}

/// A command buffer that is being recorded, together with the images it uses.
///
/// Images are kept in `GENERAL` layout. The first time an image is used by a recording it
/// is transitioned (or, for DMA-BUF images, acquired from the foreign queue family). Before
/// the recording is submitted, DMA-BUF images are released back to the foreign queue family.
#[derive(Debug)]
pub(super) struct Recording {
    cmd: CommandBuffer,
    images: Vec<VulkanImage>,
    buffers: Vec<Buffer>,
    programs: Vec<VulkanTexProgram>,
    /// Whether a compute dispatch wrote to an image since the last barrier.
    compute_dirty: bool,
    /// Whether the submission waits on the semaphores queued by `Renderer::wait`.
    waits_on_foreign_fences: bool,
    /// Whether the submission exports a sync_file (see `VulkanRenderer::take_sync_file`).
    exports_sync_file: bool,
}

fn image_barrier<'a>(
    image: &VulkanImage,
    old_layout: ImageLayout,
    src_queue_family: u32,
    dst_queue_family: u32,
) -> ImageMemoryBarrier2<'a> {
    ImageMemoryBarrier2::default()
        .image(*image.vk())
        .old_layout(old_layout)
        .new_layout(ImageLayout::GENERAL)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .subresource_range(
            ImageSubresourceRange::default()
                .aspect_mask(ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        )
}

impl Recording {
    pub fn cmd(&self) -> CommandBuffer {
        self.cmd
    }

    /// Keeps `program` alive until the recording has been executed.
    pub fn keep_program_alive(&mut self, program: &VulkanTexProgram) {
        if !self
            .programs
            .iter()
            .any(|used| used.pipeline() == program.pipeline())
        {
            self.programs.push(program.clone());
        }
    }

    /// Keeps `buffer` alive until the recording has been executed.
    pub fn keep_alive(&mut self, buffer: Buffer) {
        self.buffers.push(buffer);
    }

    /// Prepares `image` for the first use in this recording.
    ///
    /// Makes all earlier writes to the image (from previous submissions on this queue, or from
    /// other devices for DMA-BUFs) visible to `stage`/`access`. This does nothing if the image
    /// was already used by this recording.
    pub fn use_image(
        &mut self,
        device: &Device,
        image: &VulkanImage,
        stage: PipelineStageFlags2,
        access: AccessFlags2,
    ) {
        if self.images.iter().any(|used| used.vk() == image.vk()) {
            return;
        }

        let initialized = image.mark_initialized();
        let external = image.is_external();
        // Content of imported images was produced elsewhere and must never be discarded.
        let old_layout = if initialized || external {
            ImageLayout::GENERAL
        } else {
            ImageLayout::UNDEFINED
        };

        let barrier = if external {
            // Acquire. The source stage and access are ignored for ownership transfers.
            image_barrier(
                image,
                old_layout,
                device.external_queue_family(),
                device.queue_family_idx(),
            )
        } else {
            image_barrier(image, old_layout, QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED)
                .src_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(AccessFlags2::MEMORY_WRITE)
        }
        .dst_stage_mask(stage)
        .dst_access_mask(access);

        let barriers = [barrier];
        unsafe {
            device.vk().cmd_pipeline_barrier2(
                self.cmd,
                &DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
        self.images.push(image.clone());
    }

    /// Makes the submission skip the foreign fences queued by `Renderer::wait`, so they are
    /// left for the next one. Use this for work that does not read client buffers, like uploads
    /// and readbacks.
    pub fn skip_foreign_fences(&mut self) {
        self.waits_on_foreign_fences = false;
    }

    /// Makes the submission export a sync_file that signals when it is done, if the timeline
    /// cannot be exported. Take it with `VulkanRenderer::take_sync_file` after submitting.
    pub fn export_sync_file(&mut self) {
        self.exports_sync_file = true;
    }

    /// Records that a compute dispatch is about to write to an image.
    ///
    /// Inserts a barrier if an earlier dispatch wrote to an image, as blending reads the
    /// destination. Call this before every dispatch.
    pub fn compute_dispatch(&mut self, device: &Device) {
        if self.compute_dirty {
            let barriers = [MemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(AccessFlags2::SHADER_WRITE | AccessFlags2::SHADER_READ)
                .dst_stage_mask(PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(AccessFlags2::SHADER_READ | AccessFlags2::SHADER_WRITE)];
            unsafe {
                device
                    .vk()
                    .cmd_pipeline_barrier2(self.cmd, &DependencyInfo::default().memory_barriers(&barriers));
            }
        }
        self.compute_dirty = true;
    }
}

impl VulkanRenderer {
    /// Takes the sync_file exported by the last submission, if it exported one.
    pub(super) fn take_sync_file(&mut self) -> Option<OwnedFd> {
        self.sync_file.take()
    }

    /// Allocates and begins a command buffer.
    pub(super) fn begin_recording(&mut self) -> Result<Recording, Error> {
        // Free what the GPU is done with.
        self.cleanup()?;
        Ok(Recording {
            cmd: self.cmd_pool.create_and_begin_buffer()?,
            images: Vec::new(),
            buffers: Vec::new(),
            programs: Vec::new(),
            compute_dirty: false,
            waits_on_foreign_fences: true,
            exports_sync_file: false,
        })
    }

    /// Ends the recording and submits it, signalling the timeline.
    ///
    /// Returns the sequence number the timeline will reach when the work is done.
    pub(super) fn submit_recording(&mut self, recording: Recording) -> Result<u64, Error> {
        let Recording {
            cmd,
            images,
            buffers,
            programs,
            waits_on_foreign_fences,
            exports_sync_file,
            ..
        } = recording;

        // Release DMA-BUF images back to the foreign queue family.
        let barriers = images
            .iter()
            .filter(|image| image.is_external())
            .map(|image| {
                image_barrier(
                    image,
                    ImageLayout::GENERAL,
                    self.device.queue_family_idx(),
                    self.device.external_queue_family(),
                )
                .src_stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(AccessFlags2::MEMORY_WRITE)
            })
            .collect::<Vec<_>>();
        unsafe {
            if !barriers.is_empty() {
                self.device
                    .vk()
                    .cmd_pipeline_barrier2(cmd, &DependencyInfo::default().image_memory_barriers(&barriers));
            }
            self.device
                .vk()
                .end_command_buffer(cmd)
                .map_err(Error::CommandBufferError)?;
        }

        self.seq_no += 1;
        let seq_no = self.seq_no;
        let cmds = [CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let mut signals = vec![
            SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.vk())
                .value(seq_no)
                .stage_mask(PipelineStageFlags2::ALL_COMMANDS),
        ];
        let mut semaphores = if waits_on_foreign_fences {
            std::mem::take(&mut self.foreign_fences)
        } else {
            Vec::new()
        };
        let waits = semaphores
            .iter()
            .map(|semaphore| {
                SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        let sync_file_semaphore = if exports_sync_file
            && !self.timeline.is_exportable()
            && self.capabilities.contains(&Capability::ExportSyncFile)
        {
            let mut export_info =
                ExportSemaphoreCreateInfo::default().handle_types(ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let semaphore = unsafe {
                self.device
                    .vk()
                    .create_semaphore(&SemaphoreCreateInfo::default().push_next(&mut export_info), None)
                    .map_err(Error::SemaphoreError)?
            };
            signals.push(
                SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .stage_mask(PipelineStageFlags2::ALL_COMMANDS),
            );
            semaphores.push(semaphore);
            Some(semaphore)
        } else {
            None
        };
        unsafe {
            self.device
                .vk()
                .queue_submit2(
                    *self.device.queue(),
                    &[SubmitInfo2::default()
                        .wait_semaphore_infos(&waits)
                        .command_buffer_infos(&cmds)
                        .signal_semaphore_infos(&signals)],
                    Fence::null(),
                )
                .map_err(Error::SubmitError)?;
        }

        // A SYNC_FD export can only happen after the signal operation was submitted.
        self.sync_file = sync_file_semaphore.and_then(|semaphore| {
            let ext = self.device.vk_khr_external_semaphore_fd()?;
            let info = SemaphoreGetFdInfoKHR::default()
                .semaphore(semaphore)
                .handle_type(ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            match unsafe { ext.get_semaphore_fd(&info) } {
                Ok(fd) => Some(unsafe { OwnedFd::from_raw_fd(fd) }),
                Err(err) => {
                    tracing::warn!(?err, "Failed to export a sync_file");
                    None
                }
            }
        });

        self.cmd_pool
            .store_pending_buffer(cmd, seq_no, images, buffers, semaphores, programs);
        Ok(seq_no)
    }
}
