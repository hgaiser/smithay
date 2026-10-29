//! Implementation of the rendering traits using Vulkan

use crate::{
    backend::{
        SwapBuffersError,
        allocator::{
            Format, Fourcc,
            dmabuf::{Dmabuf, WeakDmabuf},
            format::{FormatSet, get_bpp, get_opaque, has_alpha},
        },
        drm::{DrmDeviceFd, sync::DrmSyncPoint},
        renderer::{
            Bind, Blit, BlitFrame, ContextId, ExportMem, Frame, ImportDma, ImportDmaWl, ImportMem,
            ImportMemWl, Offscreen, Renderer, RendererSuper, Texture, TextureMapping, texture_matrix,
            vulkan::{
                buffer::Buffer,
                shaders::{ClearPushConstants, TEX_FLAG_OPAQUE, TexPushConstants},
            },
        },
        vulkan::{
            PhysicalDevice, UnsupportedProperty,
            device::{Device, DeviceError, select_queue_family},
            format::{get_drm_format, get_vk_format, known_formats},
            image::{Error as ImageError, ImageUsageFlags, VulkanImage},
            version::Version,
        },
    },
    reexports::drm::node::DrmNode,
    utils::{Buffer as BufferCoords, Physical, Point, Rectangle, Size, Transform},
    wayland,
};

use ash::khr;
use ash::vk::{
    self, AccessFlags2, BorderColor, BufferImageCopy2, BufferUsageFlags, CompareOp, CopyImageToBufferInfo2,
    DescriptorImageInfo, DescriptorType, Extent3D, Filter, FormatFeatureFlags, ImageAspectFlags, ImageLayout,
    ImageSubresourceLayers, MemoryPropertyFlags, Offset3D, PipelineBindPoint, PipelineStageFlags2,
    QueueFlags, Result as VkResult, SamplerAddressMode, SamplerCreateFlags, SamplerCreateInfo,
    SamplerMipmapMode, SemaphoreWaitInfo, ShaderStageFlags,
};
use drm_fourcc::DrmModifier as Modifier;
use glam::{Affine2, Vec2};
use indexmap::IndexSet;
use tracing::{trace, warn};
use wayland_server::protocol::wl_buffer;

use std::{
    collections::HashMap,
    ffi::CStr,
    os::fd::{FromRawFd, IntoRawFd, OwnedFd},
    ptr::NonNull,
    slice,
    sync::{Arc, Mutex},
};

use self::cmds::Recording;
use super::{Color32F, TextureFilter, sync::SyncPoint};

mod buffer;
mod capabilities;
mod cmds;
mod device;
mod image;
mod mem;
mod shaders;
mod sync;

pub use self::buffer::Error as BufferError;
pub use self::capabilities::*;
use self::shaders::Pipelines;
pub use self::shaders::{Error as PipelineError, VulkanTexProgram};
pub use self::sync::*;

#[derive(Debug)]
/// A renderer that draws with Vulkan compute shaders.
pub struct VulkanRenderer {
    pub(crate) phd: PhysicalDevice,
    capabilities: Vec<Capability>,
    dmabuf_tex_cache: HashMap<WeakDmabuf, VulkanImage>,
    dmabuf_fb_cache: HashMap<WeakDmabuf, VulkanImage>,

    pipelines: Pipelines,
    cmd_pool: cmds::CommandPool,
    nearest_sampler: vk::Sampler,
    linear_sampler: vk::Sampler,
    tex_up_filter: TextureFilter,
    tex_down_filter: TextureFilter,
    debug_flags: super::DebugFlags,

    seq_no: u64,
    timeline: sync::VulkanTimeline,
    /// Binary semaphores holding imported foreign fences, waited on by the next submission
    /// that reads client buffers.
    foreign_fences: Vec<vk::Semaphore>,
    /// The sync_file exported by the last submission, see `take_sync_file`.
    sync_file: Option<OwnedFd>,
    // Only read by the multi-GPU backend.
    #[allow(dead_code)]
    pub(crate) node: Option<DrmNode>,
    /// Size of the texture pipeline's push constant range.
    tex_push_constants_size: u32,

    // A bunch of the previous structs contain Weak-device references.
    // So we want to drop this last for proper cleanup and avoiding accidental
    // resource leaks.
    pub(crate) device: Device,
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            // Command buffers may still use the pipelines and the sampler.
            let _ = self.device.vk().device_wait_idle();
            self.device.vk().destroy_sampler(self.nearest_sampler, None);
            self.device.vk().destroy_sampler(self.linear_sampler, None);
            for semaphore in self.foreign_fences.drain(..) {
                self.device.vk().destroy_semaphore(semaphore, None);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur when using the [`VulkanRenderer`].
pub enum Error {
    #[error("Unsupported version")]
    /// The Vulkan version is not supported.
    UnsupportedVersion,
    #[error("Provided drm device doesn't match PhysicalDevice")]
    /// The provided DRM device does not match the physical device.
    MismatchedDrmDevice,
    #[error("Missing device feature: `{0}`")]
    /// A required device feature is missing.
    MissingFeature(&'static str),
    #[error("Missing extension: `{0:?}`")]
    /// A required extension is missing.
    MissingExtension(&'static CStr),
    #[error(transparent)]
    /// A Vulkan device error.
    DeviceError(#[from] DeviceError),
    #[error("Failed to copy from host memory to an image")]
    /// Copying between host memory and an image failed.
    ImageMapError(#[source] VkResult),
    #[error("Failed to invalidate or flush mapped memory ranges")]
    /// Invalidating or flushing the mapped memory ranges failed.
    MapUpdateError(#[source] VkResult),
    #[error(transparent)]
    /// A buffer error.
    BufferError(#[from] BufferError),
    #[error(transparent)]
    /// An image error.
    ImageError(#[from] ImageError),
    #[error(transparent)]
    /// A pipeline error.
    PipelineError(#[from] PipelineError),
    #[error("Failed to create vulkan command pool")]
    /// Creating the command pool failed.
    CommandPoolError(#[source] VkResult),
    #[error("Failed to create vulkan command buffer")]
    /// Creating or allocating a command buffer failed.
    CommandBufferError(#[source] VkResult),
    #[error("Failed to create vulkan fence")]
    /// Creating a fence failed.
    FenceError(#[source] VkResult),
    #[error("Failed to create an image sampler")]
    /// Creating an image sampler failed.
    SamplerError(#[source] VkResult),
    #[error("Failed to create vulkan semaphore")]
    /// Creating a semaphore failed.
    SemaphoreError(#[source] VkResult),
    #[error("Failed to submit command buffer")]
    /// Submitting a command buffer failed.
    SubmitError(#[source] VkResult),
    #[error("Error interfacing with the underlying drm device")]
    /// Interfacing with the underlying DRM device failed.
    DrmError(#[source] std::io::Error),
    #[error("Underlying Vulkan Device was destroyed")]
    /// The underlying Vulkan device was destroyed.
    DeadDevice,
    #[error("Failure accessing wayland buffer")]
    /// Accessing the contents of a wayland buffer failed.
    BufferAccessError(wayland::shm::BufferAccessError),
    #[error("ForeignSyncInterrupted")]
    /// Waiting for a foreign sync was interrupted.
    ForeignSyncInterrupted,
    /// Custom texture program uniforms do not fit in the push constants, or do not match the
    /// size the program was created with.
    #[error("Invalid custom uniforms size {0}")]
    InvalidUniformsSize(usize),
    /// The texture program was created by a different renderer.
    #[error("Texture program belongs to a different renderer")]
    ForeignTexProgram,
}

impl From<Error> for SwapBuffersError {
    fn from(value: Error) -> Self {
        match value {
            x @ Error::UnsupportedVersion
            | x @ Error::MismatchedDrmDevice
            | x @ Error::MissingFeature(_)
            | x @ Error::MissingExtension(_)
            | x @ Error::DeviceError(DeviceError::NoUsableQueue)
            | x @ Error::DeviceError(DeviceError::Vk(VkResult::ERROR_DEVICE_LOST))
            | x @ Error::ImageMapError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::CommandPoolError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::CommandBufferError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::FenceError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::SamplerError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::SubmitError(VkResult::ERROR_DEVICE_LOST)
            | x @ Error::DeadDevice => SwapBuffersError::ContextLost(Box::new(x)),
            x => SwapBuffersError::TemporaryFailure(Box::new(x)),
        }
    }
}

/// Which queue family a [`VulkanRenderer`] submits its work to.
///
/// All rendering is done with compute shaders, so any compute-capable family works.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QueueSelection {
    /// Prefer a compute family without graphics support (an async compute queue), and fall
    /// back to a graphics family if the device has none.
    ///
    /// On drivers that map graphics families to a single hardware queue (RADV), this keeps
    /// rendering from being serialized behind other clients' graphics work.
    #[default]
    PreferDedicatedCompute,
    /// Prefer a family that supports both graphics and compute.
    Graphics,
    /// Use the queue family with this index. It must support compute.
    Family(u32),
}

impl QueueSelection {
    fn select(self, phd: &PhysicalDevice) -> Result<u32, Error> {
        let families = phd.queue_families();
        let index = match self {
            QueueSelection::PreferDedicatedCompute => {
                select_queue_family(&families, QueueFlags::COMPUTE, QueueFlags::GRAPHICS)
            }
            QueueSelection::Graphics => families
                .iter()
                .position(|family| {
                    family.queue_count > 0
                        && family
                            .queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                })
                .map(|index| index as u32)
                .or_else(|| select_queue_family(&families, QueueFlags::COMPUTE, QueueFlags::empty())),
            QueueSelection::Family(index) => families
                .get(index as usize)
                .filter(|family| family.queue_count > 0 && family.queue_flags.contains(QueueFlags::COMPUTE))
                .map(|_| index),
        };
        index.ok_or(Error::DeviceError(DeviceError::NoUsableQueue))
    }
}

impl VulkanRenderer {
    /// Minimum device version that is required by the renderer.
    pub const MIN_DEVICE_VERSION: Version = Version::VERSION_1_3;
    /// Maximum instance version that may be used with the renderer.
    pub const MAX_INSTANCE_VERSION: Version = Version::VERSION_1_3;

    //#[instrument(err, skip(phd), fields(physical_device = phd.name()))]
    /// Creates a renderer on `phd` with the default [`QueueSelection`].
    ///
    /// See [`VulkanRenderer::with_queue_selection`].
    pub fn new(phd: &PhysicalDevice, drm: Option<DrmDeviceFd>) -> Result<VulkanRenderer, Error> {
        Self::with_queue_selection(phd, drm, QueueSelection::default())
    }

    /// Creates a renderer on `phd` that submits to the queue family picked by `queue`.
    ///
    /// `drm` must refer to the same device as `phd`. It is used to export sync points as
    /// DRM syncobjs, and thereby as sync_files.
    pub fn with_queue_selection(
        phd: &PhysicalDevice,
        drm: Option<DrmDeviceFd>,
        queue: QueueSelection,
    ) -> Result<VulkanRenderer, Error> {
        // Check matching drm descriptor, if provided
        let node = if let Some(fd) = drm.as_ref() {
            let node = DrmNode::from_file(fd).map_err(|_| Error::MismatchedDrmDevice)?;
            if !(phd.render_node().ok().flatten() == Some(node)
                || phd.primary_node().ok().flatten() == Some(node))
            {
                return Err(Error::MismatchedDrmDevice);
            }
            Some(node)
        } else {
            phd.render_node().ok().flatten()
        };

        // Check instance version
        if phd.api_version() < Self::MIN_DEVICE_VERSION
            || phd.instance().api_version() > Self::MAX_INSTANCE_VERSION
        {
            return Err(Error::UnsupportedVersion);
        }

        // Check features
        let supported_features = Features::supported_features(phd);
        if let Err(feat) = supported_features.has_required_features() {
            return Err(Error::MissingFeature(feat));
        }
        let mut required_features = Features::required_features();

        // Check optional extensions
        let mut capabilities = Vec::new();
        let dmabuf = Capability::supports_dmabuf_memory(phd);
        capabilities.extend(dmabuf);
        if dmabuf.is_some() {
            capabilities.extend(Capability::supports_queue_family_foreign(phd));
        }
        capabilities.extend(Capability::supports_host_memory(phd));
        capabilities.extend(Capability::supports_export_timeline(phd));
        capabilities.extend(Capability::supports_import_sync_file(phd));
        capabilities.extend(Capability::supports_export_sync_file(phd));

        // Get extensions
        let mut extensions = Capability::as_extensions(&capabilities);

        // Required extensions
        if !phd.has_device_extension(khr::push_descriptor::NAME) {
            return Err(Error::MissingExtension(khr::push_descriptor::NAME));
        }
        extensions.push(khr::push_descriptor::NAME);

        // Create device
        let queue_family = queue.select(phd)?;
        let device = Device::new(phd, &extensions, unsafe { required_features.vk() }, queue_family)?;

        // Create pipelines
        // Custom texture programs get the push constant space after the built-in constants.
        let tex_push_constants_size = phd.limits().max_push_constants_size.min(256);
        let pipelines = Pipelines::new(&device, tex_push_constants_size)?;

        // Create command pool
        let cmd_pool = device.create_command_pool()?;

        // Create timeline semaphore for our queue
        let timeline =
            device.create_timeline_semaphore(if capabilities.contains(&Capability::ExportTimeline) {
                drm
            } else {
                None
            })?;

        let nearest_sampler = create_sampler(device.vk(), TextureFilter::Nearest)?;
        let linear_sampler = create_sampler(device.vk(), TextureFilter::Linear)?;

        Ok(VulkanRenderer {
            phd: phd.clone(),
            device,
            capabilities,
            dmabuf_tex_cache: HashMap::new(),
            dmabuf_fb_cache: HashMap::new(),
            pipelines,
            cmd_pool,
            nearest_sampler,
            linear_sampler,
            tex_up_filter: TextureFilter::Linear,
            tex_down_filter: TextureFilter::Linear,
            debug_flags: super::DebugFlags::empty(),
            seq_no: 0,
            foreign_fences: Vec::new(),
            sync_file: None,
            node,
            tex_push_constants_size,
            timeline,
        })
    }

    /// Byte offset in the push constant block where custom texture program uniforms start.
    pub const CUSTOM_UNIFORMS_OFFSET: usize = std::mem::size_of::<TexPushConstants>();

    /// Creates a custom texture program from compute shader SPIR-V.
    ///
    /// The shader must be a copy of the renderer's `texture.glsl` with the same bindings and
    /// push constants, usually with a different `process()` function. It may declare
    /// `uniforms_size` bytes of extra uniforms at the end of the push constant block, starting
    /// at [`Self::CUSTOM_UNIFORMS_OFFSET`]. Use it with
    /// [`VulkanFrame::override_default_tex_program`].
    pub fn compile_custom_texture_shader(
        &mut self,
        spirv: &[u32],
        uniforms_size: usize,
    ) -> Result<VulkanTexProgram, Error> {
        if Self::CUSTOM_UNIFORMS_OFFSET + uniforms_size > self.tex_push_constants_size as usize {
            return Err(Error::InvalidUniformsSize(uniforms_size));
        }
        let pipeline = self.pipelines.create_tex_pipeline(&self.device, spirv)?;
        Ok(VulkanTexProgram::new(&self.device, pipeline, uniforms_size))
    }

    /// Returns the index of the queue family this renderer submits to.
    pub fn queue_family_index(&self) -> u32 {
        self.device.queue_family_idx()
    }

    /// Returns `true` if the renderer's queue family has no graphics support, so its work
    /// does not share a hardware queue with graphics work.
    pub fn is_dedicated_compute_queue(&self) -> bool {
        self.phd
            .queue_families()
            .get(self.device.queue_family_idx() as usize)
            .is_some_and(|family| !family.queue_flags.contains(QueueFlags::GRAPHICS))
    }

    /// Releases the resources of submissions that the GPU has finished, and drops cached imports of destroyed DMA-BUFs.
    pub fn cleanup(&mut self) -> Result<(), Error> {
        let val = unsafe {
            self.device
                .vk()
                .get_semaphore_counter_value(self.timeline.vk())
                .map_err(Error::SemaphoreError)?
        };
        self.cmd_pool.clean_old_buffers(val);
        self.dmabuf_tex_cache.retain(|dmabuf, _| !dmabuf.is_gone());
        self.dmabuf_fb_cache.retain(|dmabuf, _| !dmabuf.is_gone());
        Ok(())
    }
}

/// Creates a sampler that uses `filter` for magnification and minification.
///
/// Compute shaders sample at LOD 0, so the sampler cannot select the filter by scale. The
/// renderer selects the sampler per draw instead.
fn create_sampler(device: &ash::Device, filter: TextureFilter) -> Result<vk::Sampler, Error> {
    let filter = match filter {
        TextureFilter::Nearest => Filter::NEAREST,
        TextureFilter::Linear => Filter::LINEAR,
    };
    unsafe {
        device
            .create_sampler(
                &SamplerCreateInfo::default()
                    .flags(SamplerCreateFlags::empty())
                    .mag_filter(filter)
                    .min_filter(filter)
                    .mipmap_mode(SamplerMipmapMode::NEAREST)
                    .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE)
                    .mip_lod_bias(0.0)
                    .anisotropy_enable(false)
                    .max_anisotropy(0.0)
                    .compare_enable(false)
                    .compare_op(CompareOp::NEVER)
                    .min_lod(0.0)
                    .max_lod(0.0)
                    .border_color(BorderColor::FLOAT_TRANSPARENT_BLACK)
                    .unnormalized_coordinates(false),
                None,
            )
            .map_err(Error::SamplerError)
    }
}

impl RendererSuper for VulkanRenderer {
    type Error = Error;
    type TextureId = VulkanImage;
    type Framebuffer<'buffer> = VulkanFramebuffer;

    type Frame<'frame, 'buffer>
        = VulkanFrame<'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.device.context()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.tex_down_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.tex_up_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: super::DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> super::DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        // Element geometry is in the transformed output space, as in the GLES renderer.
        let output_size = dst_transform.transform_size(output_size);
        Ok(VulkanFrame {
            renderer: self,
            fb: framebuffer,
            transform: dst_transform,
            size: output_size,
            recording: None,
            last_sequence: None,
            last_sync_file: None,
            tex_program_override: None,
            _marker: std::marker::PhantomData,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        if sync.is_reached() {
            return Ok(());
        }

        // Our own work runs on the same queue. Every recording starts with barriers that
        // order it after all earlier submissions, so there is nothing to wait for.
        if let Some(sync) = sync.get::<VulkanSync>() {
            if sync.ctx().is_some_and(|id| id == self.device.context()) {
                return Ok(());
            }
        }

        // Let the GPU wait for the foreign fence.
        if self.capabilities.contains(&Capability::ImportSyncFile) {
            if let (Some(ext), Some(fd)) = (self.device.vk_khr_external_semaphore_fd(), sync.export()) {
                let semaphore = unsafe {
                    self.device
                        .vk()
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(Error::SemaphoreError)?
                };
                let raw_fd = fd.into_raw_fd();
                let info = vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                    .fd(raw_fd);
                match unsafe { ext.import_semaphore_fd(&info) } {
                    // The semaphore owns the fd now.
                    Ok(()) => {
                        self.foreign_fences.push(semaphore);
                        return Ok(());
                    }
                    Err(err) => {
                        unsafe {
                            drop(OwnedFd::from_raw_fd(raw_fd));
                            self.device.vk().destroy_semaphore(semaphore, None);
                        }
                        trace!("Failed to import sync_file, waiting on the CPU: {err}");
                    }
                }
            }
        }

        sync.wait().map_err(|_| Error::ForeignSyncInterrupted)
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        self.dmabuf_tex_cache.clear();
        self.dmabuf_fb_cache.clear();
        Ok(())
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoords>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let bpp = get_bpp(format).ok_or(ImageError::UnsupportedFormat)? / 8;
        let mut image = self.upload_from_slice(
            data,
            size,
            size.w as usize * bpp,
            format,
            ImageUsageFlags::SAMPLED,
        )?;
        image.flipped = flipped;
        Ok(image)
    }

    fn update_memory(
        &mut self,
        image: &VulkanImage,
        data: &[u8],
        region: Rectangle<i32, BufferCoords>,
    ) -> Result<(), Self::Error> {
        let format = get_drm_format(image.format()).ok_or(ImageError::UnsupportedFormat)?;
        let bpp = get_bpp(format).ok_or(ImageError::UnsupportedFormat)? / 8;
        self.update_from_slice(data, image.size().w as usize * bpp, image, &[region])?;

        Ok(())
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        let required = vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
        let phd = self.phd.clone();
        Box::new(
            [
                Fourcc::Abgr8888,
                Fourcc::Xbgr8888,
                Fourcc::Argb8888,
                Fourcc::Xrgb8888,
                Fourcc::Abgr2101010,
                Fourcc::Xbgr2101010,
                Fourcc::Argb2101010,
                Fourcc::Xrgb2101010,
                Fourcc::Abgr16161616f,
                Fourcc::Xbgr16161616f,
            ]
            .into_iter()
            .filter(move |&fourcc| {
                let Some(format) = get_vk_format(fourcc) else {
                    return false;
                };
                let mut props = vk::FormatProperties2::default();
                // SAFETY: `props` has no pNext chain.
                unsafe { phd.get_format_properties(format, &mut props) };
                props.format_properties.optimal_tiling_features.contains(required)
            }),
        )
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        surface: Option<&crate::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoords>],
    ) -> Result<Self::TextureId, Self::Error> {
        use crate::wayland::shm::{shm_format_to_fourcc, with_buffer_contents};
        struct VulkanImageWrapper(VulkanImage);

        // why not store a `VulkanImage`? because the user might do so.
        // this is guaranteed a non-public internal type, so we are good.
        type CacheMap = HashMap<ContextId<VulkanImage>, VulkanImageWrapper>;

        let mut surface_lock = surface.as_ref().map(|surface_data| {
            surface_data
                .data_map
                .get_or_insert_threadsafe(|| Arc::new(Mutex::new(CacheMap::new())))
                .lock()
                .unwrap()
        });

        with_buffer_contents(buffer, |ptr, len, data| {
            let offset = data.offset;
            let width = data.width;
            let height = data.height;
            let stride = data.stride;
            let fourcc = shm_format_to_fourcc(data.format).ok_or(ImageError::UnsupportedFormat)?;
            let bpp = get_bpp(fourcc).ok_or(ImageError::UnsupportedFormat)? / 8;
            let format = get_vk_format(fourcc).ok_or(ImageError::UnsupportedFormat)?;
            assert!((offset + (height - 1) * stride + width * bpp as i32) as usize <= len);

            let mut upload_full = false;
            let id = self.context_id();
            let image = surface_lock
                .as_ref()
                .and_then(|cache| cache.get(&id).map(|wrap| &wrap.0).cloned())
                .filter(|texture| texture.size() == (width, height).into())
                .map(Result::<_, ImageError>::Ok)
                .unwrap_or_else(|| {
                    upload_full = true;
                    let new = VulkanImage::new(
                        &self.device,
                        width as u32,
                        height as u32,
                        format,
                        has_alpha(fourcc),
                        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        false,
                    )?;
                    if let Some(cache) = surface_lock.as_mut() {
                        cache.insert(id, VulkanImageWrapper(new.clone()));
                    }
                    Ok(new)
                })?;

            let regions = if upload_full || damage.is_empty() {
                let rect = Rectangle::new(Point::<i32, BufferCoords>::new(0, 0), Size::new(width, height));

                &[rect]
            } else {
                damage
            };

            self.update_from_shm(ptr, len, stride as usize, &image, regions)?;

            Ok(image)
        })
        .map_err(Error::BufferAccessError)?
    }
}

impl VulkanRenderer {
    /// The DMA-BUF formats and modifiers the renderer can sample.
    ///
    /// Multi-planar YCbCr formats are excluded: sampling them needs a sampler YCbCr conversion,
    /// which the texture shader does not use.
    fn importable_dmabuf_formats(&self) -> impl Iterator<Item = Format> + '_ {
        self.device
            .formats()
            .filter(|entry| {
                entry
                    .modifier_properties
                    .drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
            })
            .filter(|entry| get_vk_format(entry.format.code).is_some_and(|format| !is_ycbcr(format)))
            .flat_map(|entry| with_opaque(entry.format))
    }
}

/// Returns `format` and, if it has an alpha channel, its variant without alpha.
///
/// The Vulkan format table lists only the formats with alpha. A format without alpha (for
/// example XRGB8888, which Xwayland uses) has the same memory layout and maps to the same
/// Vulkan format; the renderer ignores its alpha channel.
fn with_opaque(format: Format) -> impl Iterator<Item = Format> {
    let opaque = get_opaque(format.code).map(|code| Format {
        code,
        modifier: format.modifier,
    });
    std::iter::once(format).chain(opaque)
}

/// Whether `format` is a YCbCr format (`VK_KHR_sampler_ycbcr_conversion` or
/// `VK_EXT_ycbcr_2plane_444_formats`).
fn is_ycbcr(format: vk::Format) -> bool {
    let raw = format.as_raw();
    (vk::Format::G8B8G8R8_422_UNORM.as_raw()..=vk::Format::G16_B16_R16_3PLANE_444_UNORM.as_raw())
        .contains(&raw)
        || (vk::Format::G8_B8R8_2PLANE_444_UNORM.as_raw()..=vk::Format::G16_B16R16_2PLANE_444_UNORM.as_raw())
            .contains(&raw)
}

impl ImportDma for VulkanRenderer {
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoords>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        let key = dmabuf.weak();
        if let Some(image) = self.dmabuf_tex_cache.get(&key).cloned() {
            trace!("Re-using image {:?} for {:?}", image, dmabuf);
            return Ok(image);
        }

        let image = VulkanImage::new_from_dmabuf(
            &self.device,
            dmabuf,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC,
        )
        .map_err(Error::ImageError)?;
        self.dmabuf_tex_cache.insert(key, image.clone());
        Ok(image)
    }

    fn dmabuf_formats(&self) -> FormatSet {
        self.importable_dmabuf_formats().collect()
    }

    fn has_dmabuf_format(&self, format: Format) -> bool {
        self.importable_dmabuf_formats().any(|entry| entry == format)
    }
}
#[cfg(feature = "wayland_frontend")]
impl ImportDmaWl for VulkanRenderer {}

#[derive(Debug)]
/// A mapping of a [`VulkanImage`] in host memory, created by [`ExportMem`].
pub struct VulkanMapping {
    ptr: NonNull<u8>,
    stride: usize,
    len: usize,
    // Keeps the host-coherent staging buffer that `ptr` points into alive.
    _staging_buffer: Buffer,
    staging_seq_no: u64,
    image: VulkanImage,
}

impl TextureMapping for VulkanMapping {
    fn flipped(&self) -> bool {
        false
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl Texture for VulkanMapping {
    fn width(&self) -> u32 {
        self.image.width()
    }

    fn height(&self) -> u32 {
        self.image.height()
    }

    fn format(&self) -> Option<Fourcc> {
        get_drm_format(self.image.format())
    }
}

impl VulkanRenderer {
    fn copy_from_image(
        &mut self,
        image: &VulkanImage,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<VulkanMapping, Error> {
        // The copy does not convert: the requested format must have the image's memory layout.
        // Formats that differ only in an unused alpha channel (XRGB8888 and ARGB8888) qualify.
        if get_vk_format(format) != Some(image.format()) {
            return Err(Error::ImageError(ImageError::UnsupportedFormat));
        }
        // Always copy through a tightly packed staging buffer. A direct mapping of a linear
        // image would include row padding and ignore `region`.
        let stride = get_drm_format(image.format())
            .and_then(get_bpp)
            .ok_or(ImageError::UnsupportedFormat)?
            / 8
            * region.size.w as usize;
        let len = stride * region.size.h as usize;

        let buffer = Buffer::new(&self.device, len, None, BufferUsageFlags::TRANSFER_DST, true)?;

        let mut recording = self.begin_recording()?;
        recording.skip_foreign_fences();
        recording.use_image(
            &self.device,
            image,
            PipelineStageFlags2::ALL_TRANSFER,
            AccessFlags2::TRANSFER_READ,
        );
        let cmd = recording.cmd();
        unsafe {
            let regions = [BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    ImageSubresourceLayers::default()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .mip_level(0),
                )
                .image_offset(Offset3D::default().x(region.loc.x).y(region.loc.y))
                .image_extent(
                    Extent3D::default()
                        .width(region.size.w as u32)
                        .height(region.size.h as u32)
                        .depth(1),
                )];
            let info = CopyImageToBufferInfo2::default()
                .dst_buffer(*buffer.vk())
                .src_image(*image.vk())
                .src_image_layout(ImageLayout::GENERAL)
                .regions(&regions);
            self.device.vk().cmd_copy_image_to_buffer2(cmd, &info);
        }

        let seq_no = self.submit_recording(recording)?;

        Ok(VulkanMapping {
            ptr: buffer.mapped_ptr().unwrap().cast(),
            len,
            stride,
            _staging_buffer: buffer,
            staging_seq_no: seq_no,
            image: image.clone(),
        })
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        self.copy_from_image(&target.0, region, format)
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoords>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        self.copy_from_image(texture, region, format)
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        Ok(texture.mem_bits().contains(MemoryPropertyFlags::HOST_VISIBLE))
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        let semaphores = [self.timeline.vk()];
        let sequences = [texture_mapping.staging_seq_no];
        let info = SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&sequences);
        // TODO: Timeout
        unsafe {
            self.device
                .vk()
                .wait_semaphores(&info, u64::MAX)
                .map_err(Error::SemaphoreError)?;
        }

        // The staging buffer is host-coherent, so no invalidation is necessary.
        Ok(unsafe { slice::from_raw_parts(texture_mapping.ptr.as_ptr(), texture_mapping.len) })
    }
}

#[derive(Debug)]
/// A [`VulkanImage`] that is bound as the render target.
pub struct VulkanFramebuffer(VulkanImage);

impl Texture for VulkanFramebuffer {
    fn width(&self) -> u32 {
        Texture::width(&self.0)
    }

    fn height(&self) -> u32 {
        Texture::height(&self.0)
    }

    fn format(&self) -> Option<Fourcc> {
        Texture::format(&self.0)
    }

    fn size(&self) -> Size<i32, BufferCoords> {
        Texture::size(&self.0)
    }
}

#[derive(Debug)]
/// A frame that is being rendered by a [`VulkanRenderer`].
pub struct VulkanFrame<'frame, 'buffer> {
    renderer: &'frame mut VulkanRenderer,
    fb: &'frame mut VulkanFramebuffer,
    // Implies `'buffer: 'frame`, as `&'frame mut GlesTarget<'buffer>` does for `GlesFrame`.
    _marker: std::marker::PhantomData<&'frame mut &'buffer ()>,

    transform: Transform,
    size: Size<i32, Physical>,
    /// The command buffer all draws of this frame are recorded into. It is begun by the
    /// first draw and submitted by `finish`.
    recording: Option<Recording>,
    last_sequence: Option<u64>,
    last_sync_file: Option<Arc<OwnedFd>>,
    /// Texture program used instead of the default one, with its uniforms.
    tex_program_override: Option<(VulkanTexProgram, Vec<u8>)>,
}

impl VulkanFrame<'_, '_> {
    /// Draws textures with `program` instead of the default texture shader, until
    /// [`Self::clear_tex_program_override`] is called.
    ///
    /// `uniforms` are the program's custom uniforms, laid out as the shader declares them. Their
    /// length must match the size the program was created with.
    pub fn override_default_tex_program(
        &mut self,
        program: VulkanTexProgram,
        uniforms: Vec<u8>,
    ) -> Result<(), Error> {
        if program.context() != self.renderer.device.context() {
            return Err(Error::ForeignTexProgram);
        }
        if uniforms.len() != program.uniforms_size() {
            return Err(Error::InvalidUniformsSize(uniforms.len()));
        }
        self.tex_program_override = Some((program, uniforms));
        Ok(())
    }

    /// Draws textures with the default texture shader again.
    pub fn clear_tex_program_override(&mut self) {
        self.tex_program_override = None;
    }
}

impl Frame for VulkanFrame<'_, '_> {
    type Error = Error;
    type TextureId = VulkanImage;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        self.renderer.device.context()
    }

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        self.draw_color(Rectangle::from_size(self.size), at, color, false)
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        self.draw_color(dst, damage, color, true)
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        let tex_size = texture.size();
        if src.size.is_empty() || tex_size.is_empty() {
            return Ok(());
        }
        let damage = self.damage_to_fb(dst, damage);
        if damage.is_empty() {
            return Ok(());
        }

        // Framebuffer pixel -> output space -> element -> texture coordinates.
        let mut uv = texture_matrix(src, dst, tex_size, src_transform)
            * Affine2::from_translation(Vec2::new(-dst.loc.x as f32, -dst.loc.y as f32))
            * self.output_to_fb().inverse();
        if texture.flipped {
            uv = Affine2::from_cols_array(&[1.0, 0.0, 0.0, -1.0, 0.0, 1.0]) * uv;
        }

        // Compute shaders sample at LOD 0, so select the filter here, as GL does without mipmaps.
        let scale = src_transform.transform_size(src.size) / dst.size.to_f64();
        let filter = if scale.x > 1.0 || scale.y > 1.0 {
            self.renderer.tex_down_filter
        } else {
            self.renderer.tex_up_filter
        };
        let sampler = match filter {
            TextureFilter::Nearest => self.renderer.nearest_sampler,
            TextureFilter::Linear => self.renderer.linear_sampler,
        };

        let view = *texture
            .vk_view()
            .ok_or(Error::ImageError(ImageError::MissingOrInvalidUsage))?;

        self.ensure_recording()?;
        let recording = self.recording.as_mut().unwrap();
        let device = &self.renderer.device;
        recording.use_image(
            device,
            texture,
            PipelineStageFlags2::COMPUTE_SHADER,
            AccessFlags2::SHADER_READ,
        );
        let buf = recording.cmd();

        // SAFETY: If we were able to bind it, it has a view
        let fb_view = *self.fb.0.vk_view().unwrap();
        let fb_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(fb_view)];
        let tex_image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(view)
            .sampler(sampler)];
        let descriptor_update = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::STORAGE_IMAGE)
                .image_info(&fb_image_info),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tex_image_info),
        ];
        let layout = *self.renderer.pipelines.tex_pipeline_layout();
        let pipeline = match &self.tex_program_override {
            Some((program, _)) => {
                recording.keep_program_alive(program);
                program.pipeline()
            }
            None => *self.renderer.pipelines.tex_pipeline(),
        };

        unsafe {
            device
                .vk()
                .cmd_bind_pipeline(buf, PipelineBindPoint::COMPUTE, pipeline);
            device
                .vk_khr_push_descriptor()
                .expect("push descriptors are required")
                .cmd_push_descriptor_set(buf, PipelineBindPoint::COMPUTE, layout, 0, &descriptor_update);
        }

        for chunk in damage.chunks(4) {
            let push_constants = TexPushConstants {
                uv_x: [uv.x_axis.x, uv.y_axis.x, uv.translation.x, 0.0],
                uv_y: [uv.x_axis.y, uv.y_axis.y, uv.translation.y, 0.0],
                _reserved: 0,
                alpha,
                flags: self.renderer.debug_flags.bits() | if texture.has_alpha { 0 } else { TEX_FLAG_OPAQUE },
                damage_size: chunk.len() as u32,
                damage: std::array::from_fn(|i| chunk.get(i).copied().unwrap_or_else(Rectangle::zero)),
            };

            // Blending reads what the previous dispatch wrote.
            recording.compute_dispatch(device);
            unsafe {
                device.vk().cmd_push_constants(
                    buf,
                    layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                if let Some((_, uniforms)) = &self.tex_program_override {
                    if !uniforms.is_empty() {
                        device.vk().cmd_push_constants(
                            buf,
                            layout,
                            ShaderStageFlags::COMPUTE,
                            VulkanRenderer::CUSTOM_UNIFORMS_OFFSET as u32,
                            uniforms,
                        );
                    }
                }
                dispatch_rects(device.vk(), buf, chunk);
            }
        }

        Ok(())
    }

    fn transformation(&self) -> Transform {
        self.transform
    }

    fn output_size(&self) -> Size<i32, Physical> {
        self.size
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.wait(sync)
    }

    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        self.submit()?;
        if let Some(point) = self.last_sequence {
            Ok(VulkanSync {
                timeline: self.renderer.timeline.clone(),
                seq: point,
                sync_file: self.last_sync_file.take(),
            }
            .into())
        } else {
            Ok(SyncPoint::signaled())
        }
    }
}

impl Drop for VulkanFrame<'_, '_> {
    fn drop(&mut self) {
        // Like the GLES frame, submit what was recorded if the frame is not finished explicitly.
        if let Err(err) = self.submit() {
            warn!("Failed to submit dropped frame: {}", err);
        }
    }
}

impl VulkanFrame<'_, '_> {
    /// Begins the frame's command buffer, if the frame has not recorded anything yet.
    fn ensure_recording(&mut self) -> Result<(), Error> {
        if self.recording.is_none() {
            let mut recording = self.renderer.begin_recording()?;
            // Transition the target once, it then stays in GENERAL layout for the whole frame.
            recording.use_image(
                &self.renderer.device,
                &self.fb.0,
                PipelineStageFlags2::COMPUTE_SHADER,
                AccessFlags2::SHADER_READ | AccessFlags2::SHADER_WRITE,
            );
            self.recording = Some(recording);
        }
        Ok(())
    }

    /// Submits everything recorded so far.
    fn submit(&mut self) -> Result<(), Error> {
        if let Some(mut recording) = self.recording.take() {
            recording.export_sync_file();
            self.last_sequence = Some(self.renderer.submit_recording(recording)?);
            self.last_sync_file = self.renderer.take_sync_file().map(Arc::new);
        }
        Ok(())
    }

    /// Returns the map from output space (element geometry) to framebuffer pixel positions.
    fn output_to_fb(&self) -> Affine2 {
        let size = self.size.to_f64();
        let map = |x: f64, y: f64| {
            let point = Rectangle::<f64, Physical>::new((x, y).into(), (0.0, 0.0).into());
            let point = self.transform.transform_rect_in(point, &size).loc;
            Vec2::new(point.x as f32, point.y as f32)
        };
        let origin = map(0.0, 0.0);
        Affine2::from_cols(map(1.0, 0.0) - origin, map(0.0, 1.0) - origin, origin)
    }

    /// Converts `damage`, relative to `dst` in output space, to framebuffer pixel rectangles.
    ///
    /// The rectangles are clipped to `dst` and to the framebuffer, and empty ones are removed.
    fn damage_to_fb(
        &self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
    ) -> Vec<Rectangle<i32, Physical>> {
        let fb = Rectangle::from_size(Size::new(self.fb.width() as i32, self.fb.height() as i32));
        damage
            .iter()
            .filter_map(|rect| Rectangle::new(rect.loc + dst.loc, rect.size).intersection(dst))
            .map(|rect| self.transform.transform_rect_in(rect, &self.size))
            .filter_map(|rect| rect.intersection(fb))
            .filter(|rect| !rect.is_empty())
            .collect()
    }

    fn draw_color(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
        should_blend: bool,
    ) -> Result<(), Error> {
        let damage = self.damage_to_fb(dst, damage);
        if damage.is_empty() {
            return Ok(());
        }

        self.ensure_recording()?;
        let recording = self.recording.as_mut().unwrap();
        let device = &self.renderer.device;
        let buf = recording.cmd();

        // SAFETY: If we were able to bind it, it has a view
        let view = *self.fb.0.vk_view().unwrap();
        let image_info = [DescriptorImageInfo::default()
            .image_layout(ImageLayout::GENERAL)
            .image_view(view)];
        let descriptor_update = [vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .image_info(&image_info)];
        let layout = *self.renderer.pipelines.clear_pipeline_layout();

        unsafe {
            device.vk().cmd_bind_pipeline(
                buf,
                PipelineBindPoint::COMPUTE,
                *self.renderer.pipelines.clear_pipeline(),
            );
            device
                .vk_khr_push_descriptor()
                .expect("push descriptors are required")
                .cmd_push_descriptor_set(buf, PipelineBindPoint::COMPUTE, layout, 0, &descriptor_update);
        }

        for chunk in damage.chunks(6) {
            let push_constants = ClearPushConstants {
                color: color.components(),
                blend: should_blend as u32,
                size: chunk.len() as u32,
                rects: std::array::from_fn(|i| chunk.get(i).copied().unwrap_or_else(Rectangle::zero)),
                _padding0: [0; 2],
            };

            // Blending reads what the previous dispatch wrote.
            recording.compute_dispatch(device);
            unsafe {
                device.vk().cmd_push_constants(
                    buf,
                    layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                dispatch_rects(device.vk(), buf, chunk);
            }
        }

        Ok(())
    }
}

/// Dispatches 8x8 workgroups over the bounding box of `rects`, in framebuffer pixels.
///
/// The pipelines are created with `DISPATCH_BASE`, so `gl_GlobalInvocationID` stays the
/// framebuffer position.
unsafe fn dispatch_rects(device: &ash::Device, buf: vk::CommandBuffer, rects: &[Rectangle<i32, Physical>]) {
    let Some(bbox) = rects.iter().copied().reduce(|a, b| a.merge(b)) else {
        return;
    };
    let x0 = bbox.loc.x.max(0) as u32 / 8;
    let y0 = bbox.loc.y.max(0) as u32 / 8;
    let x1 = ((bbox.loc.x + bbox.size.w).max(0) as u32).div_ceil(8);
    let y1 = ((bbox.loc.y + bbox.size.h).max(0) as u32).div_ceil(8);
    if x1 > x0 && y1 > y0 {
        unsafe { device.cmd_dispatch_base(buf, x0, y0, 0, x1 - x0, y1 - y0, 1) };
    }
}

impl Bind<VulkanImage> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut VulkanImage) -> Result<Self::Framebuffer<'a>, Self::Error> {
        if target.vk_usage().contains(ImageUsageFlags::STORAGE) {
            Ok(VulkanFramebuffer(target.clone()))
        } else {
            Err(Error::ImageError(ImageError::MissingOrInvalidUsage))
        }
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = self.phd.get_format_modifier_properties(format)?;
                // TODO: else use get_format_properties and map to LINEAR and INVALID
                set.extend(
                    props
                        .into_iter()
                        .filter(|prop| {
                            prop.drm_format_modifier_tiling_features
                                .contains(FormatFeatureFlags::STORAGE_IMAGE)
                        })
                        .flat_map(|prop| {
                            with_opaque(Format {
                                code: *fourcc,
                                modifier: Modifier::from(prop.drm_format_modifier),
                            })
                        }),
                );
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        let key = target.weak();
        if let Some(image) = self.dmabuf_fb_cache.get(&key).cloned() {
            trace!("Re-using image {:?} for {:?}", image, target);
            return Ok(VulkanFramebuffer(image));
        }

        let image = VulkanImage::new_from_dmabuf(
            &self.device,
            target,
            ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC,
        )
        .map_err(Error::ImageError)?;
        self.dmabuf_fb_cache.insert(key, image.clone());
        Ok(VulkanFramebuffer(image))
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        // TODO: Can external_memory_dma_buf also provide formats?
        known_formats()
            .iter()
            .try_fold(IndexSet::new(), |mut set, fourcc| {
                let format = get_vk_format(*fourcc).unwrap();
                let props = self.phd.get_format_modifier_properties(format)?;
                // TODO: else use get_format_properties and map to LINEAR and INVALID
                set.extend(
                    props
                        .into_iter()
                        .filter(|prop| {
                            prop.drm_format_modifier_tiling_features
                                .contains(FormatFeatureFlags::STORAGE_IMAGE)
                        })
                        .flat_map(|prop| {
                            with_opaque(Format {
                                code: *fourcc,
                                modifier: Modifier::from(prop.drm_format_modifier),
                            })
                        }),
                );
                Result::<_, UnsupportedProperty>::Ok(set)
            })
            .ok()
            .map(FormatSet::from_formats)
    }
}

impl Offscreen<VulkanImage> for VulkanRenderer {
    fn create_buffer(&mut self, fourcc: Fourcc, size: Size<i32, BufferCoords>) -> Result<VulkanImage, Error> {
        let format = get_vk_format(fourcc).ok_or(Error::ImageError(ImageError::UnsupportedFormat))?;
        VulkanImage::new(
            &self.device,
            size.w as u32,
            size.h as u32,
            format,
            has_alpha(fourcc),
            // TRANSFER_SRC for `ExportMem`.
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            false,
        )
        .map_err(Error::from)
    }
}

impl VulkanRenderer {
    fn blit_internal(
        &mut self,
        from: &VulkanFramebuffer,
        to: &mut VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<u64, Error> {
        let mut recording = self.begin_recording()?;
        recording.use_image(
            &self.device,
            &from.0,
            PipelineStageFlags2::ALL_TRANSFER,
            AccessFlags2::TRANSFER_READ,
        );
        recording.use_image(
            &self.device,
            &to.0,
            PipelineStageFlags2::ALL_TRANSFER,
            AccessFlags2::TRANSFER_WRITE,
        );
        let buf = recording.cmd();

        unsafe {
            self.device.vk().cmd_blit_image(
                buf,
                *from.0.vk(),
                ImageLayout::GENERAL,
                *to.0.vk(),
                ImageLayout::GENERAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0),
                    )
                    .src_offsets([
                        Offset3D::default().x(src.loc.x).y(src.loc.y),
                        Offset3D::default()
                            .x(src.loc.x + src.size.w)
                            .y(src.loc.y + src.size.h),
                    ])
                    .dst_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0),
                    )
                    .dst_offsets([
                        Offset3D::default().x(dst.loc.x).y(dst.loc.y),
                        Offset3D::default()
                            .x(dst.loc.x + dst.size.w)
                            .y(dst.loc.y + dst.size.h),
                    ])],
                match filter {
                    TextureFilter::Linear => vk::Filter::LINEAR,
                    TextureFilter::Nearest => vk::Filter::NEAREST,
                },
            );
        }

        self.submit_recording(recording)
    }
}

impl Blit for VulkanRenderer {
    fn blit(
        &mut self,
        from: &VulkanFramebuffer,
        to: &mut VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.cleanup()?;

        let seq_no = self.blit_internal(from, to, src, dst, filter)?;

        if let Some(timeline) = self.timeline.export() {
            Ok(DrmSyncPoint {
                timeline,
                point: seq_no,
            }
            .into())
        } else {
            // TODO: vulkan syncpoint
            while let Err(VkResult::TIMEOUT) = unsafe {
                self.device.vk().wait_semaphores(
                    &SemaphoreWaitInfo::default()
                        .semaphores(&[self.timeline.vk()])
                        .values(&[seq_no]),
                    u64::MAX,
                )
            } {}
            Ok(SyncPoint::signaled())
        }
    }
}

impl<'frame, 'buffer> BlitFrame<VulkanFramebuffer> for VulkanFrame<'frame, 'buffer> {
    fn blit_to(
        &mut self,
        to: &mut VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        // Submit what was recorded so far, the blit is ordered after it on the queue.
        self.submit()?;
        let seq_no = self.renderer.blit_internal(self.fb, to, src, dst, filter)?;
        self.last_sequence = Some(seq_no);
        Ok(VulkanSync {
            timeline: self.renderer.timeline.clone(),
            seq: seq_no,
            sync_file: None,
        }
        .into())
    }

    fn blit_from(
        &mut self,
        from: &VulkanFramebuffer,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        self.submit()?;
        let seq_no = self.renderer.blit_internal(from, self.fb, src, dst, filter)?;
        self.last_sequence = Some(seq_no);
        Ok(VulkanSync {
            timeline: self.renderer.timeline.clone(),
            seq: seq_no,
            sync_file: None,
        }
        .into())
    }
}
