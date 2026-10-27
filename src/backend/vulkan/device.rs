use crate::backend::{
    renderer::ContextId,
    vulkan::{PhysicalDevice, format::FormatEntry, image::VulkanImage},
};
use ash::{
    Device as VkDevice, ext, khr,
    vk::{
        self, DeviceCreateInfo, DeviceQueueCreateInfo, PhysicalDeviceExternalMemoryHostPropertiesEXT,
        PhysicalDeviceFeatures2, PhysicalDeviceMemoryProperties, PhysicalDeviceProperties2, Queue,
        QueueFlags,
    },
};
use drm::node::DrmNode;
use std::{
    ffi::CStr,
    fmt,
    sync::{Arc, Weak},
};

#[derive(Debug, Clone)]
/// A Vulkan logical device, with its enabled extensions and queue.
///
/// Clones share the same device.
pub struct Device(Arc<InnerDevice>);
#[derive(Debug, Clone)]
/// A weak reference to a [`Device`], that does not keep it alive.
pub struct WeakDevice(Weak<InnerDevice>);

/// Picks a queue family that supports `required` and, if possible, none of `avoid`.
///
/// Families that have none of the `avoid` flags are preferred. If there is no such
/// family, the first family that supports `required` is returned. Graphics and compute
/// families count as transfer-capable even when they do not advertise
/// [`QueueFlags::TRANSFER`], as the Vulkan specification allows.
///
/// Passing `required = COMPUTE, avoid = GRAPHICS` selects a dedicated (async) compute
/// family where one exists, so compute work does not share a hardware queue with
/// graphics work on drivers such as RADV.
pub fn select_queue_family(
    families: &[vk::QueueFamilyProperties],
    required: QueueFlags,
    avoid: QueueFlags,
) -> Option<u32> {
    let supports = |properties: &vk::QueueFamilyProperties| {
        let flags = if properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            properties.queue_flags | QueueFlags::TRANSFER
        } else {
            properties.queue_flags
        };
        properties.queue_count > 0 && flags.contains(required)
    };
    families
        .iter()
        .position(|properties| supports(properties) && !properties.queue_flags.intersects(avoid))
        .or_else(|| families.iter().position(supports))
        .map(|idx| idx as u32)
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur when creating a [`Device`].
pub enum DeviceError {
    #[error("No matching queue")]
    /// No usable queue was found.
    NoUsableQueue,
    /// Vulkan API error.
    #[error(transparent)]
    Vk(#[from] vk::Result),
}

impl Device {
    /// Creates a device on `phd` with `extensions` and `required_features` enabled, using the queue family `queue_index`.
    pub fn new(
        phd: &PhysicalDevice,
        extensions: &[&CStr],
        required_features: &mut PhysicalDeviceFeatures2<'_>,
        queue_index: u32,
    ) -> Result<Self, DeviceError> {
        let extension_pointers = extensions.iter().copied().map(CStr::as_ptr).collect::<Vec<_>>();

        if queue_index as usize >= phd.queue_families().len() {
            return Err(DeviceError::NoUsableQueue);
        }

        let mem_properties = unsafe {
            phd.instance()
                .handle()
                .get_physical_device_memory_properties(phd.handle())
        };

        let queue_info = DeviceQueueCreateInfo::default()
            .queue_family_index(queue_index)
            .queue_priorities(&[0.0]);

        let queue_create_infos: &[DeviceQueueCreateInfo<'_>] = &[queue_info];

        let device_info = DeviceCreateInfo::default()
            .queue_create_infos(queue_create_infos)
            .enabled_extension_names(&extension_pointers)
            .push_next(required_features);

        let device = unsafe {
            phd.instance()
                .handle()
                .create_device(phd.handle(), &device_info, None)
                .map_err(DeviceError::Vk)?
        };

        let queue = unsafe { device.get_device_queue(queue_index, 0) };

        let khr_external_semaphore_fd = if extensions
            .iter()
            .any(|ext| ext == &khr::external_semaphore_fd::NAME)
        {
            Some(khr::external_semaphore_fd::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let khr_external_memory_fd = if extensions.iter().any(|ext| ext == &khr::external_memory_fd::NAME) {
            Some(khr::external_memory_fd::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let ext_image_drm_format_modifier = if extensions
            .iter()
            .any(|ext| ext == &ext::image_drm_format_modifier::NAME)
        {
            Some(ext::image_drm_format_modifier::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let ext_external_memory_host = if extensions
            .iter()
            .any(|ext| ext == &ext::external_memory_host::NAME)
        {
            let mut host_props = PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut props = PhysicalDeviceProperties2::default().push_next(&mut host_props);
            unsafe {
                phd.get_properties(&mut props);
            }

            Some((
                ext::external_memory_host::Device::new(phd.instance().handle(), &device),
                host_props.min_imported_host_pointer_alignment,
            ))
        } else {
            None
        };
        let khr_external_fence_fd = if extensions.iter().any(|ext| ext == &khr::external_fence_fd::NAME) {
            Some(khr::external_fence_fd::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };

        let khr_push_descriptor = if extensions.iter().any(|ext| ext == &khr::push_descriptor::NAME) {
            Some(khr::push_descriptor::Device::new(
                phd.instance().handle(),
                &device,
            ))
        } else {
            None
        };
        let queue_family_foreign = extensions
            .iter()
            .any(|ext| ext.to_bytes() == b"VK_EXT_queue_family_foreign");

        Ok(Device(Arc::new(InnerDevice {
            vk: device,
            khr_push_descriptor,
            queue_family_foreign,
            khr_external_semaphore_fd,
            khr_external_memory_fd,
            ext_image_drm_format_modifier,
            ext_external_memory_host,
            khr_external_fence_fd,

            mem_properties,
            formats: phd.drm_formats(),
            node: phd
                .render_node()
                .ok()
                .flatten()
                .or_else(|| phd.primary_node().ok().flatten()),

            queue,
            queue_idx: queue_index,

            context: ContextId::new(),
            _instance: phd.instance().clone(),
        })))
    }

    /// Returns the underlying `ash` device.
    pub fn vk(&self) -> &VkDevice {
        &self.0.vk
    }

    /// Returns the `VK_KHR_external_semaphore_fd` functions, if the extension is enabled.
    pub fn vk_khr_external_semaphore_fd(&self) -> Option<&khr::external_semaphore_fd::Device> {
        self.0.khr_external_semaphore_fd.as_ref()
    }

    /// Returns the `VK_KHR_external_memory_fd` functions, if the extension is enabled.
    pub fn vk_khr_external_memory_fd(&self) -> Option<&khr::external_memory_fd::Device> {
        self.0.khr_external_memory_fd.as_ref()
    }

    /// Returns the `VK_EXT_image_drm_format_modifier` functions, if the extension is enabled.
    pub fn vk_ext_image_drm_format_modifier(&self) -> Option<&ext::image_drm_format_modifier::Device> {
        self.0.ext_image_drm_format_modifier.as_ref()
    }

    /// Returns the `VK_EXT_external_memory_host` functions and the minimum host pointer alignment, if the extension is enabled.
    pub fn vk_ext_external_memory_host(&self) -> Option<&(ext::external_memory_host::Device, u64)> {
        self.0.ext_external_memory_host.as_ref()
    }

    /// Returns the `VK_KHR_external_fence_fd` functions, if the extension is enabled.
    pub fn vk_khr_external_fence_fd(&self) -> Option<&khr::external_fence_fd::Device> {
        self.0.khr_external_fence_fd.as_ref()
    }

    /// Returns the `VK_KHR_push_descriptor` functions, if the extension is enabled.
    pub fn vk_khr_push_descriptor(&self) -> Option<&khr::push_descriptor::Device> {
        self.0.khr_push_descriptor.as_ref()
    }

    /// The queue family index to use when transferring ownership of an image to or from
    /// another device or process (e.g. a DMA-BUF).
    ///
    /// This is `VK_QUEUE_FAMILY_FOREIGN_EXT` if `VK_EXT_queue_family_foreign` is enabled,
    /// and `VK_QUEUE_FAMILY_EXTERNAL` otherwise.
    pub fn external_queue_family(&self) -> u32 {
        if self.0.queue_family_foreign {
            vk::QUEUE_FAMILY_FOREIGN_EXT
        } else {
            vk::QUEUE_FAMILY_EXTERNAL
        }
    }

    /// Returns the memory properties of the physical device.
    pub fn memory_properties(&self) -> &PhysicalDeviceMemoryProperties {
        &self.0.mem_properties
    }

    /// Returns the queue of the device.
    pub fn queue(&self) -> &Queue {
        &self.0.queue
    }

    /// Returns the index of the queue family that the queue belongs to.
    pub fn queue_family_idx(&self) -> u32 {
        self.0.queue_idx
    }

    /// Returns the DRM formats that the device supports.
    pub fn formats(&self) -> impl Iterator<Item = &FormatEntry> {
        self.0.formats.iter()
    }

    /// Returns the DRM node of the device, if it has one.
    pub fn node(&self) -> Option<DrmNode> {
        self.0.node.clone()
    }

    /// Creates a [`WeakDevice`] for this device.
    pub fn downgrade(&self) -> WeakDevice {
        WeakDevice(Arc::downgrade(&self.0))
    }

    /// Returns the context id of the device.
    pub fn context(&self) -> ContextId<VulkanImage> {
        self.0.context.clone()
    }
}

impl WeakDevice {
    /// Upgrades to a [`Device`], if it was not destroyed yet.
    pub fn upgrade(&self) -> Option<Device> {
        self.0.upgrade().map(Device)
    }
}

struct InnerDevice {
    vk: VkDevice,
    khr_external_semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    khr_external_memory_fd: Option<khr::external_memory_fd::Device>,
    ext_image_drm_format_modifier: Option<ext::image_drm_format_modifier::Device>,
    ext_external_memory_host: Option<(ext::external_memory_host::Device, u64)>,
    khr_external_fence_fd: Option<khr::external_fence_fd::Device>,
    khr_push_descriptor: Option<khr::push_descriptor::Device>,
    queue_family_foreign: bool,

    mem_properties: PhysicalDeviceMemoryProperties,
    formats: super::FormatList,
    #[cfg(feature = "backend_drm")]
    node: Option<DrmNode>,

    queue: Queue,
    queue_idx: u32,

    context: ContextId<VulkanImage>,

    /// Keeps the instance alive until the device is destroyed. `Drop` destroys the device
    /// before the fields are dropped.
    _instance: super::Instance,
}

impl fmt::Debug for InnerDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VkDevice").finish_non_exhaustive()
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
        unsafe {
            self.vk.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn families(flags: &[QueueFlags]) -> Vec<vk::QueueFamilyProperties> {
        flags
            .iter()
            .map(|&queue_flags| vk::QueueFamilyProperties {
                queue_flags,
                queue_count: 1,
                ..Default::default()
            })
            .collect()
    }

    const GFX: QueueFlags = QueueFlags::from_raw(
        QueueFlags::GRAPHICS.as_raw() | QueueFlags::COMPUTE.as_raw() | QueueFlags::TRANSFER.as_raw(),
    );
    const COMPUTE: QueueFlags =
        QueueFlags::from_raw(QueueFlags::COMPUTE.as_raw() | QueueFlags::TRANSFER.as_raw());
    const ENCODE: QueueFlags =
        QueueFlags::from_raw(QueueFlags::TRANSFER.as_raw() | QueueFlags::VIDEO_ENCODE_KHR.as_raw());

    fn dedicated_compute(flags: &[QueueFlags]) -> Option<u32> {
        select_queue_family(&families(flags), QueueFlags::COMPUTE, QueueFlags::GRAPHICS)
    }

    #[test]
    fn prefers_dedicated_compute_family() {
        // RADV: graphics, compute, encode.
        assert_eq!(dedicated_compute(&[GFX, COMPUTE, ENCODE]), Some(1));
        // NVIDIA: graphics, transfer, compute, encode.
        assert_eq!(
            dedicated_compute(&[GFX, QueueFlags::TRANSFER, COMPUTE, ENCODE]),
            Some(2)
        );
    }

    #[test]
    fn falls_back_to_graphics_family() {
        // ANV: graphics, encode.
        assert_eq!(dedicated_compute(&[GFX, ENCODE]), Some(0));
        assert_eq!(dedicated_compute(&[ENCODE]), None);
    }

    #[test]
    fn skips_families_without_queues() {
        let mut families = families(&[GFX, COMPUTE]);
        families[1].queue_count = 0;
        assert_eq!(
            select_queue_family(&families, QueueFlags::COMPUTE, QueueFlags::GRAPHICS),
            Some(0)
        );
    }

    #[test]
    fn graphics_and_compute_families_imply_transfer() {
        let families = families(&[QueueFlags::GRAPHICS | QueueFlags::COMPUTE]);
        assert_eq!(
            select_queue_family(&families, QueueFlags::TRANSFER, QueueFlags::empty()),
            Some(0)
        );
    }
}
