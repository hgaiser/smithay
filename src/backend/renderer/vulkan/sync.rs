use std::{os::fd::OwnedFd, sync::Arc};

use crate::backend::{
    drm::sync::DrmTimeline,
    renderer::ContextId,
    vulkan::{Device, image::VulkanImage},
};
use ash::vk::{Result as VkResult, Semaphore, SemaphoreWaitInfo};

#[derive(Debug, Clone)]
/// A timeline semaphore, optionally backed by a DRM syncobj so that it can be exported.
pub struct VulkanTimeline(Arc<VulkanTimelineInternal>);

#[derive(Debug)]
struct VulkanTimelineInternal {
    // A strong reference: sync points may outlive the renderer, and the semaphore must be
    // destroyed before the device.
    pub(super) device: Device,
    pub(super) vk: Semaphore,
    pub(super) drm: Option<DrmTimeline>,
}

impl VulkanTimeline {
    /// Wraps the timeline semaphore `vk` of `device`, with the DRM timeline `drm` if it is exportable.
    pub fn new(device: &Device, vk: Semaphore, drm: impl Into<Option<DrmTimeline>>) -> Self {
        VulkanTimeline(Arc::new(VulkanTimelineInternal {
            device: device.clone(),
            vk,
            drm: drm.into(),
        }))
    }

    /// Returns the Vulkan semaphore.
    pub fn vk(&self) -> Semaphore {
        self.0.vk
    }

    /// Returns the highest point that the semaphore has reached, if querying it succeeds.
    pub fn query_signalled_point(&self) -> Option<u64> {
        unsafe { self.0.device.vk().get_semaphore_counter_value(self.0.vk).ok() }
    }

    /// Whether the timeline can be exported as a [`DrmTimeline`].
    pub fn is_exportable(&self) -> bool {
        self.0.drm.is_some()
    }

    /// Returns the DRM timeline backing the semaphore, if it has one.
    pub fn export(&self) -> Option<DrmTimeline> {
        self.0.drm.clone()
    }
}

impl Drop for VulkanTimelineInternal {
    fn drop(&mut self) {
        unsafe { self.device.vk().destroy_semaphore(self.vk, None) };
    }
}

#[derive(Debug)]
/// A point on a [`VulkanTimeline`] that is reached once a submission has finished.
pub struct VulkanSync {
    /// The timeline the point is on.
    pub timeline: VulkanTimeline,
    /// The value the timeline reaches when the work is done.
    pub seq: u64,
    /// A sync_file that signals together with `seq`, for when the timeline is not exportable.
    pub sync_file: Option<Arc<OwnedFd>>,
}

impl VulkanSync {
    /// Returns the context of the device that the sync belongs to.
    pub fn ctx(&self) -> Option<ContextId<VulkanImage>> {
        Some(self.timeline.0.device.context())
    }

    /// Blocks until the timeline reaches [`Self::seq`], or `timeout` (in nanoseconds) passes.
    pub fn wait(&self, timeout: u64) -> Result<(), VkResult> {
        let device = &self.timeline.0.device;

        let semaphores = [self.timeline.0.vk];
        let values = [self.seq];
        unsafe {
            device.vk().wait_semaphores(
                &SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                timeout,
            )
        }
    }
}
