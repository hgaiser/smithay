use std::os::unix::prelude::OwnedFd;

use crate::backend::{drm::sync::DrmSyncPoint, renderer::vulkan::VulkanSync};

impl super::Fence for VulkanSync {
    fn is_signaled(&self) -> bool {
        self.timeline
            .query_signalled_point()
            .is_some_and(|val| val >= self.seq)
    }

    fn wait(&self) -> Result<(), super::Interrupted> {
        self.wait(u64::MAX).map_err(|_| super::Interrupted)
    }

    fn is_exportable(&self) -> bool {
        self.timeline.is_exportable() || self.sync_file.is_some()
    }

    fn export(&self) -> Option<OwnedFd> {
        if let Some(sync_file) = &self.sync_file {
            return sync_file.try_clone().ok();
        }
        let timeline = self.timeline.export()?;
        let sync = DrmSyncPoint {
            timeline,
            point: self.seq,
        };
        sync.export_sync_file().ok()
    }
}
