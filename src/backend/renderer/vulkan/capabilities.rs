use std::{
    ffi::{CStr, c_void},
    pin::Pin,
};

use ash::{
    ext, khr,
    vk::{
        ExternalSemaphoreFeatureFlags, ExternalSemaphoreHandleTypeFlags, ExternalSemaphoreProperties,
        PhysicalDeviceExternalSemaphoreInfo, PhysicalDeviceFeatures2, PhysicalDeviceVulkan11Features,
        PhysicalDeviceVulkan12Features, PhysicalDeviceVulkan13Features, SemaphoreType,
        SemaphoreTypeCreateInfo,
    },
};

use crate::backend::vulkan::{PhysicalDevice, version::Version};

/// `VK_EXT_queue_family_foreign`, which ash has no module for.
pub(crate) const QUEUE_FAMILY_FOREIGN_NAME: &CStr = c"VK_EXT_queue_family_foreign";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// An optional capability of a physical device that the renderer can make use of.
pub enum Capability {
    /// Import and export of DMA-BUF memory with DRM format modifiers.
    DmabufMemory,
    /// `VK_EXT_queue_family_foreign`, used to hand DMA-BUF images to other devices and back.
    QueueFamilyForeign,
    /// Import of host memory (`VK_EXT_external_memory_host`).
    HostMemory,
    /// Timeline semaphores can be imported and exported as opaque fds.
    ExportTimeline,
    /// sync_files can be imported into binary semaphores, so foreign fences are waited on by the GPU.
    ImportSyncFile,
    /// Binary semaphores can be exported as sync_files. Used to export sync points when the
    /// timeline is not a DRM syncobj, as on the NVIDIA proprietary driver.
    ExportSyncFile,
}

#[derive(Debug)]
/// The device features that are enabled for the renderer.
///
/// The structure is pinned, because the feature structures are chained through raw pointers.
pub struct Features {
    features: PhysicalDeviceFeatures2<'static>,
    features_11: PhysicalDeviceVulkan11Features<'static>,
    features_12: PhysicalDeviceVulkan12Features<'static>,
    // we require vulkan 1.2 at minimum, but the renderer itself needs 1.3
    features_13: Option<PhysicalDeviceVulkan13Features<'static>>,
    _self_ref: std::marker::PhantomPinned,
}

impl Features {
    /// Creates a feature chain with all features disabled, for Vulkan `version`.
    pub fn new(version: &Version) -> Pin<Box<Self>> {
        let mut features = Box::pin(Features {
            features: PhysicalDeviceFeatures2::default(),
            features_11: PhysicalDeviceVulkan11Features::default(),
            features_12: PhysicalDeviceVulkan12Features::default(),
            features_13: None,
            _self_ref: std::marker::PhantomPinned,
        });

        {
            let features = unsafe { features.as_mut().get_unchecked_mut() };
            features.features.p_next = &mut features.features_11 as *mut _ as *mut c_void;
            features.features_11.p_next = &mut features.features_12 as *mut _ as *mut c_void;

            if version >= &Version::VERSION_1_3 {
                features.features_13 = Some(PhysicalDeviceVulkan13Features::default());
                features.features_12.p_next = features.features_13.as_mut().unwrap() as *mut _ as *mut c_void;
            }
        }

        features
    }

    /// Creates a feature chain with all features that the renderer requires enabled.
    pub fn required_features() -> Pin<Box<Self>> {
        let mut features = Self::new(&Version::VERSION_1_3);

        {
            let features = unsafe { features.as_mut().get_unchecked_mut() };
            features.features_12.timeline_semaphore = 1;
            features.features_13.as_mut().unwrap().synchronization2 = 1;
            // The shaders declare the target storage image without a format, so one
            // pipeline works for every target format.
            features
                .features
                .features
                .shader_storage_image_read_without_format = 1;
            features
                .features
                .features
                .shader_storage_image_write_without_format = 1;
        }

        features
    }

    /// Checks that all required features are enabled.
    ///
    /// Returns the name of the first missing feature as the error.
    pub fn has_required_features(&self) -> Result<(), &'static str> {
        // TODO: I'd love to have a more generic `satisfies(&self, other: &Self)` method,
        // but there isn't a nice way to iterate over fields of a struct and do the necessary bool comparision.
        // And I am not at the point yet of thinking to write a vulkan generator, na-uh!

        if self.features_12.timeline_semaphore == 0 {
            return Err("timeline_semaphore");
        }
        if self.features_13.as_ref().is_none_or(|f| f.synchronization2 == 0) {
            return Err("synchronization2");
        }
        if self.features.features.shader_storage_image_read_without_format == 0 {
            return Err("shader_storage_image_read_without_format");
        }
        if self.features.features.shader_storage_image_write_without_format == 0 {
            return Err("shader_storage_image_write_without_format");
        }

        Ok(())
    }

    /// Queries the features that `phd` supports.
    pub fn supported_features(phd: &PhysicalDevice) -> Pin<Box<Self>> {
        let mut features = Self::new(&phd.api_version());

        unsafe {
            phd.instance().handle().get_physical_device_features2(
                phd.handle(),
                &mut features.as_mut().get_unchecked_mut().features,
            );
        }

        features
    }

    // SAFETY: You must not move the struct
    /// Returns the head of the feature chain, to be passed to device creation.
    ///
    /// # Safety
    ///
    /// The returned structure must not be moved out of the pinned allocation.
    pub unsafe fn vk(self: &mut Pin<Box<Self>>) -> &mut PhysicalDeviceFeatures2<'static> {
        unsafe { &mut self.as_mut().get_unchecked_mut().features }
    }
}

impl Capability {
    /// Returns [`Capability::ExportTimeline`] if `phd` can import and export timeline semaphores as opaque fds.
    pub fn supports_export_timeline(phd: &PhysicalDevice) -> Option<Capability> {
        if !phd.has_device_extension(khr::external_semaphore_fd::NAME) {
            return None;
        }

        let mut create_info = SemaphoreTypeCreateInfo::default().semaphore_type(SemaphoreType::TIMELINE);
        let external_info = PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
            .push_next(&mut create_info);
        let mut semaphore_props = ExternalSemaphoreProperties::default();
        unsafe {
            phd.instance()
                .handle()
                .get_physical_device_external_semaphore_properties(
                    phd.handle(),
                    &external_info,
                    &mut semaphore_props,
                )
        };

        semaphore_props
            .external_semaphore_features
            .contains(ExternalSemaphoreFeatureFlags::IMPORTABLE | ExternalSemaphoreFeatureFlags::EXPORTABLE)
            .then_some(Capability::ExportTimeline)
    }

    /// Returns [`Capability::HostMemory`] if `phd` supports importing host memory.
    pub fn supports_host_memory(phd: &PhysicalDevice) -> Option<Capability> {
        if !phd.has_device_extension(ext::external_memory_host::NAME) {
            return None;
        }

        Some(Capability::HostMemory)
    }

    /// Returns [`Capability::ExportSyncFile`] if `phd` can export binary semaphores as sync_files.
    pub fn supports_export_sync_file(phd: &PhysicalDevice) -> Option<Capability> {
        Self::binary_sync_fd_features(phd)?
            .contains(ExternalSemaphoreFeatureFlags::EXPORTABLE)
            .then_some(Capability::ExportSyncFile)
    }

    /// Returns [`Capability::ImportSyncFile`] if `phd` can import sync_files into binary semaphores.
    pub fn supports_import_sync_file(phd: &PhysicalDevice) -> Option<Capability> {
        Self::binary_sync_fd_features(phd)?
            .contains(ExternalSemaphoreFeatureFlags::IMPORTABLE)
            .then_some(Capability::ImportSyncFile)
    }

    fn binary_sync_fd_features(phd: &PhysicalDevice) -> Option<ExternalSemaphoreFeatureFlags> {
        if !phd.has_device_extension(khr::external_semaphore_fd::NAME) {
            return None;
        }

        let mut create_info = SemaphoreTypeCreateInfo::default().semaphore_type(SemaphoreType::BINARY);
        let external_info = PhysicalDeviceExternalSemaphoreInfo::default()
            .handle_type(ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .push_next(&mut create_info);
        let mut semaphore_props = ExternalSemaphoreProperties::default();
        unsafe {
            phd.instance()
                .handle()
                .get_physical_device_external_semaphore_properties(
                    phd.handle(),
                    &external_info,
                    &mut semaphore_props,
                )
        };

        Some(semaphore_props.external_semaphore_features)
    }

    /// Returns [`Capability::DmabufMemory`] if `phd` supports importing and exporting DMA-BUF memory.
    pub fn supports_dmabuf_memory(phd: &PhysicalDevice) -> Option<Capability> {
        for ext in [
            khr::external_memory_fd::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
        ] {
            if !phd.has_device_extension(ext) {
                return None;
            }
        }

        Some(Capability::DmabufMemory)
    }

    /// Returns [`Capability::QueueFamilyForeign`] if `phd` supports `VK_EXT_queue_family_foreign`.
    pub fn supports_queue_family_foreign(phd: &PhysicalDevice) -> Option<Capability> {
        phd.has_device_extension(QUEUE_FAMILY_FOREIGN_NAME)
            .then_some(Capability::QueueFamilyForeign)
    }

    /// Returns the device extensions that are needed for `caps`.
    pub fn as_extensions(caps: &[Capability]) -> Vec<&CStr> {
        caps.iter()
            .flat_map(|cap| match cap {
                Capability::DmabufMemory => &[
                    khr::external_memory_fd::NAME,
                    ext::image_drm_format_modifier::NAME,
                    ext::external_memory_dma_buf::NAME,
                ] as &'static [&CStr],
                Capability::QueueFamilyForeign => &[QUEUE_FAMILY_FOREIGN_NAME],
                Capability::HostMemory => &[ext::external_memory_host::NAME],
                Capability::ExportTimeline => &[khr::external_semaphore_fd::NAME],
                Capability::ImportSyncFile | Capability::ExportSyncFile => {
                    &[khr::external_semaphore_fd::NAME]
                }
            })
            .copied()
            .fold(Vec::new(), |mut extensions, extension| {
                // Several capabilities can need the same extension.
                if !extensions.contains(&extension) {
                    extensions.push(extension);
                }
                extensions
            })
    }
}
