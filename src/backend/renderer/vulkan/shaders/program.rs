use std::sync::Arc;

use ash::vk::Pipeline;

use crate::backend::{
    renderer::ContextId,
    vulkan::{Device, image::VulkanImage},
};

/// A custom texture program, created by
/// [`VulkanRenderer::compile_custom_texture_shader`](super::super::VulkanRenderer::compile_custom_texture_shader).
///
/// It replaces the default texture shader for the draws of a frame, see
/// [`VulkanFrame::override_default_tex_program`](super::super::VulkanFrame::override_default_tex_program).
#[derive(Debug, Clone)]
pub struct VulkanTexProgram(Arc<ProgramInner>);

#[derive(Debug)]
struct ProgramInner {
    device: Device,
    pipeline: Pipeline,
    uniforms_size: usize,
}

impl VulkanTexProgram {
    pub(in crate::backend::renderer::vulkan) fn new(
        device: &Device,
        pipeline: Pipeline,
        uniforms_size: usize,
    ) -> Self {
        VulkanTexProgram(Arc::new(ProgramInner {
            device: device.clone(),
            pipeline,
            uniforms_size,
        }))
    }

    pub(in crate::backend::renderer::vulkan) fn pipeline(&self) -> Pipeline {
        self.0.pipeline
    }

    pub(in crate::backend::renderer::vulkan) fn context(&self) -> ContextId<VulkanImage> {
        self.0.device.context()
    }

    /// Size in bytes of the custom uniforms this program takes.
    pub fn uniforms_size(&self) -> usize {
        self.0.uniforms_size
    }
}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // Recordings keep the program alive until the GPU is done with them.
        unsafe { self.device.vk().destroy_pipeline(self.pipeline, None) };
    }
}
