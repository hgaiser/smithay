use ash::vk::{self, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo, ShaderStageFlags};

use crate::backend::vulkan::Device;

mod clear;
mod program;
mod texture;
use self::clear::*;
use self::texture::*;
pub use self::{
    clear::ClearPushConstants,
    program::VulkanTexProgram,
    texture::{TEX_FLAG_OPAQUE, TexPushConstants},
};

pub fn spirv_u32(shader: &[u8]) -> &[u32] {
    let len = shader.len();
    assert!(len.is_multiple_of(4));
    bytemuck::cast_slice(shader)
}

#[derive(Debug)]
pub struct Pipelines {
    device: Device,
    cache: vk::PipelineCache,

    clear_pipeline: Pipeline,
    clear_layout: PipelineLayout,
    clear_set_layout: vk::DescriptorSetLayout,
    tex_pipeline: Pipeline,
    tex_layout: PipelineLayout,
    tex_set_layout: vk::DescriptorSetLayout,
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur when creating the pipelines.
pub enum Error {
    #[error("Failed to create pipeline cache")]
    /// Creating the pipeline cache failed.
    CacheCreation(#[source] vk::Result),
    #[error("Failed to create shader module")]
    /// Creating a shader module failed.
    Shader(#[source] vk::Result),
    #[error("Failed to create descriptor set layout")]
    /// Creating a descriptor set layout failed.
    DescriptorSetLayout(#[source] vk::Result),
    #[error("Failed to create pipeline layout")]
    /// Creating a pipeline layout failed.
    PipelineLayout(#[source] vk::Result),
    #[error("Failed to create pipelines")]
    /// Creating a pipeline failed.
    Pipeline(#[source] vk::Result),
}

impl Pipelines {
    /// Creates the built-in pipelines.
    ///
    /// The texture pipeline layout's push constant range is `tex_push_constants_size` bytes,
    /// so custom texture programs can add uniforms after [`TexPushConstants`].
    pub fn new(device: &Device, tex_push_constants_size: u32) -> Result<Self, Error> {
        let create_info = vk::PipelineCacheCreateInfo::default();
        let cache = unsafe {
            device
                .vk()
                .create_pipeline_cache(&create_info, None)
                .map_err(Error::CacheCreation)?
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(CLEAR_SHADER));
        let clear_shader = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(CLEAR_BINDINGS.as_slice());
        let descriptor_set = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let layouts = [descriptor_set];

        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ClearPushConstants>() as u32)];
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&constants);
        let clear_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };
        let clear_set_layout = descriptor_set;

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32(TEX_SHADER));
        let tex_shader = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(TEX_BINDINGS.as_slice());
        let descriptor_set = unsafe {
            device
                .vk()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(Error::DescriptorSetLayout)?
        };
        let layouts = [descriptor_set];

        let constants = [vk::PushConstantRange::default()
            .stage_flags(ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(tex_push_constants_size)];
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&constants);
        let tex_layout = unsafe {
            device
                .vk()
                .create_pipeline_layout(&create_info, None)
                .map_err(Error::PipelineLayout)?
        };
        let tex_set_layout = descriptor_set;

        let create_infos = [
            vk::ComputePipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                .stage(
                    PipelineShaderStageCreateInfo::default()
                        .stage(ShaderStageFlags::COMPUTE)
                        .name(c"main")
                        .module(clear_shader),
                )
                .layout(clear_layout.clone()),
            vk::ComputePipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                .stage(
                    PipelineShaderStageCreateInfo::default()
                        .stage(ShaderStageFlags::COMPUTE)
                        .name(c"main")
                        .module(tex_shader),
                )
                .layout(tex_layout.clone()),
        ];
        let pipelines = unsafe {
            device
                .vk()
                .create_compute_pipelines(cache, &create_infos, None)
                .map_err(|(pipelines, res)| {
                    for pipeline in pipelines {
                        device.vk().destroy_pipeline(pipeline, None);
                    }
                    res
                })
                .map_err(Error::Pipeline)?
        };

        unsafe {
            device.vk().destroy_shader_module(clear_shader, None);
            device.vk().destroy_shader_module(tex_shader, None);
        }

        Ok(Pipelines {
            device: device.clone(),
            cache,

            clear_pipeline: pipelines[0],
            clear_layout,
            clear_set_layout,
            tex_pipeline: pipelines[1],
            tex_layout,
            tex_set_layout,
        })
    }

    pub fn clear_pipeline(&self) -> &Pipeline {
        &self.clear_pipeline
    }
    pub fn clear_pipeline_layout(&self) -> &PipelineLayout {
        &self.clear_layout
    }

    pub fn tex_pipeline(&self) -> &Pipeline {
        &self.tex_pipeline
    }
    pub fn tex_pipeline_layout(&self) -> &PipelineLayout {
        &self.tex_layout
    }

    /// Creates a compute pipeline from `spirv` with the texture pipeline's layout.
    pub fn create_tex_pipeline(&self, device: &Device, spirv: &[u32]) -> Result<Pipeline, Error> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader = unsafe {
            device
                .vk()
                .create_shader_module(&create_info, None)
                .map_err(Error::Shader)?
        };
        let create_infos = [vk::ComputePipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
            .stage(
                PipelineShaderStageCreateInfo::default()
                    .stage(ShaderStageFlags::COMPUTE)
                    .name(c"main")
                    .module(shader),
            )
            .layout(self.tex_layout)];
        let pipelines = unsafe {
            device
                .vk()
                .create_compute_pipelines(self.cache, &create_infos, None)
                .map_err(|(_, res)| Error::Pipeline(res))
        };
        unsafe { device.vk().destroy_shader_module(shader, None) };
        Ok(pipelines?[0])
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        {
            let device = &self.device;
            unsafe {
                device.vk().destroy_pipeline_layout(self.clear_layout, None);
                device.vk().destroy_pipeline(self.clear_pipeline, None);
                device.vk().destroy_pipeline_layout(self.tex_layout, None);
                device.vk().destroy_pipeline(self.tex_pipeline, None);
                device
                    .vk()
                    .destroy_descriptor_set_layout(self.clear_set_layout, None);
                device
                    .vk()
                    .destroy_descriptor_set_layout(self.tex_set_layout, None);
                device.vk().destroy_pipeline_cache(self.cache, None);
            }
        }
    }
}
