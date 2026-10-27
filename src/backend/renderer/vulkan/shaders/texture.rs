use ash::vk::{DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use bytemuck::NoUninit;
use include_bytes_aligned::include_bytes_aligned;
use std::sync::LazyLock;

use crate::utils::{Physical, Rectangle};

pub const TEX_SHADER: &[u8] = include_bytes_aligned!(32, concat!(env!("OUT_DIR"), "/vk/texture.glsl"));
pub static TEX_BINDINGS: LazyLock<[DescriptorSetLayoutBinding<'static>; 2]> = LazyLock::new(|| {
    [
        DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::STORAGE_IMAGE)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
        DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::COMPUTE)
            .descriptor_count(1),
    ]
});
/// Set in [`TexPushConstants::flags`] when the texture has no alpha channel.
pub const TEX_FLAG_OPAQUE: u32 = 1 << 31;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, NoUninit)]
pub struct TexPushConstants {
    /// Maps a framebuffer pixel position to texture coordinates: `u = dot(uv_x.xyz, (x, y, 1))`.
    pub uv_x: [f32; 4],
    /// Maps a framebuffer pixel position to texture coordinates: `v = dot(uv_y.xyz, (x, y, 1))`.
    pub uv_y: [f32; 4],
    pub _reserved: u32,
    pub alpha: f32,
    /// [`DebugFlags`](crate::backend::renderer::DebugFlags) bits, plus [`TEX_FLAG_OPAQUE`].
    pub flags: u32,
    pub damage_size: u32,
    pub damage: [Rectangle<i32, Physical>; 4],
}
