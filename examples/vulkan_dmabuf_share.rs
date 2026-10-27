//! Renderer A clears a GBM DMA-BUF. Renderer A reads it back as the framebuffer and as an
//! imported texture, and an independent renderer B reads it back as an imported texture.
//!
//! This is the handoff of a composited frame to another Vulkan device (for example a video
//! encoder). Both readbacks must show the clear colour.

use std::os::fd::OwnedFd;

use drm_fourcc::{DrmFourcc, DrmModifier};
use smithay::{
    backend::{
        allocator::{
            Allocator, Buffer as _,
            dmabuf::AsDmabuf,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
        },
        drm::DrmDeviceFd,
        renderer::{
            Bind, Color32F, ExportMem, Frame, ImportDma, Renderer, sync::Interrupted, vulkan::VulkanRenderer,
        },
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    utils::{DeviceFd, Physical, Rectangle, Size, Transform},
};

const RENDER_NODE: &str = "/dev/dri/renderD128";

fn renderer() -> VulkanRenderer {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(RENDER_NODE)
        .unwrap();
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let phd = PhysicalDevice::enumerate(&instance).unwrap().next().unwrap();
    VulkanRenderer::new(&phd, Some(DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file))))).unwrap()
}

fn main() {
    let mut a = renderer();
    let mut b = renderer();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(RENDER_NODE)
        .unwrap();
    let mut allocator = GbmAllocator::new(GbmDevice::new(file).unwrap(), GbmBufferFlags::RENDERING);
    let size = Size::<i32, Physical>::new(16, 16);
    let region = Rectangle::from_size((16, 16).into());
    let mut failed = false;

    for fourcc in [
        DrmFourcc::Abgr8888,
        DrmFourcc::Abgr2101010,
        DrmFourcc::Abgr16161616f,
    ] {
        let modifiers = Bind::<smithay::backend::allocator::dmabuf::Dmabuf>::supported_formats(&a)
            .unwrap()
            .iter()
            .filter(|f| f.code == fourcc)
            .map(|f| f.modifier)
            .collect::<Vec<_>>();
        for modifier_set in [modifiers.clone(), vec![DrmModifier::Linear]] {
            if !modifiers.contains(&DrmModifier::Linear) && modifier_set == [DrmModifier::Linear] {
                continue;
            }
            let Ok(buffer) = allocator.create_buffer(16, 16, fourcc, &modifier_set) else {
                println!("{fourcc:?}: allocation failed for {modifier_set:?}");
                continue;
            };
            let mut dmabuf = buffer.export().unwrap();
            let modifier = dmabuf.format().modifier;
            let read_a = {
                let mut fb = a.bind(&mut dmabuf).unwrap();
                let mut frame = a.render(&mut fb, size, Transform::Normal).unwrap();
                frame
                    .clear(Color32F::new(1.0, 0.5, 0.25, 1.0), &[Rectangle::from_size(size)])
                    .unwrap();
                let sync = frame.finish().unwrap();
                while let Err(Interrupted) = sync.wait() {}
                let mapping = a.copy_framebuffer(&fb, region, fourcc).unwrap();
                a.map_texture(&mapping).unwrap()[..8].to_vec()
            };
            let read_a_tex = {
                let tex = a.import_dmabuf(&dmabuf, None).unwrap();
                let mapping = a.copy_texture(&tex, region, fourcc).unwrap();
                a.map_texture(&mapping).unwrap()[..8].to_vec()
            };
            let read_b = {
                let tex = b.import_dmabuf(&dmabuf, None).unwrap();
                let mapping = b.copy_texture(&tex, region, fourcc).unwrap();
                b.map_texture(&mapping).unwrap()[..8].to_vec()
            };
            let ok = read_a == read_b && read_a == read_a_tex && read_a.iter().any(|&x| x != 0);
            failed |= !ok;
            println!(
                "{fourcc:?} {modifier:?}: A {read_a:?} A texture {read_a_tex:?} B {read_b:?} {}",
                if ok { "ok" } else { "WRONG" }
            );
        }
    }
    std::process::exit(failed as i32);
}
