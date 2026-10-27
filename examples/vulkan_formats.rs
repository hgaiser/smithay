//! Imports one pixel in each memory format, draws it over blue and reads it back.
//!
//! The source colour is (1.0, 0.5, 0.25). Formats with alpha use alpha 1.0. Formats without alpha
//! store 0 in the padding bits, so the renderer must ignore them. The expected result is
//! (255, 128, 64, 255) for every format.

use drm_fourcc::{DrmFourcc, DrmModifier};
use smithay::{
    backend::{
        allocator::{Allocator, vulkan::VulkanAllocator},
        renderer::{
            Bind, Color32F, ExportMem, Frame, ImportMem, Renderer, sync::Interrupted, vulkan::VulkanRenderer,
        },
        vulkan::{Instance, PhysicalDevice, image::ImageUsageFlags, version::Version},
    },
    utils::{Buffer, Physical, Rectangle, Size, Transform},
};

fn pixel(fourcc: DrmFourcc) -> Vec<u8> {
    let (r8, g8, b8) = (255u32, 128u32, 64u32);
    let (r10, g10, b10) = (1023u32, 512u32, 256u32);
    let (r16, g16, b16, one16) = (0x3c00u16, 0x3800u16, 0x3400u16, 0x3c00u16);
    let words: Vec<u8> = match fourcc {
        DrmFourcc::Argb8888 => (255 << 24 | r8 << 16 | g8 << 8 | b8).to_le_bytes().to_vec(),
        DrmFourcc::Xrgb8888 => (r8 << 16 | g8 << 8 | b8).to_le_bytes().to_vec(),
        DrmFourcc::Abgr8888 => (255 << 24 | b8 << 16 | g8 << 8 | r8).to_le_bytes().to_vec(),
        DrmFourcc::Xbgr8888 => (b8 << 16 | g8 << 8 | r8).to_le_bytes().to_vec(),
        DrmFourcc::Argb2101010 => (3 << 30 | r10 << 20 | g10 << 10 | b10).to_le_bytes().to_vec(),
        DrmFourcc::Xrgb2101010 => (r10 << 20 | g10 << 10 | b10).to_le_bytes().to_vec(),
        DrmFourcc::Abgr2101010 => (3 << 30 | b10 << 20 | g10 << 10 | r10).to_le_bytes().to_vec(),
        DrmFourcc::Xbgr2101010 => (b10 << 20 | g10 << 10 | r10).to_le_bytes().to_vec(),
        DrmFourcc::Abgr16161616f => [r16, g16, b16, one16]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DrmFourcc::Xbgr16161616f => [r16, g16, b16, 0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => unreachable!(),
    };
    words.repeat(16)
}

fn main() {
    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let phd = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("device");
    let mut renderer = VulkanRenderer::new(&phd, None).expect("renderer");
    let mut allocator = VulkanAllocator::from_renderer(
        &renderer,
        ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC,
    );
    let size = Size::<i32, Physical>::new(4, 4);
    let buf_size = Size::<i32, Buffer>::new(4, 4);
    let mut target = allocator
        .create_buffer(4, 4, DrmFourcc::Abgr8888, &[DrmModifier::Linear])
        .expect("target");

    let mem_formats = renderer.mem_formats().collect::<Vec<_>>();
    println!("mem_formats: {mem_formats:?}");
    let mut failed = mem_formats.len() != 10;
    for fourcc in [
        DrmFourcc::Argb8888,
        DrmFourcc::Xrgb8888,
        DrmFourcc::Abgr8888,
        DrmFourcc::Xbgr8888,
        DrmFourcc::Argb2101010,
        DrmFourcc::Xrgb2101010,
        DrmFourcc::Abgr2101010,
        DrmFourcc::Xbgr2101010,
        DrmFourcc::Abgr16161616f,
        DrmFourcc::Xbgr16161616f,
    ] {
        let tex = match renderer.import_memory(&pixel(fourcc), fourcc, buf_size, false) {
            Ok(tex) => tex,
            Err(err) => {
                println!("{fourcc:?}: import failed: {err:?}");
                failed = true;
                continue;
            }
        };
        let mut fb = renderer.bind(&mut target).expect("bind");
        let mut frame = renderer.render(&mut fb, size, Transform::Normal).expect("frame");
        frame
            .clear(Color32F::new(0., 0., 1., 1.), &[Rectangle::from_size(size)])
            .expect("clear");
        frame
            .render_texture_from_to(
                &tex,
                Rectangle::from_size(buf_size.to_f64()),
                Rectangle::from_size(size),
                &[Rectangle::from_size(size)],
                &[],
                Transform::Normal,
                1.0,
            )
            .expect("draw");
        let sync = frame.finish().unwrap();
        while let Err(Interrupted) = sync.wait() {}
        let mapping = renderer
            .copy_framebuffer(&fb, Rectangle::from_size(buf_size), DrmFourcc::Abgr8888)
            .expect("copy");
        let px = renderer.map_texture(&mapping).expect("map")[..4].to_vec();
        // 8-bit sources give 128 and 64. 10-bit and FP16 sources give 128 and 64 within rounding.
        let ok = px[0] == 255 && px[1].abs_diff(128) <= 1 && px[2].abs_diff(64) <= 1 && px[3] == 255;
        failed |= !ok;
        println!("{fourcc:?}: {px:?} {}", if ok { "ok" } else { "WRONG" });
    }
    renderer.cleanup().unwrap();
    std::process::exit(failed as i32);
}
