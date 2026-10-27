//! Renders the same scenes with the GLES and the Vulkan renderer and compares the pixels.
//!
//! Each scene covers a combination of output transform, texture transform, scaling, source crop,
//! element alpha and damage. The test fails if a channel differs by more than `TOLERANCE`.

use drm_fourcc::DrmFourcc;
use smithay::{
    backend::{
        egl::{EGLContext, EGLDevice, EGLDisplay},
        renderer::{
            Bind, Color32F, ExportMem, Frame, ImportMem, Offscreen, Renderer, Texture, TextureFilter,
            gles::GlesRenderer, sync::Interrupted, vulkan::VulkanRenderer,
        },
        vulkan::{Instance, PhysicalDevice, version::Version},
    },
    utils::{Buffer, Physical, Point, Rectangle, Size, Transform},
};

const TOLERANCE: u8 = 2;
#[allow(non_snake_case)]
fn SIZE() -> Size<i32, Physical> {
    Size::new(96, 64)
}
#[allow(non_snake_case)]
fn TEX_SIZE() -> Size<i32, Buffer> {
    // Odd sizes: a pixel centre never falls exactly on a texel border, where nearest filtering
    // may select either texel.
    Size::new(9, 7)
}

const TRANSFORMS: [Transform; 8] = [
    Transform::Normal,
    Transform::_90,
    Transform::_180,
    Transform::_270,
    Transform::Flipped,
    Transform::Flipped90,
    Transform::Flipped180,
    Transform::Flipped270,
];

/// An asymmetric texture with premultiplied alpha, so every transform gives a different image.
fn texture_data() -> Vec<u8> {
    let mut data = Vec::new();
    for y in 0..TEX_SIZE().h {
        for x in 0..TEX_SIZE().w {
            let a = if (x + y) % 3 == 0 { 128u32 } else { 255 };
            let r = (x * 32) as u32 * a / 255;
            let g = (y * 40) as u32 * a / 255;
            let b = if x == 0 && y == 0 { a } else { 0 };
            data.extend_from_slice(&[r as u8, g as u8, b as u8, a as u8]);
        }
    }
    data
}

#[derive(Debug, Clone, Copy)]
struct Scene {
    output: Transform,
    texture: Transform,
    up: TextureFilter,
    down: TextureFilter,
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    damage: Option<Rectangle<i32, Physical>>,
    alpha: f32,
    flipped: bool,
}

fn draw<R>(renderer: &mut R, target: &mut R::Framebuffer<'_>, tex: &R::TextureId, scene: &Scene)
where
    R: Renderer,
{
    renderer.upscale_filter(scene.up).unwrap();
    renderer.downscale_filter(scene.down).unwrap();
    let mut frame = renderer.render(target, SIZE(), scene.output).unwrap();
    let full = Rectangle::from_size(SIZE());
    frame.clear(Color32F::new(0.1, 0.2, 0.3, 1.0), &[full]).unwrap();
    frame
        .draw_solid(
            Rectangle::new(Point::new(4, 4), Size::new(40, 20)),
            &[Rectangle::from_size(Size::new(40, 20))],
            Color32F::new(0.4, 0.0, 0.0, 0.5),
        )
        .unwrap();
    // Damage is relative to the element destination.
    let damage = scene
        .damage
        .map(|d| Rectangle::new(d.loc - scene.dst.loc, d.size))
        .unwrap_or(Rectangle::from_size(scene.dst.size));
    frame
        .render_texture_from_to(
            tex,
            scene.src,
            scene.dst,
            &[damage],
            &[],
            scene.texture,
            scene.alpha,
        )
        .unwrap();
    let sync = frame.finish().unwrap();
    while let Err(Interrupted) = sync.wait() {}
}

fn read<R>(renderer: &mut R, target: &R::Framebuffer<'_>) -> Vec<u8>
where
    R: Renderer + ExportMem,
{
    let size = SIZE().to_logical(1).to_buffer(1, Transform::Normal);
    let mapping = renderer
        .copy_framebuffer(target, Rectangle::from_size(size), DrmFourcc::Abgr8888)
        .unwrap();
    renderer.map_texture(&mapping).unwrap().to_vec()
}

fn main() {
    let egl_device = EGLDevice::enumerate()
        .unwrap()
        .find(|d| d.try_get_render_node().ok().flatten().is_some())
        .expect("EGL device");
    let display = unsafe { EGLDisplay::new(egl_device) }.unwrap();
    let mut gles = unsafe { GlesRenderer::new(EGLContext::new(&display).unwrap()) }.unwrap();

    let instance = Instance::new(Version::VERSION_1_3, None).unwrap();
    let phd = PhysicalDevice::enumerate(&instance)
        .unwrap()
        .next()
        .expect("device");
    let mut vk = VulkanRenderer::new(&phd, None).unwrap();

    let buf_size = SIZE().to_logical(1).to_buffer(1, Transform::Normal);
    let mut gles_target = Offscreen::<smithay::backend::renderer::gles::GlesTexture>::create_buffer(
        &mut gles,
        DrmFourcc::Abgr8888,
        buf_size,
    )
    .unwrap();
    let mut vk_target = Offscreen::<smithay::backend::vulkan::image::VulkanImage>::create_buffer(
        &mut vk,
        DrmFourcc::Abgr8888,
        buf_size,
    )
    .unwrap();
    let data = texture_data();
    let gles_tex = gles
        .import_memory(&data, DrmFourcc::Abgr8888, TEX_SIZE(), false)
        .unwrap();
    let vk_tex = vk
        .import_memory(&data, DrmFourcc::Abgr8888, TEX_SIZE(), false)
        .unwrap();
    assert_eq!(gles_tex.size(), vk_tex.size());
    // The reference for a flipped texture is the upside-down data, not flipped. The GLES renderer
    // maps v to -v for flipped textures, which gives the edge row with CLAMP_TO_EDGE.
    let row = TEX_SIZE().w as usize * 4;
    let upside_down = data.chunks(row).rev().flatten().copied().collect::<Vec<_>>();
    let gles_flipped = gles
        .import_memory(&upside_down, DrmFourcc::Abgr8888, TEX_SIZE(), false)
        .unwrap();
    let vk_flipped = vk
        .import_memory(&data, DrmFourcc::Abgr8888, TEX_SIZE(), true)
        .unwrap();

    let full_src = Rectangle::from_size(TEX_SIZE().to_f64());
    let mut scenes = Vec::new();
    for output in TRANSFORMS {
        for texture in TRANSFORMS {
            // An integer upscale with nearest filtering: the exact texel must match.
            scenes.push(Scene {
                output,
                texture,
                up: TextureFilter::Nearest,
                down: TextureFilter::Nearest,
                src: full_src,
                dst: Rectangle::new(Point::new(20, 10), Size::new(48, 48)),
                damage: None,
                alpha: 1.0,
                flipped: false,
            });
        }
        // A cropped source, alpha and a partial damage rectangle.
        scenes.push(Scene {
            output,
            texture: Transform::_90,
            up: TextureFilter::Nearest,
            down: TextureFilter::Nearest,
            src: Rectangle::new(Point::new(2.0, 1.0), Size::new(4.0, 4.0)),
            dst: Rectangle::new(Point::new(30, 8), Size::new(40, 40)),
            damage: Some(Rectangle::new(Point::new(40, 16), Size::new(24, 12))),
            alpha: 0.6,
            flipped: false,
        });
        // A non-integer scale with linear filtering.
        scenes.push(Scene {
            output,
            texture: Transform::Normal,
            up: TextureFilter::Linear,
            down: TextureFilter::Linear,
            src: full_src,
            dst: Rectangle::new(Point::new(5, 3), Size::new(70, 45)),
            damage: None,
            alpha: 1.0,
            flipped: false,
        });
        // A texture imported upside down, with a crop and a transform.
        scenes.push(Scene {
            output,
            texture: Transform::_270,
            up: TextureFilter::Nearest,
            down: TextureFilter::Nearest,
            src: Rectangle::new(Point::new(1.0, 2.0), Size::new(7.0, 3.0)),
            dst: Rectangle::new(Point::new(10, 5), Size::new(33, 49)),
            damage: None,
            alpha: 1.0,
            flipped: true,
        });
        // A downscale must use the downscale filter, and an upscale the upscale filter.
        for (up, down) in [
            (TextureFilter::Nearest, TextureFilter::Linear),
            (TextureFilter::Linear, TextureFilter::Nearest),
        ] {
            for dst_size in [Size::new(5, 3), Size::new(60, 50)] {
                scenes.push(Scene {
                    output,
                    texture: Transform::_90,
                    up,
                    down,
                    src: full_src,
                    dst: Rectangle::new(Point::new(9, 7), dst_size),
                    damage: None,
                    alpha: 1.0,
                    flipped: false,
                });
            }
        }
    }

    let mut failures = 0;
    for scene in &scenes {
        let gles_px = {
            let mut fb = gles.bind(&mut gles_target).unwrap();
            draw(
                &mut gles,
                &mut fb,
                if scene.flipped { &gles_flipped } else { &gles_tex },
                scene,
            );
            read(&mut gles, &fb)
        };
        let vk_px = {
            let mut fb = vk.bind(&mut vk_target).unwrap();
            draw(
                &mut vk,
                &mut fb,
                if scene.flipped { &vk_flipped } else { &vk_tex },
                scene,
            );
            read(&mut vk, &fb)
        };
        let mut worst = (0u8, 0usize);
        for (i, (a, b)) in gles_px.iter().zip(&vk_px).enumerate() {
            let diff = a.abs_diff(*b);
            if diff > worst.0 {
                worst = (diff, i);
            }
        }
        if worst.0 > TOLERANCE {
            // Set DUMP=1 to write both images of each failing scene.
            if std::env::var_os("DUMP").is_some() {
                let n = failures;
                let bad = gles_px
                    .chunks(4)
                    .zip(vk_px.chunks(4))
                    .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > TOLERANCE))
                    .count();
                println!("dump {n}: {bad} pixels differ");
                image::save_buffer(
                    format!("dump-{n}-gles.png"),
                    &gles_px,
                    SIZE().w as u32,
                    SIZE().h as u32,
                    image::ColorType::Rgba8,
                )
                .unwrap();
                image::save_buffer(
                    format!("dump-{n}-vk.png"),
                    &vk_px,
                    SIZE().w as u32,
                    SIZE().h as u32,
                    image::ColorType::Rgba8,
                )
                .unwrap();
            }
            failures += 1;
            let px = worst.1 / 4;
            let (x, y) = (px as i32 % SIZE().w, px as i32 / SIZE().w);
            println!(
                "FAIL {:?}/{:?}/{:?} src {:?} damage {:?}: diff {} at ({x},{y}) gles {:?} vulkan {:?}",
                scene.output,
                scene.texture,
                (scene.up, scene.down),
                scene.src.loc,
                scene.damage.is_some(),
                worst.0,
                &gles_px[px * 4..px * 4 + 4],
                &vk_px[px * 4..px * 4 + 4],
            );
        }
    }
    println!("{} of {} scenes match", scenes.len() - failures, scenes.len());
    vk.cleanup().unwrap();
    std::process::exit((failures != 0) as i32);
}
