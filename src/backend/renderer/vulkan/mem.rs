use ash::vk::{
    self, AccessFlags2, BufferImageCopy2, BufferUsageFlags, CopyBufferToImageInfo2, Extent3D,
    ImageAspectFlags, ImageLayout, ImageSubresourceLayers, ImageUsageFlags, Offset3D, PipelineStageFlags2,
};
use smallvec::SmallVec;

use crate::{
    backend::{
        allocator::{
            Buffer as _, Fourcc,
            format::{get_bpp, has_alpha},
        },
        renderer::vulkan::{Error, VulkanRenderer, buffer::Buffer},
        vulkan::{
            format::{get_drm_format, get_vk_format},
            image::{Error as ImageError, VulkanImage},
        },
    },
    utils::{Buffer as BufferCoords, Point, Rectangle, Size},
};

impl VulkanRenderer {
    /// Copies `data` into a host visible staging buffer.
    ///
    /// The data is always copied, as the caller's memory is only valid for the duration of
    /// the call, while the upload is executed asynchronously.
    fn staging_buffer(&self, data: &[u8]) -> Result<Buffer, Error> {
        let buffer = Buffer::new(
            &self.device,
            data.len(),
            None,
            BufferUsageFlags::TRANSFER_SRC,
            true,
        )?;
        let dst = buffer.mapped_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst.as_ptr() as *mut u8, data.len());
        }
        Ok(buffer)
    }

    /// Records and submits a copy of `regions` of `buf` into `img`.
    ///
    /// `buf` holds the image data with the given `stride`, regions are located at their
    /// position in the image. The copy is not waited for, the buffer is kept alive until it
    /// is done. Later work on the same queue that uses the image is ordered after the copy
    /// by the barrier that is recorded when the image is first used.
    fn update_from_buffer(
        &mut self,
        img: &VulkanImage,
        buf: Buffer,
        stride: usize,
        bpp: usize,
        regions: &[Rectangle<i32, BufferCoords>],
    ) -> Result<u64, Error> {
        let size = img.size();
        let mut recording = self.begin_recording()?;
        recording.skip_foreign_fences();
        recording.use_image(
            &self.device,
            img,
            PipelineStageFlags2::ALL_TRANSFER,
            AccessFlags2::TRANSFER_WRITE,
        );
        let cmd = recording.cmd();
        unsafe {
            let regions = regions
                .iter()
                .map(|rect| {
                    BufferImageCopy2::default()
                        .buffer_offset((stride * rect.loc.y as usize + bpp * rect.loc.x as usize) as u64)
                        .buffer_row_length((stride / bpp) as u32)
                        .buffer_image_height(size.h as u32)
                        .image_subresource(
                            ImageSubresourceLayers::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .base_array_layer(0)
                                .layer_count(1)
                                .mip_level(0),
                        )
                        .image_offset(Offset3D::default().x(rect.loc.x).y(rect.loc.y))
                        .image_extent(
                            Extent3D::default()
                                .width(rect.size.w as u32)
                                .height(rect.size.h as u32)
                                .depth(1),
                        )
                })
                .collect::<SmallVec<[BufferImageCopy2<'static>; 4]>>();
            let info = CopyBufferToImageInfo2::default()
                .src_buffer(*buf.vk())
                .dst_image(*img.vk())
                .dst_image_layout(ImageLayout::GENERAL)
                .regions(&regions);
            self.device.vk().cmd_copy_buffer_to_image2(cmd, &info);
        }

        recording.keep_alive(buf);
        self.submit_recording(recording)
    }

    fn new_image(
        &self,
        size: Size<i32, BufferCoords>,
        fourcc: Fourcc,
        usage: vk::ImageUsageFlags,
    ) -> Result<VulkanImage, Error> {
        let format = get_vk_format(fourcc).ok_or(ImageError::UnsupportedFormat)?;
        Ok(VulkanImage::new(
            &self.device,
            size.w as u32,
            size.h as u32,
            format,
            has_alpha(fourcc),
            usage | ImageUsageFlags::TRANSFER_DST,
            false,
        )?)
    }

    pub(super) fn upload_from_slice(
        &mut self,
        data: &[u8],
        size: Size<i32, BufferCoords>,
        stride: usize,
        fourcc: Fourcc,
        usage: vk::ImageUsageFlags,
    ) -> Result<VulkanImage, Error> {
        let bpp = get_bpp(fourcc).ok_or(ImageError::UnsupportedFormat)? / 8;
        let buffer = self.staging_buffer(data)?;
        let image = self.new_image(size, fourcc, usage)?;

        self.update_from_buffer(
            &image,
            buffer,
            stride,
            bpp,
            &[Rectangle::new(Point::new(0, 0), size)],
        )?;

        Ok(image)
    }

    pub(super) fn update_from_slice(
        &mut self,
        data: &[u8],
        stride: usize,
        image: &VulkanImage,
        regions: &[Rectangle<i32, BufferCoords>],
    ) -> Result<(), Error> {
        let format = image.format();
        let bpp = get_drm_format(format)
            .and_then(get_bpp)
            .ok_or(ImageError::UnsupportedFormat)?
            / 8;

        let buffer = self.staging_buffer(data)?;
        self.update_from_buffer(image, buffer, stride, bpp, regions)?;

        Ok(())
    }

    /// Updates regions of an image from shm memory.
    ///
    /// The memory is copied before returning, so it does not need to outlive the call.
    pub(super) fn update_from_shm(
        &mut self,
        data: *const u8,
        len: usize,
        stride: usize,
        image: &VulkanImage,
        regions: &[Rectangle<i32, BufferCoords>],
    ) -> Result<(), Error> {
        // SAFETY: The caller guarantees `data` points to `len` readable bytes.
        let data = unsafe { std::slice::from_raw_parts(data, len) };
        self.update_from_slice(data, stride, image, regions)
    }
}
