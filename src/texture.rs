use crate::include::*;
use crate::main_vulkan::*;
use crate::buffer::*;
use crate::swapchain::*;
use crate::app::*;

pub unsafe fn create_texture_image(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Load

    //let image = File::open("./resources/viking_room.png")?;
    let image = File::open("./resources/grey_square.png")?;

    let decoder = png::Decoder::new(image);
    let (info, mut reader) = decoder.read_info()?;

    let mut pixels = vec![0; info.buffer_size()];
    reader.next_frame(&mut pixels)?;

    let size = info.buffer_size() as u64;
    data.mip_levels = (info.width.max(info.height) as f32).log2().floor() as u32 + 1;

    // Create (staging)

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (image)

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        info.width,
        info.height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // Transition + Copy (image)

    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;

    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        info.width,
        info.height,
    )?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    // Mipmaps

    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        info.width,
        info.height,
        data.mip_levels,
    )?;

    Ok(())
}

/*
Chap 33
Mipmaps are precalculated, downscaled versions of an image. 
Each new image is half the width and height of the previous 
one. Mipmaps are used as a form of Level of Detail or LOD. 
Objects that are far away from the camera will sample their 
textures from the smaller mip images. Using smaller images 
increases the rendering speed and avoids artifacts such as 
Moiré patterns. 
*/
pub unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // Support

    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!("Texture image format does not support linear blitting!"));
    }

    // Mipmaps

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}