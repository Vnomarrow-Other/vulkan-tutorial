use crate::main_vulkan::*;
use crate::include::*;
use crate::texture::*;
use crate::device::*;
use crate::swapchain::*;
use crate::pipeline::*;
use crate::buffer::*;
use crate::model::*;
use crate::buffer::*;
use crate::debug::*;

#[rustfmt::skip]
pub fn main(game_loop: &mut dyn GameLoop, model_loaders: Vec<ModelLoader>) -> Result<()> {
    pretty_env_logger::init();

    // Init the Window

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;
        window.set_cursor_visible(false);
    //window.set_cursor_grab(true)?;
    // Init the App

    let mut app = unsafe { App::create(&window, model_loaders)? };
    game_loop.create(&mut app);
    let mut destroying = false;
    let mut minimized = false;

    let mut loop_timer = TimeTracker::new();
    event_loop.run_return(move |event, _, control_flow| {

        //loop_timer.print_elapsed("since loop end");

        let mut time_tracker = TimeTracker::new();
        // Update the game loop
        game_loop.handle_event(&mut app, &event, &window);

        // Handle incomming events
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized => unsafe {
                //loop_timer.print_elapsed("loop_time");
                game_loop.update(&mut app); 
                loop_timer.reset(); 
                app.render(&window).unwrap();
            },
            // Mark the window as having been resized.
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            // Destroy our Vulkan app.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.destroy(); }
            }
            _ => {}
        }
        //time_tracker.print_elapsed("main");
    });
    return Ok(());
}

/// Our Vulkan app.
#[derive(Clone, Debug)]
pub struct App {
    entry: Entry,
    instance: Instance,
    pub data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    models: usize,
    fps_tracker: FPSTracker,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window, model_loaders: Vec<ModelLoader>) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_command_pools(&instance, &device, &mut data)?;
        create_color_objects(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;
        for i in 0..model_loaders.len() {
            model_loaders[i].load(&instance, &device, &mut data)?;
        }
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            models: 1,
            fps_tracker: FPSTracker::new(true)
        })
    }

    /// Render a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        // Wait for GPU
        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::max_value())?;

        // Get the next free image on the gpu to draw to
        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::max_value(),
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        // Make sure window was not resized or altered so that the image became invalid
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // Wait for GPU to finish with the image
        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::max_value())?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        //TODO: copy the image from the gpu

        // Update which commands are sent to GPU
        self.update_command_buffer(image_index)?;

        self.update_uniform_buffer(image_index)?;

        // Do other stuff; not sure what it does
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        // Reset fences, used to make sure cpu waits before gpu is finished
        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        // Get a reference to the swapchain of the image
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Record that the frame was updated in order to track fps(frames per second)
        self.fps_tracker.record_frame();

        Ok(())
    }

    /// Updates a command buffer for our Vulkan app
    /// (The command buffer contains our shaders and determine what will execute on GPU)
    #[rustfmt::skip]
    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Reset
        //println!("{}", image_index);
        let command_pool = self.data.command_pools[image_index];
        //self.device.free_command_buffers(self.data.command_pool, &[self.data.command_buffers[image_index]]);
        //self.device.destroy_buffer(&self.data.command_buffers[image_index], None);

        for i in &self.data.secondary_command_buffers[image_index] {
            self.device.free_command_buffers(command_pool, &[*i]);
        }
        // Fix bug! (reset_command_pool does not fully deallocate all buffers)
        self.device.free_command_buffers(command_pool, &[self.data.command_buffers[image_index]]);
        self.data.secondary_command_buffers[image_index] = vec!();

        // This call takes longer and longer time
        let time_tracker = TimeTracker::new();
        self.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        //time_tracker.print_elapsed("reset_command_pool");

        // None of the other calls are any slower

        let time_tracker = TimeTracker::new();

        //self.device.free_command_buffers(self.data.command_pool, &[self.data.command_buffers[image_index]]);

        // Allocate the command Buffer

        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
        self.data.command_buffers[image_index] = command_buffer;

        // Add commands to command buffer

        let info = vk::CommandBufferBeginInfo::builder();

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);

        // Add secondary command buffers to primary command buffer
        let secondary_command_buffers = (0..self.data.model_instances.len())
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;

        //println!("{}", self.data.model_instances.len());

        // Ask GPU to execute the commands (I think? )
        if secondary_command_buffers.len() != 0 {
            self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
            //device.free_memory(self.index_buffer_memory, None);
            //device.destroy_buffer(self.index_buffer, None);
        }

        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        //time_tracker.print_elapsed("update_command_buffer");

        Ok(())
    }

    /// Updates a secondary command buffer for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_instance_index: usize,
    ) -> Result<vk::CommandBuffer> {
        // Allocate the secondary command buffer

        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.data.command_pools[image_index])
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);

        let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];

        // Push Constants

        let x = self.data.model_instances[model_instance_index].position[0] / RENDER_DISTANCE;
        let y = self.data.model_instances[model_instance_index].position[1] / RENDER_DISTANCE;
        let z = self.data.model_instances[model_instance_index].position[2] / RENDER_DISTANCE;

        //let model_index = 0;

        // Create the Model matrix(applied to all vertecies), set the positions of the model
        let model = glm::translate(
            &glm::identity(),
            &glm::vec3(x, y, z),
        );

        // Get the time since the app started
        let time = self.fps_tracker.start.elapsed().as_secs_f32();

        let rotate_vec = glm::vec3( self.data.model_instances[model_instance_index].rotate_vec[0], 
            self.data.model_instances[model_instance_index].rotate_vec[1], 
            self.data.model_instances[model_instance_index].rotate_vec[2]);
        //let rotate_rad = time * glm::radians(&glm::vec1(90.0))[0];
        let rotate_rad = self.data.model_instances[model_instance_index].rotate_rad;

        // Change to model matrix to also do a rotation
        let model = glm::rotate(
            &model,
            rotate_rad,
            &rotate_vec.clone(),
        );

        let model_index = self.data.model_instances[model_instance_index].model_index;

        let (_, model_bytes, _) = model.as_slice().align_to::<u8>();

        // Set the opacity of the model(How transperant it is)
        let opacity = 1.0 as f32;
        //let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        // Add commands to the secondary command buffer
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);
        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.models[model_index].vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(command_buffer, self.data.models[model_index].index_buffer, 0, vk::IndexType::UINT32);
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );

        self.data.secondary_command_buffers[image_index].push(command_buffer);
        self.device.cmd_draw_indexed(command_buffer, self.data.models[model_index].indices.len() as u32, 1, 0, 0, 0);

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    /// Updates the uniform buffer object for our Vulkan app. 
    /// The uniform buffer are applied to all models and contains things such
    ///  as transformations matrix for the position and angle of the camera looking
    ///  at the world
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // Set the camera view point

        // Set camera position
        let eye = glm::vec3(self.data.camera.position[0] / RENDER_DISTANCE, self.data.camera.position[1] / RENDER_DISTANCE, self.data.camera.position[2] / RENDER_DISTANCE);

        // Set where camera is looking
        let center = glm::vec3(self.data.camera.looking_at[0] / RENDER_DISTANCE, self.data.camera.looking_at[1] / RENDER_DISTANCE, self.data.camera.looking_at[2] / RENDER_DISTANCE);

        // Set what is "up" (normally y vector)
        let up = glm::vec3(0.0, 0.0, 1.0);
        let view = glm::look_at(
            &eye.clone(),
            &center.clone(),
            &up.clone(),
        );

        // Set some camera parameters
        let mut proj = glm::perspective_rh_zo(
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            glm::radians(&glm::vec1(45.0))[0],
            0.1,
            10.0,
        );

        // Handle a bug with coordinate being flipped
        proj[(1, 1)] *= -1.0;

        let ubo = UniformBufferObject { view, proj };

        // Copy to buffer to the GPU
        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    /// Recreates the swapchain for our Vulkan app. 
    /// (Swapchain are used so we can have one image to render to, and one image to draw to. 
    ///     When the draw image is complete, they "swap")
    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        let time_tracker = TimeTracker::new();
        self.device.device_wait_idle()?;
        time_tracker.print_elapsed("time recreate_swapchain: ");
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroy our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.command_pools.iter().for_each(|p| self.device.destroy_command_pool(*p, None));

        // Delete our models
        for i in 0..self.data.models.len() {
            self.data.models[i].destroy(&self.device);
        }

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
        self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys the parts of our Vulkan app related to the swapchain.
    /// (Swapchain are used so we can have one image to render to, and one image to draw to. 
    ///     When the draw image is complete, they "swap")
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

// Game game loop that can be passed in for someoune using our library
//  Holds all the game code and is updated frequently
pub trait GameLoop {
    fn create(&mut self, app: &mut App);
    fn update(&mut self, app: &mut App);
    fn handle_event(&mut self, app: &mut App, event: &Event<()>, window: &Window);
}

// The camera specifying where and how to look at the world
#[derive(Clone, Debug, Default)]
pub struct Camera {
    pub position: glm::TVec3<f32>,
    pub looking_at: glm::TVec3<f32>
}

/*
Chap 6 & 7
The instance is the connection between your application and the Vulkan library 
 and creating it involves specifying some details about your application to the driver. 

The Vulkan API is designed around the idea of minimal driver overhead and one of the 
manifestations of that goal is that there is very limited error checking in the API by 
default. Vulkan introduces an elegant system for this known as validation layers. 
Validation layers are optional components that hook into Vulkan function calls to 
apply additional operations. Common operations in validation layers are:
    Checking the values of parameters against the specification to detect misuse
    Tracking creation and destruction of objects to find resource leaks
    Checking thread safety by tracking the threads that calls originate from
    Logging every call and its parameters to the standard output
    Tracing Vulkan calls for profiling and replaying
*/
pub unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial (Rust)\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
pub struct AppData {
    // Debug
    pub messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    pub surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    // Swapchain
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    // Framebuffers
    pub framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    pub command_pool: vk::CommandPool,
    // Color
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
    // Depth
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    // Texture
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    // Model
    pub models: Vec<MyModel>,
    pub model_instances: Vec<ModelInstance>,
    pub camera: Camera,
    // Buffers
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<CommandBuffer>>,
    // Sync Objects
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}