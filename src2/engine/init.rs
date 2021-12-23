use crate::engine::use_decl::*;
use crate::engine::app::*;

pub trait GameLoop {
    /*fn run(&mut self) -> Result<()> {
        pretty_env_logger::init();

        // Window

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Vulkan Tutorial (Rust)")
            .with_inner_size(LogicalSize::new(1024, 768))
            .build(&event_loop)?;

        // App

        let mut app = unsafe { App::create(&window)? };
        let mut destroying = false;
        let mut minimized = false;

        // Run the app, also add event handling
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                // Render a frame if our Vulkan app is not being destroyed.
                Event::MainEventsCleared if !destroying && !minimized => unsafe { self.update(&window, &mut app)}.unwrap(),
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
        });
    }*/
    fn update(&self, window: &Window, app: &mut App) -> Result<()> {
        unsafe { app.render(); };
        Ok(())
    }
}