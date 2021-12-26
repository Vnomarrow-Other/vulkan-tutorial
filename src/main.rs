use anyhow::{anyhow, Result};
use winit::event::Event;

pub mod main_vulkan;
use main_vulkan::*;

struct MyGameLoop {
}

impl main_vulkan::GameLoop for MyGameLoop {
    fn create(&mut self, app: &mut main_vulkan::App) {
        app.data.camera.position[0] = 6.0;
        app.data.camera.position[2] = 2.0;

        // Load some models on screen
        for model_index in 0..4 {
            let y = (((model_index % 2) as f32) * 2.5) - 1.25;
            let z = (((model_index / 2) as f32) * -2.0) + 1.0;
    
            app.data.model_instances.push(ModelInstance{ model_index: model_index%2,  position: [0.0, y, z]});
        }
    }
    fn update(&mut self, _app: &mut main_vulkan::App) {
        
    }
    fn handle_event(&mut self, _app: &mut main_vulkan::App, _event: &Event<()>) {

    }
}

fn main() -> Result<()> {
    let mut a = MyGameLoop{};
    main_vulkan::main(&mut a, vec!("./resources/gifts.obj".to_string(), "./resources/viking_room.obj".to_string()))?;
    return Ok(());
}