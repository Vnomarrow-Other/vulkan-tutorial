use anyhow::{anyhow, Result};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use nalgebra_glm as glm;

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
    
            app.data.model_instances.push(ModelInstance{ 
                model_index: model_index%2,  
                position: [0.0, y, z],
                rotate_rad: glm::radians(&glm::vec1(90.0))[0],
                rotate_vec: [0.0, 0.0, 0.1]
            });
        }
    }
    fn update(&mut self, _app: &mut main_vulkan::App) {
        
    }
    fn handle_event(&mut self, app: &mut main_vulkan::App, event: &Event<()>) {
        match event {
            // Handle keyboard events.
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => app.data.camera.position[0] += 1.0,
                        Some(VirtualKeyCode::S) => app.data.camera.position[0] -= 1.0,
                        Some(VirtualKeyCode::A) => app.data.camera.position[1] += 1.0,
                        Some(VirtualKeyCode::D) => app.data.camera.position[1] -= 1.0,
                        Some(VirtualKeyCode::Space) => app.data.camera.position[2] += 1.0,
                        Some(VirtualKeyCode::LShift) => app.data.camera.position[2] -= 1.0,
                        _ => { }
                    }
                }
            }
            _ => {

            }
        }
    }
}

fn main() -> Result<()> {
    let mut a = MyGameLoop{};
    main_vulkan::main(&mut a, vec!("./resources/gifts.obj".to_string(), "./resources/viking_room.obj".to_string()))?;
    return Ok(());
}