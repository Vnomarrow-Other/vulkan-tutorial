use anyhow::{anyhow, Result};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use nalgebra_glm as glm;

pub mod main_vulkan;
use main_vulkan::*;

#[derive(Debug, Default)]
struct MyGameLoop {
    look_vec: glm::TVec3<f32>,
    angle_x: f64,
    angle_y: f64,
    grab: bool,
    w_pressed: bool,
    s_pressed: bool,
    a_pressed: bool,
    d_pressed: bool,
    space_pressed: bool,
    shift_pressed: bool
}

impl MyGameLoop {
    pub fn load_chess_board(&mut self, app: &mut main_vulkan::App) {
        app.data.model_instances = Default::default();
        self.load_board(app);

        for x in 0..8 {
            self.load_chess_model(app, ChessModel::white_pawn, x, 1);
            self.load_chess_model(app, ChessModel::black_pawn, x, 6);
        }

        self.load_chess_model(app, ChessModel::white_rook, 0, 0);
        self.load_chess_model(app, ChessModel::black_rook, 0, 7);

        self.load_chess_model(app, ChessModel::white_knight, 1, 0);
        self.load_chess_model(app, ChessModel::black_knight, 1, 7);

        self.load_chess_model(app, ChessModel::white_bishop, 2, 0);
        self.load_chess_model(app, ChessModel::black_bishop, 2, 7);

        self.load_chess_model(app, ChessModel::white_queen, 3, 0);
        self.load_chess_model(app, ChessModel::black_queen, 3, 7);

        self.load_chess_model(app, ChessModel::white_king, 4, 0);
        self.load_chess_model(app, ChessModel::black_king, 4, 7);

        self.load_chess_model(app, ChessModel::white_bishop, 5, 0);
        self.load_chess_model(app, ChessModel::black_bishop, 5, 7);

        self.load_chess_model(app, ChessModel::white_knight, 6, 0);
        self.load_chess_model(app, ChessModel::black_knight, 6, 7);

        self.load_chess_model(app, ChessModel::white_rook, 7, 0);
        self.load_chess_model(app, ChessModel::black_rook, 7, 7);
    }
    pub fn load_chess_model(&mut self, app: &mut main_vulkan::App, id: ChessModel, x: usize, y: usize){
        app.data.model_instances.push(ModelInstance{ 
            model_index: id as usize,  
            position: glm::vec3(x as f32 * 2.0, y as f32 * 2.0, 0.0),
            rotate_rad: glm::radians(&glm::vec1(90.0))[0],
            rotate_vec: glm::vec3(0.0, 0.0, 1.0),
        });
    }
    pub fn load_board(&mut self, app: &mut main_vulkan::App) {
        app.data.model_instances.push(ModelInstance{ 
            model_index: ChessModel::board as usize,  
            position: glm::vec3(7.0, 7.0, 0.0),
            rotate_rad: glm::radians(&glm::vec1(90.0))[0],
            rotate_vec: glm::vec3(0.0, 0.0, 1.0),
        });
    }
}

impl main_vulkan::GameLoop for MyGameLoop {
    fn create(&mut self, app: &mut main_vulkan::App) {
        self.look_vec = glm::vec3(6.0, 0.0, 2.0);
        self.angle_x = 0.0;
        self.angle_y = 0.0;
        self.grab = true;
        app.data.camera.position[0] = 6.0;
        app.data.camera.position[2] = 2.0;

        // Load some models on screen
        for model_index in 0..4 {
            let y = (((model_index % 2) as f32) * 2.5) - 1.25;
            let z = (((model_index / 2) as f32) * -2.0) + 1.0;
        }
    }
    fn update(&mut self, app: &mut main_vulkan::App) {
        let speed: f32 = 0.05;
        let vec1: glm::TVec3<f32> = glm::vec3(0.0, 1.0, 0.0);
        let forward: glm::TVec3<f32> = glm::rotate_vec3(&vec1, self.angle_x as f32 + std::f32::consts::PI, &glm::vec3(0.0, 0.0, 1.0));
        let right: glm::TVec3<f32> = glm::rotate_vec3(&vec1, self.angle_x as f32 + std::f64::consts::PI as f32 / 2.0, &glm::vec3(0.0, 0.0, 1.0));
        if self.w_pressed {
            app.data.camera.position += forward * speed;
        }
        if self.s_pressed {
            app.data.camera.position -= forward * speed;
        }
        if self.a_pressed {
            app.data.camera.position -= right * speed;
        }
        if self.d_pressed {
            app.data.camera.position += right * speed;
        }
        if self.space_pressed {
            app.data.camera.position[2] += speed;
        }
        if self.shift_pressed {
            app.data.camera.position[2] -= speed;
        }
        self.load_chess_board(app);
        app.data.camera.looking_at = app.data.camera.position - self.look_vec;
    }
    fn handle_event(&mut self, app: &mut main_vulkan::App, event: &Event<()>, window: &winit::window::Window) {
        match event {
            // Handle keyboard events.
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => self.w_pressed = true,
                        Some(VirtualKeyCode::S) => self.s_pressed = true,
                        Some(VirtualKeyCode::A) => self.a_pressed = true,
                        Some(VirtualKeyCode::D) => self.d_pressed = true,
                        Some(VirtualKeyCode::Space) => self.space_pressed = true,
                        Some(VirtualKeyCode::LShift) => self.shift_pressed = true,
                        Some(VirtualKeyCode::Escape) => {self.grab = !self.grab; window.set_cursor_visible(!self.grab);},
                        _ => { }
                    }
                }
                if input.state == ElementState::Released {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => self.w_pressed = false,
                        Some(VirtualKeyCode::S) => self.s_pressed = false,
                        Some(VirtualKeyCode::A) => self.a_pressed = false,
                        Some(VirtualKeyCode::D) => self.d_pressed = false,
                        Some(VirtualKeyCode::Space) => self.space_pressed = false,
                        Some(VirtualKeyCode::LShift) => self.shift_pressed = false,
                        _ => { }
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                if self.grab {
                    let x = position.x as f32;
                    let y = position.y as f32;
                    let sensitivity = 2.0;
                    self.angle_x -= ((2.0 * std::f64::consts::PI * (position.x - 512.0)) / 1024.0) / sensitivity;
                    self.angle_y -= ((2.0 * std::f64::consts::PI * (position.y - 384.0)) / 768.0) / sensitivity;

                    // Make sure the world does not become upside down
                    if self.angle_y < -std::f64::consts::PI/2.0 {
                        self.angle_y = -std::f64::consts::PI/2.0;
                    }
                    else if self.angle_y > std::f64::consts::PI/2.0 {
                        self.angle_y = std::f64::consts::PI/2.0;
                    }

                    let vec1: glm::TVec3<f32> = glm::vec3(0.0, 1.0, 0.0);
                    let vec2: glm::TVec3<f32> = glm::rotate_vec3(&vec1, self.angle_x as f32, &glm::vec3(0.0, 0.0, 1.0));
                    let vec3: glm::TVec3<f32> = glm::rotate_vec3(&vec1, self.angle_x as f32 + std::f64::consts::PI as f32 / 2.0, &glm::vec3(0.0, 0.0, 1.0));
                    let vec1: glm::TVec3<f32> = glm::rotate_vec3(&vec2, self.angle_y as f32, &vec3.clone());
                    self.look_vec = vec1;

                    //self.look_vec.x = (angle_x - std::f64::consts::PI) as f32;
                    //self.look_vec.y = (angle_y - std::f64::consts::PI) as f32;

                    //println!("x {}  y {}", self.look_vec.x, self.look_vec.y);
                    let _ = window.set_cursor_position(winit::dpi::LogicalPosition::new(512, 384));
                }
            }
            _ => {

            }
        }
    }
}

enum ChessModel {
    board,
    white_pawn,
    white_queen,
    white_knight,
    white_rook,
    white_king,
    white_bishop,
    black_pawn,
    black_queen,
    black_knight,
    black_rook,
    black_king,
    black_bishop,
}

fn main() -> Result<()> {
    let mut a = MyGameLoop::default();
    main_vulkan::main(&mut a, vec!(
        "./resources/chess/board.obj".to_string(),
        "./resources/chess/white_pawn.obj".to_string(),
        "./resources/chess/white_queen.obj".to_string(),
        "./resources/chess/white_knight.obj".to_string(),
        "./resources/chess/white_rook.obj".to_string(),
        "./resources/chess/white_king.obj".to_string(),
        "./resources/chess/white_bishop.obj".to_string(),
        "./resources/chess/black_pawn.obj".to_string(),
        "./resources/chess/black_queen.obj".to_string(),
        "./resources/chess/black_knight.obj".to_string(),
        "./resources/chess/black_rook.obj".to_string(),
        "./resources/chess/black_king.obj".to_string(),
        "./resources/chess/black_bishop.obj".to_string(),
    )
    )?;
    return Ok(());
}