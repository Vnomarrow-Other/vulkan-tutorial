use anyhow::{anyhow, Result};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use nalgebra_glm as glm;

pub mod main_vulkan;
pub mod util;
pub mod time;
pub mod app;
pub mod include;
pub mod model;
pub mod texture;
pub mod debug;
pub mod device;
pub mod swapchain;
pub mod pipeline;
pub mod buffer;
use main_vulkan::*;
use crate::app::*;
use crate::model::*;

use chess_engine::chess_game::*;

/*
line_p: a point on the line
line_v: the vector of the line
pln_eq: equation of the plane[a, b, c, d]: ax + by + cy + d = 0
*/
pub fn line_intersect_plane(line_p: glm::TVec3<f32>, line_v: glm::TVec3<f32>, pln_eq: [f32; 4]) -> Option<glm::TVec3<f32>> {
    let dot_v_plane = line_v[0] * pln_eq[0] + line_v[1] * pln_eq[1] + line_v[2] * pln_eq[2];
    if dot_v_plane == 0.0 {
        // Line does not intersect plane
        // (The normal of the plane is at 90dgr from the line, the plane and the line are parallell)
        return None;
    }
    let t = -(line_p[0] * pln_eq[0] + line_p[1] * pln_eq[1] + line_p[2] * pln_eq[2] + pln_eq[3]) / dot_v_plane;

    let x = line_p[0] + t * line_v[0];
    let y = line_p[1] + t * line_v[1];
    let z = line_p[2] + t * line_v[2];

    return Some(glm::vec3(x as f32, y as f32, z as f32));
}

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
    shift_pressed: bool,
    selected_x: usize,
    selected_y: usize,
    chess_engine: chess_engine::chess_game::Game
}

impl MyGameLoop {
    pub fn load_chess_board(&mut self, app: &mut App) {
        app.data.model_instances = Default::default();
        self.load_board(app);

        for x in 0..8 {
            for y in 0..8 {
                let chess_piece = self.chess_engine.get_board_piece_clone(BoardPosition::new(x as u8, y as u8));
                if chess_piece.is_none() {
                    continue;
                }
                let chess_piece = chess_piece.unwrap();
                match chess_piece.color {
                    ChessPieceColor::White => {
                        match chess_piece.id {
                            ChessPieceId::Bishop => {
                                self.load_chess_model(app, ChessModel::white_bishop, x, y);
                            }
                            ChessPieceId::Rook => {
                                self.load_chess_model(app, ChessModel::white_rook, x, y);
                            }
                            ChessPieceId::King => {
                                self.load_chess_model(app, ChessModel::white_king, x, y);
                            }
                            ChessPieceId::Queen => {
                                self.load_chess_model(app, ChessModel::white_queen, x, y);
                            }
                            ChessPieceId::Knight => {
                                self.load_chess_model(app, ChessModel::white_knight, x, y);
                            }
                            ChessPieceId::Pawn => {
                                self.load_chess_model(app, ChessModel::white_pawn, x, y);
                            }
                        }
                    }
                    ChessPieceColor::Black => {
                        match chess_piece.id {
                            ChessPieceId::Bishop => {
                                self.load_chess_model(app, ChessModel::black_bishop, x, y);
                            }
                            ChessPieceId::Rook => {
                                self.load_chess_model(app, ChessModel::black_rook, x, y);
                            }
                            ChessPieceId::King => {
                                self.load_chess_model(app, ChessModel::black_king, x, y);
                            }
                            ChessPieceId::Queen => {
                                self.load_chess_model(app, ChessModel::black_queen, x, y);
                            }
                            ChessPieceId::Knight => {
                                self.load_chess_model(app, ChessModel::black_knight, x, y);
                            }
                            ChessPieceId::Pawn => {
                                self.load_chess_model(app, ChessModel::black_pawn, x, y);
                            }
                        }
                    }
                }
                
            }
        }
    }
    pub fn load_chess_model(&mut self, app: &mut App, id: ChessModel, x: usize, y: usize){
        app.data.model_instances.push(ModelInstance{ 
            model_index: id as usize,  
            position: glm::vec3(x as f32 * 2.0, y as f32 * 2.0, 0.0),
            rotate_rad: glm::radians(&glm::vec1(90.0))[0],
            rotate_vec: glm::vec3(0.0, 0.0, 1.0),
        });
    }
    pub fn load_board(&mut self, app: &mut App) {
        app.data.model_instances.push(ModelInstance{ 
            model_index: ChessModel::board as usize,  
            position: glm::vec3(7.0, 7.0, 0.0),
            rotate_rad: glm::radians(&glm::vec1(90.0))[0],
            rotate_vec: glm::vec3(0.0, 0.0, 1.0),
        });
    }
    pub fn load_selected(&mut self, app: &mut App, x: usize, y: usize){
        app.data.model_instances.push(ModelInstance{ 
            model_index: ChessModel::selected as usize,  
            position: glm::vec3(x as f32 * 2.0, y as f32 * 2.0, 0.001),
            rotate_rad: glm::radians(&glm::vec1(90.0))[0],
            rotate_vec: glm::vec3(0.0, 0.0, 1.0),
        });
    }
    pub fn get_selected_square(&mut self, app: &mut App) -> Option<(usize, usize)> {
        let pos = 
        line_intersect_plane(app.data.camera.position, 
            self.look_vec, 
            [0.0, 0.0, 1.0, 0.0]);

        // Draw selected square
        if pos.is_some() {
            let pos = pos.unwrap();
            //println!("{}",  pos);
            let board_x = (pos[0]/2.0) as i32;
            let board_y = (pos[1]/2.0) as i32;
            if board_x >= 0 && board_x <= 7 && board_y >= 0 && board_y <= 7 {
                //self.load_selected(app, board_x as usize, board_y as usize);
                return Some((board_x as usize, board_y as usize));
            }
        }
        return None;
    }
}

impl GameLoop for MyGameLoop {
    fn create(&mut self, app: &mut App) {
        self.look_vec = glm::vec3(6.0, 0.0, 2.0);
        self.angle_x = 0.0;
        self.angle_y = 0.0;
        self.grab = true;
        app.data.camera.position[0] = 6.0;
        app.data.camera.position[2] = 2.0;
        self.selected_x = 8;
        self.selected_y = 8;

        // Load some models on screen
        for model_index in 0..4 {
            let y = (((model_index % 2) as f32) * 2.5) - 1.25;
            let z = (((model_index / 2) as f32) * -2.0) + 1.0;
        }
    }
    fn update(&mut self, app: &mut App) {
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
        let pos = 
        line_intersect_plane(app.data.camera.position, 
            self.look_vec, 
            [0.0, 0.0, 1.0, 0.0]);

        // Draw selected square
        let pos = self.get_selected_square(app);
        if pos.is_some() {
            let (x, y) = pos.unwrap();
            self.load_selected(app, x as usize, y as usize);
        }
        if self.selected_x <= 7 && self.selected_y <= 7 {
            self.load_selected(app, self.selected_x as usize, self.selected_y as usize);
        }
    }
    fn handle_event(&mut self, app: &mut App, event: &Event<()>, window: &winit::window::Window) {
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

            Event::WindowEvent { event: WindowEvent::MouseInput { button, state, .. }, .. } => {
                if *state == ElementState::Pressed {
                    match *button {
                        winit::event::MouseButton::Left => {
                            let pos = self.get_selected_square(app);
                            if pos.is_some() {
                                let (x, y) = pos.unwrap();
                                if self.selected_x <= 7 && self.selected_y <= 7 {
                                    self.chess_engine.move_piece(
                                        BoardMove::new(self.selected_x as u8, 
                                                self.selected_y as u8, 
                                                x as u8, 
                                                y as u8), 
                                            true, None);
                                    self.selected_x = 8;
                                    self.selected_y = 8;
                                }
                                else {
                                    self.selected_x = x;
                                    self.selected_y = y;
                                }
                                
                            }
                            else {
                                self.selected_x = 8;
                                self.selected_y = 8;
                            }
                        },
                        _ => { }
                    }
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
    selected,
}

fn main() -> Result<()> {
    let mut a = MyGameLoop::default();
    let mtl_file = "./resources/chess/chess_set.mtl".to_string();

    app::main(&mut a, vec!(
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/board.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_pawn.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_queen.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_knight.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_rook.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_king.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/white_bishop.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_pawn.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_queen.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_knight.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_rook.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_king.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/black_bishop.obj".to_string(), mtl_file.clone())),
        ModelLoader::ModelLoaderObjFile(ModelLoaderObjFile::new("./resources/chess/selected.obj".to_string(), mtl_file.clone()))
    )
    )?;
    return Ok(());
}