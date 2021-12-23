use crate::engine::use_decl::*;

pub struct Model {
    vertex_buffer: u32,
    index_buffer: u32
}

pub fn load_model() -> Option<Model> {
    return Some(Model {
        vertex_buffer: 0,
        index_buffer: 0
    })
}

pub struct ModelTracker {
    models: Vec<Model>,
}

pub struct ModelDrawer {
    draw_calls: Vec<(u32, [u8;3])> // Model index and drawing position
}

impl ModelDrawer {
    pub fn draw(&self, models: ModelTracker) {
        // Submit each draw call to separate secondary command buffers and execute them
    }
}