

pub struct Texture {

}

pub fn load_texture() -> Option<Texture> {
    Some (
        Texture{

        }
    )
}

pub struct TextureTracker {
    models: Vec<Texture>,
}

pub struct TextureDrawer {
    draw_calls: Vec<(u32, [u8;2])> // Texture index and drawing position
    // TODO, include texture size and rotations
}

impl TextureDrawer {
    pub fn draw(&self, models: TextureTracker) {
        // Submit each draw call to separate secondary command buffers and execute them
    }
}