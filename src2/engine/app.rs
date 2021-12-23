use crate::engine::use_decl::*;

pub struct App {
    pub resized: bool,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        Ok(Self {
            resized: true
        })
    }
    pub unsafe fn render(&self) {

    }
    pub unsafe fn update(&self, window: &Window) {

    }
    pub unsafe fn destroy(&self) {
        
    }
}

pub struct AppData {

}