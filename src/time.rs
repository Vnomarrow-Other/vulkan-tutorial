use std::time::Instant;

pub struct TimeTracker {
    start: Instant
}

impl TimeTracker {
    pub fn new() -> Self {
        Self {
            start: Instant::now()
        }
    }
    pub fn print_elapsed(&self, msg: &str) {
        println!("{}: {}", msg, self.start.elapsed().as_secs_f64());
    }
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

#[derive(Clone, Debug)]
pub struct FPSTracker {
    pub start: Instant,
    last_secs_update: f32,
    frame_renders: usize,
    pub print_fps: bool
}

impl FPSTracker {
    pub fn new(print_fps: bool) -> Self {
        Self {
            start: Instant::now(),
            last_secs_update: 0.0,
            frame_renders: 0,
            print_fps
        }
    }
    pub fn record_frame(&mut self) -> Option<usize> {
        self.frame_renders += 1;
        if self.start.elapsed().as_secs_f32() - self.last_secs_update > 1.0 {
            let fps = self.frame_renders;
            if self.print_fps {
                println!("fps: {}", fps);
            }
            self.last_secs_update += 1.0;
            self.frame_renders = 0;
            return Some(fps);
        }
        return None;
    }
}