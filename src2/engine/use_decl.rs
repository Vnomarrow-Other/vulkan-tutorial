pub use std::collections::{HashMap, HashSet};
pub use std::ffi::CStr;
pub use std::fs::File;
pub use std::hash::{Hash, Hasher};
pub use std::io::BufReader;
pub use std::mem::size_of;
pub use std::os::raw::c_void;
pub use std::ptr::copy_nonoverlapping as memcpy;
pub use std::time::Instant;

pub use anyhow::{anyhow, Result};
pub use log::*;
pub use nalgebra_glm as glm;
pub use thiserror::Error;
pub use vulkanalia::loader::{LibloadingLoader, LIBRARY};
pub use vulkanalia::prelude::v1_0::*;
pub use vulkanalia::window as vk_window;
pub use winit::dpi::LogicalSize;
pub use winit::event::{Event, WindowEvent};
pub use winit::event_loop::{ControlFlow, EventLoop};
pub use winit::window::{Window, WindowBuilder};

pub use vulkanalia::vk::ExtDebugUtilsExtension;
pub use vulkanalia::vk::KhrSurfaceExtension;
pub use vulkanalia::vk::KhrSwapchainExtension;
