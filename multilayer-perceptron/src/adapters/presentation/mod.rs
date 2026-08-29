pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use native::*;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;
