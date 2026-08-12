//! Hand-rolled educational CNN — Rust CPU + optional WebGPU device.

pub mod config;
pub mod conv_gemm;
pub mod data;
pub mod engine;
pub mod gpu;
pub mod layers;
pub mod simd_f32;
pub mod tensor;

#[cfg(feature = "wasm")]
mod wasm;

pub use config::{CnnConfig, IMAGE_SIZE, SIGNAL_LENGTH};
pub use engine::PlaygroundEngine;
