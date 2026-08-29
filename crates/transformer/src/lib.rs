//! Hand-rolled educational encoder-decoder transformer — Rust CPU.
//!
//! Manual forward/backward (no autograd framework), Adam optimizer, and
//! attention-probability capture for playground visualisation.

pub mod config;
pub mod engine;
pub mod model;
pub mod snapshot;
pub mod tasks;
pub mod tensor;

#[cfg(feature = "wasm")]
mod wasm;

pub use config::TransformerConfig;
pub use engine::PlaygroundEngine;
pub use tasks::TransformerTask;
