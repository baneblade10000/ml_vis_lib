//! # Multilayer Perceptron
//!
//! A from-scratch implementation of a multilayer perceptron with automatic differentiation
//! in Rust, designed for both native and WebAssembly targets.
//!
//! ## Features
//!
//! - **Automatic Differentiation**: Computation graph with reverse-mode autodiff
//! - **Neural Network Layers**: Dense layers with various activation functions
//! - **WebAssembly Support**: Deploy to browsers for interactive demos
//! - **Pure Rust**: No external ML libraries, everything built from scratch
//!
//! ## Quick Start
//!
//! ```rust
//! use multilayer_perceptron::prelude::*;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! // Create a tensor
//! let tensor = Tensor::ones(2, 3);
//! println!("Tensor shape: {:?}", tensor.shape());
//!
//! // Create a neural network with PyTorch-like API
//! let graph = Rc::new(RefCell::new(ComputationGraph::new()));
//! let mut model = Sequential::new()
//!     .relu_layer(784, 128)
//!     .relu_layer(128, 64)
//!     .softmax_layer(64, 10)
//!     .with_graph(graph);
//!
//! // Forward pass
//! let input = Tensor::random(1, 784);
//! let output = model.forward(input).unwrap();
//! println!("Output shape: {:?}", output.shape());
//! ```

#![warn(clippy::all)]

pub mod adapters;
pub mod app;
pub mod core;
pub mod domain;
pub mod usecase;

// WebAssembly bindings (only included when targeting wasm32)
#[cfg(target_arch = "wasm32")]
pub use crate::adapters::presentation::wasm;

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and functions
/// for convenient access when using the multilayer perceptron library.
pub mod prelude {
  pub use crate::adapters::data::{
    BostonHousingRepository, BreastCancerRepository, IrisRepository,
  };
  #[cfg(not(target_arch = "wasm32"))]
  pub use crate::adapters::presentation::NativeVisualizer;
  pub use crate::core::{
    ComputationGraph, EdgeId, GraphEdge, GraphNode, NodeId, OpBuilder, OpNode, Result, Tensor,
    TensorError,
  };
  pub use crate::domain::models::{
    Activation, DenseLayer, Layer, LayerInfo, ModelSummary, Sequential, WeightInit, MLP,
  };
  pub use crate::domain::services::loss::{
    BinaryCrossEntropy, CrossEntropy, Loss, MeanSquaredError, RegularizableModel,
    RegularizationConfig, RegularizedLoss,
  };
  pub use crate::domain::services::metrics::{
    Accuracy, BinaryClassificationMetrics, CategoricalAccuracy, F1Score, MeanSquaredErrorMetric,
    Metric, Precision, Recall,
  };
  pub use crate::domain::services::optimizer::{
    Adam, GradientDescent, Optimizer, RMSProp, SGDMomentum, SGD,
  };
  pub use crate::domain::types::{
    DataConfig, DataLoader, Dataset, FeatureStats, PreprocessConfig, TaskKind,
  };
  pub use crate::domain::{DataRepository, ModelRepository};
  pub use crate::usecase::train_mlp::{
    EpochHistory, TrainMLPUsecase, TrainRequest, TrainResponse, Trainer, TrainingConfig,
    TrainingHistory,
  };
}

// Temporary placeholder function for CI setup
/// Add two numbers together.
///
/// This is a temporary function for CI/CD setup and will be removed
/// once the actual neural network implementation is in place.
///
/// # Examples
///
/// ```
/// use multilayer_perceptron::add;
/// assert_eq!(add(2, 2), 4);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
  left + right
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add() {
    assert_eq!(add(2, 2), 4);
    assert_eq!(add(0, 5), 5);
    assert_eq!(add(10, 15), 25);
  }

  #[test]
  fn test_add_edge_cases() {
    assert_eq!(add(0, 0), 0);
    assert_eq!(add(u64::MAX - 1, 1), u64::MAX);
  }
}
