//! Optimization algorithms for training neural networks
//!
//! This module provides different optimizers for updating model parameters
//! during training. Currently supports SGD (Stochastic Gradient Descent),
//! SGD with Momentum, full-batch Gradient Descent, Adam, and RMSProp.

use crate::core::{Result, Tensor};
use ndarray::Zip;
use std::collections::HashMap;

/// Trait for optimizers that update model parameters
pub trait Optimizer {
  /// Update parameters using computed gradients
  ///
  /// # Arguments
  /// * `parameters` - Mutable reference to parameter tensors
  ///
  /// # Returns
  /// Result indicating success or failure
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;

  /// Zero out all gradients in the parameters
  ///
  /// # Arguments
  /// * `parameters` - Mutable reference to parameter tensors
  fn zero_grad(&self, parameters: &mut [&mut Tensor]);

  /// Get the learning rate
  fn learning_rate(&self) -> f64;

  /// Set the learning rate
  fn set_learning_rate(&mut self, lr: f64);

  /// Whether this optimizer expects gradients computed on the full dataset.
  fn requires_full_batch(&self) -> bool {
    false
  }

  /// Identifier for display/debugging.
  fn name(&self) -> &'static str;
}

/// Stochastic Gradient Descent optimizer
///
/// SGD updates parameters using the following rule:
/// θ = θ - lr * ∇θ
///
/// Where:
/// - θ are the parameters
/// - lr is the learning rate
/// - ∇θ are the gradients
#[derive(Debug, Clone)]
pub struct SGD {
  /// Learning rate
  learning_rate: f64,
}

impl SGD {
  /// Create a new SGD optimizer
  ///
  /// # Arguments
  /// * `learning_rate` - The learning rate for parameter updates
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let optimizer = SGD::new(0.01);
  /// assert_eq!(optimizer.learning_rate(), 0.01);
  /// ```
  pub fn new(learning_rate: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    Self { learning_rate }
  }
}

impl Optimizer for SGD {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for param in parameters.iter_mut() {
      if let Some(gradient) = param.grad() {
        Zip::from(&mut param.data)
          .and(&gradient)
          .for_each(|param_elem, &grad_elem| {
            *param_elem -= self.learning_rate * grad_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "SGD"
  }
}

/// SGD with momentum optimizer
///
/// SGD with momentum uses exponentially weighted averages of gradients
/// to accelerate learning and dampen oscillations:
///
/// v_t = β * v_{t-1} + (1-β) * ∇θ_t
/// θ_t = θ_{t-1} - lr * v_t
///
/// Where:
/// - v_t is the velocity (exponentially weighted average)
/// - β is the momentum coefficient (typically 0.9)
/// - lr is the learning rate
/// - ∇θ_t are the current gradients
#[derive(Debug)]
pub struct SGDMomentum {
  /// Learning rate
  learning_rate: f64,
  /// Momentum coefficient (typically 0.9)
  momentum: f64,
  /// Velocity terms for each parameter
  velocities: HashMap<usize, Tensor>,
}

impl SGDMomentum {
  /// Create a new SGD with momentum optimizer
  ///
  /// # Arguments
  /// * `learning_rate` - The learning rate for parameter updates
  /// * `momentum` - The momentum coefficient (typically 0.9)
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let optimizer = SGDMomentum::new(0.01, 0.9);
  /// assert_eq!(optimizer.learning_rate(), 0.01);
  /// ```
  pub fn new(learning_rate: f64, momentum: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    assert!(
      (0.0..1.0).contains(&momentum),
      "Momentum must be between 0 and 1"
    );

    Self {
      learning_rate,
      momentum,
      velocities: HashMap::new(),
    }
  }

  /// Get the momentum coefficient
  pub fn momentum(&self) -> f64 {
    self.momentum
  }

  /// Set the momentum coefficient
  pub fn set_momentum(&mut self, momentum: f64) {
    assert!(
      (0.0..1.0).contains(&momentum),
      "Momentum must be between 0 and 1"
    );
    self.momentum = momentum;
  }
}

impl Clone for SGDMomentum {
  fn clone(&self) -> Self {
    Self {
      learning_rate: self.learning_rate,
      momentum: self.momentum,
      velocities: HashMap::new(), // Reset velocities on clone
    }
  }
}

impl Optimizer for SGDMomentum {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for (i, param) in parameters.iter_mut().enumerate() {
      if let Some(gradient_data) = param.grad() {
        // Get or create velocity for this parameter
        let velocity = self
          .velocities
          .entry(i)
          .or_insert_with(|| Tensor::zeros(param.shape().0, param.shape().1));

        // Update velocity: v_t = β * v_{t-1} + (1-β) * ∇θ_t
        Zip::from(&mut velocity.data)
          .and(&gradient_data)
          .for_each(|vel_elem, &grad_elem| {
            *vel_elem = self.momentum * *vel_elem + (1.0 - self.momentum) * grad_elem;
          });

        // Update parameters: θ_t = θ_{t-1} - lr * v_t
        Zip::from(&mut param.data)
          .and(&velocity.data)
          .for_each(|param_elem, &vel_elem| {
            *param_elem -= self.learning_rate * vel_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "SGD with Momentum"
  }
}

/// Full-batch Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct GradientDescent {
  learning_rate: f64,
}

impl GradientDescent {
  pub fn new(learning_rate: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    Self { learning_rate }
  }
}

impl Optimizer for GradientDescent {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for param in parameters.iter_mut() {
      if let Some(gradient) = param.grad() {
        Zip::from(&mut param.data)
          .and(&gradient)
          .for_each(|param_elem, &grad_elem| {
            *param_elem -= self.learning_rate * grad_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn requires_full_batch(&self) -> bool {
    true
  }

  fn name(&self) -> &'static str {
    "Gradient Descent"
  }
}

/// Adam optimizer
///
/// Adam combines momentum and RMSProp ideas with bias correction:
///
/// m_t = β₁ m_{t-1} + (1-β₁) g_t
/// v_t = β₂ v_{t-1} + (1-β₂) g_t²
/// θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
#[derive(Debug)]
pub struct Adam {
  learning_rate: f64,
  beta1: f64,
  beta2: f64,
  epsilon: f64,
  timestep: usize,
  first_moment: HashMap<usize, Tensor>,
  second_moment: HashMap<usize, Tensor>,
}

impl Adam {
  /// Create Adam optimizer with default β and ε values
  pub fn new(learning_rate: f64) -> Self {
    Self::with_hyperparameters(learning_rate, 0.9, 0.999, 1e-8)
  }

  /// Create Adam optimizer with explicit hyperparameters
  pub fn with_hyperparameters(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    assert!((0.0..1.0).contains(&beta1), "beta1 must be in (0, 1)");
    assert!((0.0..1.0).contains(&beta2), "beta2 must be in (0, 1)");
    assert!(epsilon > 0.0, "epsilon must be positive");

    Self {
      learning_rate,
      beta1,
      beta2,
      epsilon,
      timestep: 0,
      first_moment: HashMap::new(),
      second_moment: HashMap::new(),
    }
  }

  /// Get β₁ parameter
  pub fn beta1(&self) -> f64 {
    self.beta1
  }

  /// Get β₂ parameter
  pub fn beta2(&self) -> f64 {
    self.beta2
  }

  /// Get ε parameter
  pub fn epsilon(&self) -> f64 {
    self.epsilon
  }

  /// Set β₁ parameter
  pub fn set_beta1(&mut self, beta1: f64) {
    assert!((0.0..1.0).contains(&beta1), "beta1 must be in (0, 1)");
    self.beta1 = beta1;
  }

  /// Set β₂ parameter
  pub fn set_beta2(&mut self, beta2: f64) {
    assert!((0.0..1.0).contains(&beta2), "beta2 must be in (0, 1)");
    self.beta2 = beta2;
  }

  /// Set ε parameter
  pub fn set_epsilon(&mut self, epsilon: f64) {
    assert!(epsilon > 0.0, "epsilon must be positive");
    self.epsilon = epsilon;
  }

  fn get_or_init_state<'a>(
    map: &'a mut HashMap<usize, Tensor>,
    index: usize,
    param: &Tensor,
  ) -> &'a mut Tensor {
    map
      .entry(index)
      .or_insert_with(|| Tensor::zeros(param.shape().0, param.shape().1))
  }
}

impl Clone for Adam {
  fn clone(&self) -> Self {
    Self {
      learning_rate: self.learning_rate,
      beta1: self.beta1,
      beta2: self.beta2,
      epsilon: self.epsilon,
      timestep: 0,
      first_moment: HashMap::new(),
      second_moment: HashMap::new(),
    }
  }
}

impl Optimizer for Adam {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    self.timestep += 1;
    let bias_correction1 = 1.0 - self.beta1.powf(self.timestep as f64);
    let bias_correction2 = 1.0 - self.beta2.powf(self.timestep as f64);

    for (i, param) in parameters.iter_mut().enumerate() {
      if let Some(gradient) = param.grad() {
        let m = Self::get_or_init_state(&mut self.first_moment, i, param);
        let v = Self::get_or_init_state(&mut self.second_moment, i, param);

        Zip::from(&mut m.data)
          .and(&mut v.data)
          .and(&gradient)
          .and(&mut param.data)
          .for_each(|m_elem, v_elem, &grad_elem, param_elem| {
            *m_elem = self.beta1 * *m_elem + (1.0 - self.beta1) * grad_elem;
            *v_elem = self.beta2 * *v_elem + (1.0 - self.beta2) * grad_elem * grad_elem;

            let m_hat = *m_elem / bias_correction1;
            let v_hat = *v_elem / bias_correction2;
            *param_elem -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
          });
      }
    }

    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "Adam"
  }
}

/// RMSProp optimizer
///
/// RMSProp maintains an exponential moving average of squared gradients
/// to adapt the learning rate per-parameter.
#[derive(Debug)]
pub struct RMSProp {
  learning_rate: f64,
  beta: f64,
  epsilon: f64,
  cache: HashMap<usize, Tensor>,
}

impl RMSProp {
  /// Create RMSProp optimizer with default β and ε values
  pub fn new(learning_rate: f64) -> Self {
    Self::with_hyperparameters(learning_rate, 0.9, 1e-8)
  }

  /// Create RMSProp optimizer with explicit hyperparameters
  pub fn with_hyperparameters(learning_rate: f64, beta: f64, epsilon: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    assert!((0.0..1.0).contains(&beta), "beta must be in (0, 1)");
    assert!(epsilon > 0.0, "epsilon must be positive");

    Self {
      learning_rate,
      beta,
      epsilon,
      cache: HashMap::new(),
    }
  }

  /// Get β parameter
  pub fn beta(&self) -> f64 {
    self.beta
  }

  /// Get ε parameter
  pub fn epsilon(&self) -> f64 {
    self.epsilon
  }

  /// Set β parameter
  pub fn set_beta(&mut self, beta: f64) {
    assert!((0.0..1.0).contains(&beta), "beta must be in (0, 1)");
    self.beta = beta;
  }

  /// Set ε parameter
  pub fn set_epsilon(&mut self, epsilon: f64) {
    assert!(epsilon > 0.0, "epsilon must be positive");
    self.epsilon = epsilon;
  }
}

impl Clone for RMSProp {
  fn clone(&self) -> Self {
    Self {
      learning_rate: self.learning_rate,
      beta: self.beta,
      epsilon: self.epsilon,
      cache: HashMap::new(),
    }
  }
}

impl Optimizer for RMSProp {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for (i, param) in parameters.iter_mut().enumerate() {
      if let Some(gradient) = param.grad() {
        let cache = self
          .cache
          .entry(i)
          .or_insert_with(|| Tensor::zeros(param.shape().0, param.shape().1));

        Zip::from(&mut cache.data)
          .and(&gradient)
          .and(&mut param.data)
          .for_each(|cache_elem, &grad_elem, param_elem| {
            *cache_elem = self.beta * *cache_elem + (1.0 - self.beta) * grad_elem * grad_elem;
            *param_elem -= self.learning_rate * grad_elem / (cache_elem.sqrt() + self.epsilon);
          });
      }
    }

    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "RMSProp"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::Tensor;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_sgd_creation() {
    let optimizer = SGD::new(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
    assert_eq!(optimizer.name(), "SGD");
  }

  #[test]
  #[should_panic(expected = "Learning rate must be positive")]
  fn test_sgd_negative_learning_rate() {
    SGD::new(-0.01);
  }

  #[test]
  fn test_sgd_set_learning_rate() {
    let mut optimizer = SGD::new(0.01);
    optimizer.set_learning_rate(0.1);
    assert_eq!(optimizer.learning_rate(), 0.1);
  }

  #[test]
  fn test_sgd_zero_grad() {
    let optimizer = SGD::new(0.01);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set some fake gradient
    param.set_gradient(Some(Tensor::ones(2, 2)));
    assert!(param.gradient().is_some());

    optimizer.zero_grad(&mut [&mut param]);
    // After zero_grad, gradient should still exist but be zero
    if let Some(grad) = param.grad() {
      for elem in grad.iter() {
        assert_eq!(*elem, 0.0);
      }
    } else {
      panic!("Gradient should still exist after zero_grad");
    }
  }

  #[test]
  fn test_sgd_step() {
    let mut optimizer = SGD::new(0.1);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set gradient to ones, so update should be -0.1 * 1.0 = -0.1
    param.set_gradient(Some(Tensor::ones(2, 2)));

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // θ = θ - lr * ∇θ = 1.0 - 0.1 * 1.0 = 0.9
    assert_abs_diff_eq!(param.data[[0, 0]], original_value - 0.1, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_step_no_gradient() {
    let mut optimizer = SGD::new(0.1);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // No gradient, so parameter should not change
    assert_abs_diff_eq!(param.data[[0, 0]], original_value, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_momentum_creation() {
    let optimizer = SGDMomentum::new(0.01, 0.9);
    assert_eq!(optimizer.learning_rate(), 0.01);
    assert_eq!(optimizer.name(), "SGD with Momentum");
    assert_eq!(optimizer.momentum(), 0.9);
  }

  #[test]
  #[should_panic(expected = "Momentum must be between 0 and 1")]
  fn test_sgd_momentum_invalid_momentum() {
    SGDMomentum::new(0.01, 1.5);
  }

  #[test]
  fn test_sgd_momentum_set_momentum() {
    let mut optimizer = SGDMomentum::new(0.01, 0.9);
    optimizer.set_momentum(0.95);
    assert_eq!(optimizer.momentum(), 0.95);
  }

  #[test]
  fn test_sgd_momentum_step() {
    let mut optimizer = SGDMomentum::new(0.1, 0.9);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set gradient to ones
    param.set_gradient(Some(Tensor::ones(2, 2)));

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // First step: v_0 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    // θ_1 = 1.0 - 0.1 * 0.1 = 0.99
    assert_abs_diff_eq!(param.data[[0, 0]], original_value - 0.01, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_momentum_multiple_steps() {
    let mut optimizer = SGDMomentum::new(0.1, 0.9);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // First step
    param.set_gradient(Some(Tensor::ones(2, 2)));
    let value_after_first = {
      optimizer.step(&mut [&mut param]).unwrap();
      param.data[[0, 0]]
    };

    // Second step with same gradient
    param.set_gradient(Some(Tensor::ones(2, 2)));
    optimizer.step(&mut [&mut param]).unwrap();
    let value_after_second = param.data[[0, 0]];

    // With momentum, second step should have larger update than first
    let first_update = 1.0 - value_after_first;
    let second_update = value_after_first - value_after_second;
    assert!(second_update > first_update);
  }

  #[test]
  fn test_sgd_momentum_clone() {
    let optimizer = SGDMomentum::new(0.01, 0.9);
    let cloned = optimizer.clone();

    assert_eq!(optimizer.learning_rate(), cloned.learning_rate());
    assert_eq!(optimizer.momentum(), cloned.momentum());
    // Velocities should be reset in clone
    assert!(cloned.velocities.is_empty());
  }

  #[test]
  fn test_gradient_descent_name() {
    let optimizer = GradientDescent::new(0.1);
    assert_eq!(optimizer.name(), "Gradient Descent");
    assert!(optimizer.requires_full_batch());
  }

  #[test]
  fn test_adam_creation() {
    let optimizer = Adam::new(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);
    assert_eq!(optimizer.name(), "Adam");
    assert_abs_diff_eq!(optimizer.beta1(), 0.9, epsilon = 1e-12);
    assert_abs_diff_eq!(optimizer.beta2(), 0.999, epsilon = 1e-12);
  }

  #[test]
  #[should_panic(expected = "beta1 must be in (0, 1)")]
  fn test_adam_invalid_beta1() {
    Adam::with_hyperparameters(0.001, 1.2, 0.999, 1e-8);
  }

  #[test]
  fn test_adam_step() {
    let mut optimizer = Adam::new(0.001);
    let mut param = Tensor::ones(1, 1);
    param.set_requires_grad(true);
    param.set_gradient(Some(Tensor::ones(1, 1)));

    optimizer.step(&mut [&mut param]).unwrap();

    // First Adam step with unit gradient should reduce parameter below 1.0
    assert!(param.data[[0, 0]] < 1.0);
    assert!(param.data[[0, 0]] > 0.998); // Should not overshoot with default lr
  }

  #[test]
  fn test_adam_clone_resets_state() {
    let mut optimizer = Adam::new(0.001);
    let mut param = Tensor::ones(1, 1);
    param.set_requires_grad(true);
    param.set_gradient(Some(Tensor::ones(1, 1)));
    optimizer.step(&mut [&mut param]).unwrap();

    let cloned = optimizer.clone();
    assert_eq!(cloned.learning_rate(), optimizer.learning_rate());
    assert_eq!(cloned.beta1(), optimizer.beta1());
    assert!(cloned.first_moment.is_empty());
    assert!(cloned.second_moment.is_empty());
    assert_eq!(cloned.timestep, 0);
  }

  #[test]
  fn test_rmsprop_creation() {
    let optimizer = RMSProp::new(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
    assert_eq!(optimizer.name(), "RMSProp");
    assert_abs_diff_eq!(optimizer.beta(), 0.9, epsilon = 1e-12);
  }

  #[test]
  #[should_panic(expected = "beta must be in (0, 1)")]
  fn test_rmsprop_invalid_beta() {
    RMSProp::with_hyperparameters(0.01, 1.5, 1e-8);
  }

  #[test]
  fn test_rmsprop_step() {
    let mut optimizer = RMSProp::new(0.01);
    let mut param = Tensor::ones(1, 1);
    param.set_requires_grad(true);
    param.set_gradient(Some(Tensor::ones(1, 1)));

    optimizer.step(&mut [&mut param]).unwrap();

    // RMSProp should move the parameter in negative gradient direction
    assert!(param.data[[0, 0]] < 1.0);
    assert!(param.data[[0, 0]] > 0.95);
  }
}
