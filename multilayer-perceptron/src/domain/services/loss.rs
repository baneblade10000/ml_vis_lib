//! Loss functions for training neural networks
//!
//! This module provides different loss functions for computing the error
//! between model predictions and true labels during training.

use crate::core::{Result, Tensor};

/// Trait for loss functions
pub trait Loss {
  /// Compute the loss between predictions and targets
  ///
  /// # Arguments
  /// * `predictions` - Model predictions tensor
  /// * `targets` - True target labels tensor
  ///
  /// # Returns
  /// Scalar loss value
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

  /// Compute the gradient of the loss with respect to predictions
  ///
  /// # Arguments
  /// * `predictions` - Model predictions tensor
  /// * `targets` - True target labels tensor
  ///
  /// # Returns
  /// Gradient tensor with same shape as predictions
  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

  /// Get the name of the loss function
  fn name(&self) -> &'static str;
}

/// Binary Cross Entropy Loss
///
/// Binary Cross Entropy (BCE) loss is used for binary classification tasks.
/// The loss is computed as:
///
/// BCE(y, ŷ) = -(1/N) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
///
/// Where:
/// - y is the true binary label (0 or 1)
/// - ŷ is the predicted probability (between 0 and 1)
/// - N is the number of samples
///
/// The gradient with respect to predictions is:
/// ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / N
#[derive(Debug, Clone)]
pub struct BinaryCrossEntropy {
  /// Small epsilon value to prevent log(0)
  epsilon: f64,
}

impl BinaryCrossEntropy {
  /// Create a new Binary Cross Entropy loss
  ///
  /// # Arguments
  /// * `epsilon` - Small value to add for numerical stability (default: 1e-8)
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = BinaryCrossEntropy::new();
  /// ```
  pub fn new() -> Self {
    Self { epsilon: 1e-8 }
  }

  /// Create a new Binary Cross Entropy loss with custom epsilon
  ///
  /// # Arguments
  /// * `epsilon` - Small value to add for numerical stability
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = BinaryCrossEntropy::with_epsilon(1e-10);
  /// ```
  pub fn with_epsilon(epsilon: f64) -> Self {
    assert!(epsilon > 0.0, "Epsilon must be positive");
    Self { epsilon }
  }

  /// Get the epsilon value used for numerical stability
  pub fn epsilon(&self) -> f64 {
    self.epsilon
  }

  /// Clamp predictions to avoid log(0) and log(1-0) issues
  fn clamp_predictions(&self, predictions: &Tensor) -> Result<Tensor> {
    let min_val = self.epsilon;
    let max_val = 1.0 - self.epsilon;

    let mut clamped_data = predictions.data.clone();
    for i in 0..clamped_data.nrows() {
      for j in 0..clamped_data.ncols() {
        let val = clamped_data[[i, j]];
        if val < min_val {
          clamped_data[[i, j]] = min_val;
        } else if val > max_val {
          clamped_data[[i, j]] = max_val;
        }
      }
    }

    Tensor::from_data(clamped_data)
  }
}

impl Default for BinaryCrossEntropy {
  fn default() -> Self {
    Self::new()
  }
}

impl Loss for BinaryCrossEntropy {
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent log(0) and log(1)
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Compute 1 - predictions
    let (rows, cols) = predictions.shape();
    let one_tensor = Tensor::ones(rows, cols);
    let one_minus_preds = one_tensor.sub(&clamped_preds)?;

    // Compute 1 - targets
    let one_minus_targets = one_tensor.sub(targets)?;

    // Log terms: targets * log(predictions) + (1-targets) * log(1-predictions)
    let log_preds = clamped_preds.log()?;
    let log_one_minus_preds = one_minus_preds.log()?;

    let term1 = targets.mul(&log_preds)?;
    let term2 = one_minus_targets.mul(&log_one_minus_preds)?;

    let log_likelihood = term1.add(&term2)?;

    // Take negative mean: -(1/N) * sum(log_likelihood)
    let mean_log_likelihood = log_likelihood.mean()?;
    mean_log_likelihood.mul_scalar(-1.0)
  }

  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent division by 0
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Compute 1 - predictions and 1 - targets
    let (rows, cols) = predictions.shape();
    let one_tensor = Tensor::ones(rows, cols);
    let one_minus_preds = one_tensor.sub(&clamped_preds)?;
    let one_minus_targets = one_tensor.sub(targets)?;

    // Gradient: -(targets/predictions - (1-targets)/(1-predictions)) / batch_size
    let term1 = targets.div(&clamped_preds)?;
    let term2 = one_minus_targets.div(&one_minus_preds)?;
    let gradient = term1.sub(&term2)?;

    let batch_size = predictions.shape().0 as f64;
    gradient.mul_scalar(-1.0 / batch_size)
  }

  fn name(&self) -> &'static str {
    "BinaryCrossEntropy"
  }
}

/// Cross Entropy Loss
///
/// Cross Entropy loss is used for multi-class classification tasks.
/// The loss is computed as:
///
/// CE(y, ŷ) = -(1/N) * Σ Σ y_i * log(ŷ_i)
///
/// Where:
/// - y is the true one-hot encoded labels
/// - ŷ is the predicted probabilities (softmax output)
/// - N is the number of samples
///
/// The gradient with respect to predictions is:
/// ∂CE/∂ŷ = -(y - ŷ) / N
#[derive(Debug, Clone)]
pub struct CrossEntropy {
  /// Small epsilon value to prevent log(0)
  epsilon: f64,
}

impl CrossEntropy {
  /// Create a new Cross Entropy loss
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = CrossEntropy::new();
  /// ```
  pub fn new() -> Self {
    Self { epsilon: 1e-8 }
  }

  /// Create a new Cross Entropy loss with custom epsilon
  ///
  /// # Arguments
  /// * `epsilon` - Small value to add for numerical stability
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = CrossEntropy::with_epsilon(1e-10);
  /// ```
  pub fn with_epsilon(epsilon: f64) -> Self {
    assert!(epsilon > 0.0, "Epsilon must be positive");
    Self { epsilon }
  }

  /// Get the epsilon value used for numerical stability
  pub fn epsilon(&self) -> f64 {
    self.epsilon
  }

  /// Clamp predictions to prevent log(0)
  fn clamp_predictions(&self, predictions: &Tensor) -> Result<Tensor> {
    let min_val = self.epsilon;
    let max_val = 1.0 - self.epsilon;

    let mut clamped_data = predictions.data.clone();
    for i in 0..clamped_data.nrows() {
      for j in 0..clamped_data.ncols() {
        let val = clamped_data[[i, j]];
        if val < min_val {
          clamped_data[[i, j]] = min_val;
        } else if val > max_val {
          clamped_data[[i, j]] = max_val;
        }
      }
    }

    Ok(Tensor {
      data: clamped_data,
      grad: None,
      requires_grad: predictions.requires_grad,
      graph_id: None,
      graph: predictions.graph.clone(),
    })
  }
}

impl Default for CrossEntropy {
  fn default() -> Self {
    Self::new()
  }
}

impl Loss for CrossEntropy {
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent log(0)
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Compute log(predictions)
    let log_preds = clamped_preds.log()?;

    // Compute targets * log(predictions)
    let cross_entropy = targets.mul(&log_preds)?;

    // Take negative mean: -(1/N) * sum(targets * log(predictions))
    let mean_cross_entropy = cross_entropy.mean()?;
    mean_cross_entropy.mul_scalar(-1.0)
  }

  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent division by 0
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Gradient: -(targets/predictions) / batch_size
    let gradient = targets.div(&clamped_preds)?;
    let batch_size = predictions.shape().0 as f64;
    gradient.mul_scalar(-1.0 / batch_size)
  }

  fn name(&self) -> &'static str {
    "CrossEntropy"
  }
}

/// Mean Squared Error Loss
///
/// MSE loss is computed as:
/// MSE(y, ŷ) = (1/N) * Σ(y - ŷ)²
///
/// The gradient with respect to predictions is:
/// ∂MSE/∂ŷ = -2(y - ŷ) / N
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredError;

impl MeanSquaredError {
  /// Create a new Mean Squared Error loss
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = MeanSquaredError::new();
  /// ```
  pub fn new() -> Self {
    Self
  }
}

impl Loss for MeanSquaredError {
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // (y - ŷ)²
    let diff = targets.sub(predictions)?;
    let squared_diff = diff.mul(&diff)?;

    // Mean of squared differences
    squared_diff.mean()
  }

  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // -2(y - ŷ) / N
    let total_elements = (predictions.shape().0 * predictions.shape().1) as f64;
    let diff = targets.sub(predictions)?;
    diff.mul_scalar(-2.0 / total_elements)
  }

  fn name(&self) -> &'static str {
    "MeanSquaredError"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_binary_cross_entropy_creation() {
    let loss_fn = BinaryCrossEntropy::new();
    assert_eq!(loss_fn.epsilon(), 1e-8);
    assert_eq!(loss_fn.name(), "BinaryCrossEntropy");

    let loss_fn_custom = BinaryCrossEntropy::with_epsilon(1e-10);
    assert_eq!(loss_fn_custom.epsilon(), 1e-10);
  }

  #[test]
  #[should_panic(expected = "Epsilon must be positive")]
  fn test_binary_cross_entropy_negative_epsilon() {
    BinaryCrossEntropy::with_epsilon(-1e-8);
  }

  #[test]
  fn test_binary_cross_entropy_perfect_predictions() {
    let loss_fn = BinaryCrossEntropy::new();

    // Perfect predictions (with epsilon to avoid log(0))
    let predictions = Tensor::new(vec![vec![0.99999999, 0.00000001]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Loss should be very close to 0 for perfect predictions
    assert!(loss.data[[0, 0]] < 1e-6);
  }

  #[test]
  fn test_binary_cross_entropy_worst_predictions() {
    let loss_fn = BinaryCrossEntropy::with_epsilon(1e-7);

    // Worst predictions (opposite of targets)
    let predictions = Tensor::new(vec![vec![0.0000001, 0.9999999]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Loss should be large for bad predictions
    assert!(loss.data[[0, 0]] > 10.0);
  }

  #[test]
  fn test_binary_cross_entropy_forward() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3], vec![0.9, 0.1]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0], vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Verify loss is scalar and positive
    assert_eq!(loss.shape(), (1, 1));
    assert!(loss.data[[0, 0]] >= 0.0);
  }

  #[test]
  fn test_binary_cross_entropy_backward() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3], vec![0.9, 0.1]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0], vec![1.0, 0.0]]).unwrap();

    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Gradient should have same shape as predictions
    assert_eq!(gradient.shape(), predictions.shape());

    // Check gradient signs: for target=1, pred=0.8, gradient should be negative
    // (encouraging higher prediction)
    assert!(gradient.data[[0, 0]] < 0.0);

    // For target=0, pred=0.3, gradient should be positive
    // (encouraging lower prediction)
    assert!(gradient.data[[0, 1]] > 0.0);
  }

  #[test]
  fn test_binary_cross_entropy_numerical_stability() {
    let loss_fn = BinaryCrossEntropy::new();

    // Test with extreme values that could cause log(0)
    let predictions = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();
    let targets = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();

    // Should not panic due to epsilon clamping
    let loss = loss_fn.forward(&predictions, &targets).unwrap();
    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    assert!(loss.data[[0, 0]].is_finite());
    assert!(gradient.data[[0, 0]].is_finite());
    assert!(gradient.data[[0, 1]].is_finite());
  }

  #[test]
  #[should_panic(expected = "Predictions and targets must have the same shape")]
  fn test_binary_cross_entropy_shape_mismatch() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0]]).unwrap();

    loss_fn.forward(&predictions, &targets).unwrap();
  }

  #[test]
  fn test_mean_squared_error_creation() {
    let loss_fn = MeanSquaredError::new();
    assert_eq!(loss_fn.name(), "MeanSquaredError");

    let loss_fn_default = MeanSquaredError::new();
    assert_eq!(loss_fn_default.name(), "MeanSquaredError");
  }

  #[test]
  fn test_mean_squared_error_forward() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.5, 2.5], vec![2.5, 3.5]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // MSE = mean((targets - predictions)²) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    assert_eq!(loss.shape(), (1, 1));
    assert_abs_diff_eq!(loss.data[[0, 0]], 0.25, epsilon = 1e-6);
  }

  #[test]
  fn test_mean_squared_error_backward() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.5, 2.5], vec![2.5, 3.5]]).unwrap();

    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Gradient = -2(y - ŷ) / N = -2 * [0.5, 0.5, -0.5, -0.5] / 4 = [-0.25, -0.25, 0.25, 0.25]
    assert_eq!(gradient.shape(), predictions.shape());
    assert_abs_diff_eq!(gradient.data[[0, 0]], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 1]], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[1, 1]], 0.25, epsilon = 1e-6);
  }

  #[test]
  fn test_mean_squared_error_perfect_predictions() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();
    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Perfect predictions should have zero loss and gradient
    assert_abs_diff_eq!(loss.data[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 1]], 0.0, epsilon = 1e-6);
  }

  #[test]
  #[should_panic(expected = "Predictions and targets must have the same shape")]
  fn test_mean_squared_error_shape_mismatch() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0]]).unwrap();

    loss_fn.forward(&predictions, &targets).unwrap();
  }
}

/// Configuration for regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
  /// L1 regularization strength (lambda1)
  pub l1_lambda: f64,
  /// L2 regularization strength (lambda2)
  pub l2_lambda: f64,
  /// Whether to apply regularization to bias terms
  pub apply_to_bias: bool,
}

impl RegularizationConfig {
  /// Create a new regularization configuration
  ///
  /// # Arguments
  /// * `l1_lambda` - L1 regularization strength (0.0 to disable)
  /// * `l2_lambda` - L2 regularization strength (0.0 to disable)
  /// * `apply_to_bias` - Whether to apply regularization to bias terms
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// // L2 regularization only
  /// let config = RegularizationConfig::new(0.0, 0.001, false);
  ///
  /// // Elastic Net (L1 + L2)
  /// let config = RegularizationConfig::new(0.001, 0.001, false);
  /// ```
  pub fn new(l1_lambda: f64, l2_lambda: f64, apply_to_bias: bool) -> Self {
    assert!(l1_lambda >= 0.0, "L1 lambda must be non-negative");
    assert!(l2_lambda >= 0.0, "L2 lambda must be non-negative");

    Self {
      l1_lambda,
      l2_lambda,
      apply_to_bias,
    }
  }

  /// Create L1 regularization only
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let config = RegularizationConfig::l1_only(0.001);
  /// ```
  pub fn l1_only(lambda: f64) -> Self {
    Self::new(lambda, 0.0, false)
  }

  /// Create L2 regularization only
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let config = RegularizationConfig::l2_only(0.001);
  /// ```
  pub fn l2_only(lambda: f64) -> Self {
    Self::new(0.0, lambda, false)
  }

  /// Create Elastic Net regularization (L1 + L2)
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let config = RegularizationConfig::elastic_net(0.001, 0.001);
  /// ```
  pub fn elastic_net(l1_lambda: f64, l2_lambda: f64) -> Self {
    Self::new(l1_lambda, l2_lambda, false)
  }

  /// Check if any regularization is enabled
  pub fn is_enabled(&self) -> bool {
    self.l1_lambda > 0.0 || self.l2_lambda > 0.0
  }

  /// Check if L1 regularization is enabled
  pub fn has_l1(&self) -> bool {
    self.l1_lambda > 0.0
  }

  /// Check if L2 regularization is enabled
  pub fn has_l2(&self) -> bool {
    self.l2_lambda > 0.0
  }

  /// Compute the L1 regularization penalty for the provided parameters.
  pub(crate) fn compute_l1_regularization(
    &self,
    weights: &[&Tensor],
    biases: &[&Tensor],
  ) -> Result<Tensor> {
    if !self.has_l1() {
      return Ok(Tensor::zeros(1, 1));
    }

    let mut l1_sum = 0.0;

    // L1 penalty for weights
    for weight_tensor in weights {
      for i in 0..weight_tensor.data.nrows() {
        for j in 0..weight_tensor.data.ncols() {
          l1_sum += weight_tensor.data[[i, j]].abs();
        }
      }
    }

    // L1 penalty for biases (if enabled)
    if self.apply_to_bias {
      for bias_tensor in biases {
        for i in 0..bias_tensor.data.nrows() {
          for j in 0..bias_tensor.data.ncols() {
            l1_sum += bias_tensor.data[[i, j]].abs();
          }
        }
      }
    }

    let regularization_value = self.l1_lambda * l1_sum;
    Tensor::from_scalar(regularization_value)
  }

  /// Compute the L2 regularization penalty for the provided parameters.
  pub(crate) fn compute_l2_regularization(
    &self,
    weights: &[&Tensor],
    biases: &[&Tensor],
  ) -> Result<Tensor> {
    if !self.has_l2() {
      return Ok(Tensor::zeros(1, 1));
    }

    let mut l2_sum = 0.0;

    // L2 penalty for weights
    for weight_tensor in weights {
      for i in 0..weight_tensor.data.nrows() {
        for j in 0..weight_tensor.data.ncols() {
          let w = weight_tensor.data[[i, j]];
          l2_sum += w * w;
        }
      }
    }

    // L2 penalty for biases (if enabled)
    if self.apply_to_bias {
      for bias_tensor in biases {
        for i in 0..bias_tensor.data.nrows() {
          for j in 0..bias_tensor.data.ncols() {
            let b = bias_tensor.data[[i, j]];
            l2_sum += b * b;
          }
        }
      }
    }

    let regularization_value = self.l2_lambda * l2_sum;
    Tensor::from_scalar(regularization_value)
  }

  /// Compute the L1 regularization gradients for the provided parameters.
  pub(crate) fn compute_l1_gradient(
    &self,
    weights: &[&Tensor],
    biases: &[&Tensor],
  ) -> Vec<(Tensor, Tensor)> {
    if !self.has_l1() {
      return weights
        .iter()
        .zip(biases.iter())
        .map(|(w, b)| {
          (
            Tensor::zeros_like(w).unwrap(),
            Tensor::zeros_like(b).unwrap(),
          )
        })
        .collect();
    }

    let mut gradients = Vec::new();

    for (weight_tensor, bias_tensor) in weights.iter().zip(biases.iter()) {
      // Weight gradients
      let mut weight_grad_data = weight_tensor.data.clone();
      for i in 0..weight_grad_data.nrows() {
        for j in 0..weight_grad_data.ncols() {
          let w = weight_tensor.data[[i, j]];
          weight_grad_data[[i, j]] = self.l1_lambda * w.signum();
        }
      }
      let weight_grad = Tensor::from_data(weight_grad_data).unwrap();

      // Bias gradients
      let bias_grad = if self.apply_to_bias {
        let mut bias_grad_data = bias_tensor.data.clone();
        for i in 0..bias_grad_data.nrows() {
          for j in 0..bias_grad_data.ncols() {
            let b = bias_tensor.data[[i, j]];
            bias_grad_data[[i, j]] = self.l1_lambda * b.signum();
          }
        }
        Tensor::from_data(bias_grad_data).unwrap()
      } else {
        Tensor::zeros_like(bias_tensor).unwrap()
      };

      gradients.push((weight_grad, bias_grad));
    }

    gradients
  }

  /// Compute the L2 regularization gradients for the provided parameters.
  pub(crate) fn compute_l2_gradient(
    &self,
    weights: &[&Tensor],
    biases: &[&Tensor],
  ) -> Vec<(Tensor, Tensor)> {
    if !self.has_l2() {
      return weights
        .iter()
        .zip(biases.iter())
        .map(|(w, b)| {
          (
            Tensor::zeros_like(w).unwrap(),
            Tensor::zeros_like(b).unwrap(),
          )
        })
        .collect();
    }

    let mut gradients = Vec::new();

    for (weight_tensor, bias_tensor) in weights.iter().zip(biases.iter()) {
      // Weight gradients: 2λ₂ * w
      let weight_grad = weight_tensor.mul_scalar(2.0 * self.l2_lambda).unwrap();

      // Bias gradients
      let bias_grad = if self.apply_to_bias {
        bias_tensor.mul_scalar(2.0 * self.l2_lambda).unwrap()
      } else {
        Tensor::zeros_like(bias_tensor).unwrap()
      };

      gradients.push((weight_grad, bias_grad));
    }

    gradients
  }

  /// Compute the total regularization penalty (L1 + L2) for a model.
  pub(crate) fn compute_regularization_penalty<M: RegularizableModel>(
    &self,
    model: &M,
  ) -> Result<Tensor> {
    if !self.is_enabled() {
      return Ok(Tensor::zeros(1, 1));
    }

    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1 = self.compute_l1_regularization(&weights, &biases)?;
    let l2 = self.compute_l2_regularization(&weights, &biases)?;

    l1.add(&l2)
  }

  /// Add regularization gradients to the model's parameters.
  pub(crate) fn add_regularization_gradients<M: RegularizableModel>(
    &self,
    model: &mut M,
  ) -> Result<()> {
    if !self.is_enabled() {
      return Ok(());
    }

    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1_grads = self.compute_l1_gradient(&weights, &biases);
    let l2_grads = self.compute_l2_gradient(&weights, &biases);

    let mut combined_weight_grads = Vec::new();
    let mut combined_bias_grads = Vec::new();

    for ((l1_w, l1_b), (l2_w, l2_b)) in l1_grads.into_iter().zip(l2_grads.into_iter()) {
      let combined_w = l1_w.add(&l2_w)?;
      let combined_b = l1_b.add(&l2_b)?;
      combined_weight_grads.push(combined_w);
      combined_bias_grads.push(combined_b);
    }

    model.add_regularization_gradients(combined_weight_grads, combined_bias_grads);

    Ok(())
  }
}

impl Default for RegularizationConfig {
  fn default() -> Self {
    Self::new(0.0, 0.0, false)
  }
}

/// Regularized Loss Function
///
/// This wrapper adds L1 and L2 regularization terms to any base loss function.
/// It computes the total loss as:
///
/// L_total = L_original + R_L1(W) + R_L2(W)
///
/// Where:
/// - L_original is the base loss (e.g., BinaryCrossEntropy, MSE)
/// - R_L1(W) = λ₁ * Σ|wᵢⱼ| (L1 regularization)
/// - R_L2(W) = λ₂ * Σwᵢⱼ² (L2 regularization)
///
/// # Examples
/// ```
/// use multilayer_perceptron::prelude::*;
///
/// // L2 regularized binary cross entropy
/// let base_loss = BinaryCrossEntropy::new();
/// let config = RegularizationConfig::l2_only(0.001);
/// let loss_fn = RegularizedLoss::new(base_loss, config);
/// ```
#[derive(Debug, Clone)]
pub struct RegularizedLoss<L: Loss> {
  /// Base loss function
  base_loss: L,
  /// Regularization configuration
  config: RegularizationConfig,
}

impl<L: Loss> RegularizedLoss<L> {
  /// Create a new regularized loss function
  ///
  /// # Arguments
  /// * `base_loss` - The underlying loss function to regularize
  /// * `config` - Regularization configuration
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let base_loss = BinaryCrossEntropy::new();
  /// let config = RegularizationConfig::l2_only(0.001);
  /// let loss_fn = RegularizedLoss::new(base_loss, config);
  /// ```
  pub fn new(base_loss: L, config: RegularizationConfig) -> Self {
    Self { base_loss, config }
  }

  /// Get reference to the base loss function
  pub fn base_loss(&self) -> &L {
    &self.base_loss
  }

  /// Get reference to the regularization config
  pub fn config(&self) -> &RegularizationConfig {
    &self.config
  }
}

/// Trait for models that can provide weight parameters for regularization
pub trait RegularizableModel {
  /// Get references to weight tensors for regularization
  fn weight_tensors(&self) -> Vec<&Tensor>;

  /// Get references to bias tensors for regularization  
  fn bias_tensors(&self) -> Vec<&Tensor>;

  /// Add regularization gradients to the model's parameter gradients
  fn add_regularization_gradients(&mut self, weight_grads: Vec<Tensor>, bias_grads: Vec<Tensor>);
}

/// Loss trait implementation for RegularizedLoss - DO NOT USE DIRECTLY
///
/// This implementation exists only for trait compatibility but will panic when used.
/// RegularizedLoss requires model parameters for regularization computation.
/// Use `compute_loss` and `compute_gradients` methods instead.
impl<L: Loss> Loss for RegularizedLoss<L> {
  /// DO NOT USE: Use `compute_loss` method instead
  fn forward(&self, _predictions: &Tensor, _targets: &Tensor) -> Result<Tensor> {
    panic!(
      "RegularizedLoss::forward does not include regularization terms. \
       Use RegularizedLoss::compute_loss(predictions, targets, model) instead."
    );
  }

  /// DO NOT USE: Use `compute_gradients` method instead  
  fn backward(&self, _predictions: &Tensor, _targets: &Tensor) -> Result<Tensor> {
    panic!(
      "RegularizedLoss::backward does not include regularization gradients. \
       Use RegularizedLoss::compute_gradients(predictions, targets, model) instead."
    );
  }

  fn name(&self) -> &'static str {
    "RegularizedLoss"
  }
}

impl<L: Loss> RegularizedLoss<L> {
  /// Forward pass with regularization terms
  /// Compute the total regularized loss including L1/L2 penalties
  ///
  /// This computes the total loss including regularization:
  /// L_total = L_base + R_L1 + R_L2
  ///
  /// # Arguments
  /// * `predictions` - Model predictions
  /// * `targets` - Target values
  /// * `model` - Model implementing RegularizableModel for parameter access
  ///
  /// # Returns
  /// Total loss including base loss and regularization terms
  pub fn compute_loss<M: RegularizableModel>(
    &self,
    predictions: &Tensor,
    targets: &Tensor,
    model: &M,
  ) -> Result<Tensor> {
    // Compute base loss
    let base_loss = self.base_loss.forward(predictions, targets)?;

    if !self.config.is_enabled() {
      return Ok(base_loss);
    }

    let penalty = self.config.compute_regularization_penalty(model)?;
    base_loss.add(&penalty)
  }

  /// Add regularization gradients to the model's existing parameter gradients
  ///
  /// This assumes that the base loss gradients have already been computed
  /// and are present in the model's parameter gradients. This method adds
  /// the regularization gradients on top of the existing gradients.
  ///
  /// # Arguments
  /// * `model` - Mutable model to add regularization gradients to
  ///
  /// # Returns
  /// Empty tensor (regularization doesn't contribute to prediction gradients)
  pub fn compute_gradients<M: RegularizableModel>(&self, model: &mut M) -> Result<Tensor> {
    if !self.config.is_enabled() {
      return Ok(Tensor::zeros(1, 1));
    }

    self.config.add_regularization_gradients(model)?;

    Ok(Tensor::zeros(1, 1))
  }
}

#[cfg(test)]
mod regularization_tests {
  use super::*;

  use approx::assert_abs_diff_eq;

  /// Mock model for testing regularization
  struct MockModel {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
  }

  impl MockModel {
    fn new() -> Self {
      let weight1 = Tensor::new(vec![vec![2.0, -1.0], vec![3.0, -2.0]]).unwrap(); // 2x2
      let bias1 = Tensor::new(vec![vec![0.5, -0.5]]).unwrap(); // 1x2
      let weight2 = Tensor::new(vec![vec![1.0], vec![-1.0]]).unwrap(); // 2x1
      let bias2 = Tensor::new(vec![vec![0.1]]).unwrap(); // 1x1

      Self {
        weights: vec![weight1, weight2],
        biases: vec![bias1, bias2],
      }
    }
  }

  impl RegularizableModel for MockModel {
    fn weight_tensors(&self) -> Vec<&Tensor> {
      self.weights.iter().collect()
    }

    fn bias_tensors(&self) -> Vec<&Tensor> {
      self.biases.iter().collect()
    }

    fn add_regularization_gradients(&mut self, weight_grads: Vec<Tensor>, bias_grads: Vec<Tensor>) {
      // Mock implementation - just store gradients
      for (i, weight_grad) in weight_grads.into_iter().enumerate() {
        if i < self.weights.len() {
          self.weights[i].grad = Some(weight_grad.data);
        }
      }
      for (i, bias_grad) in bias_grads.into_iter().enumerate() {
        if i < self.biases.len() {
          self.biases[i].grad = Some(bias_grad.data);
        }
      }
    }
  }

  #[test]
  fn test_regularization_config_creation() {
    let config = RegularizationConfig::new(0.01, 0.02, true);
    assert_eq!(config.l1_lambda, 0.01);
    assert_eq!(config.l2_lambda, 0.02);
    assert!(config.apply_to_bias);
    assert!(config.is_enabled());
    assert!(config.has_l1());
    assert!(config.has_l2());

    let l1_only = RegularizationConfig::l1_only(0.01);
    assert_eq!(l1_only.l1_lambda, 0.01);
    assert_eq!(l1_only.l2_lambda, 0.0);
    assert!(!l1_only.apply_to_bias);
    assert!(l1_only.has_l1());
    assert!(!l1_only.has_l2());

    let l2_only = RegularizationConfig::l2_only(0.02);
    assert_eq!(l2_only.l1_lambda, 0.0);
    assert_eq!(l2_only.l2_lambda, 0.02);
    assert!(!l2_only.has_l1());
    assert!(l2_only.has_l2());

    let elastic_net = RegularizationConfig::elastic_net(0.01, 0.02);
    assert_eq!(elastic_net.l1_lambda, 0.01);
    assert_eq!(elastic_net.l2_lambda, 0.02);
    assert!(elastic_net.has_l1());
    assert!(elastic_net.has_l2());
  }

  #[test]
  #[should_panic(expected = "L1 lambda must be non-negative")]
  fn test_regularization_config_negative_l1() {
    RegularizationConfig::new(-0.01, 0.02, false);
  }

  #[test]
  #[should_panic(expected = "L2 lambda must be non-negative")]
  fn test_regularization_config_negative_l2() {
    RegularizationConfig::new(0.01, -0.02, false);
  }

  #[test]
  fn test_regularized_loss_creation() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.001);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    assert_eq!(reg_loss.name(), "RegularizedLoss");
    assert_eq!(reg_loss.base_loss().name(), "BinaryCrossEntropy");
    assert!(reg_loss.config().has_l2());
    assert!(!reg_loss.config().has_l1());
  }

  #[test]
  fn test_l1_regularization_computation() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l1_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1_reg = reg_loss
      .config()
      .compute_l1_regularization(&weights, &biases)
      .unwrap();

    // Expected L1 = 0.1 * (|2| + |-1| + |3| + |-2| + |1| + |-1|) = 0.1 * 10 = 1.0
    assert_abs_diff_eq!(l1_reg.data[[0, 0]], 1.0, epsilon = 1e-6);
  }

  #[test]
  fn test_l2_regularization_computation() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l2_reg = reg_loss
      .config()
      .compute_l2_regularization(&weights, &biases)
      .unwrap();

    // Expected L2 = 0.1 * (2² + (-1)² + 3² + (-2)² + 1² + (-1)²) = 0.1 * (4 + 1 + 9 + 4 + 1 + 1) = 0.1 * 20 = 2.0
    assert_abs_diff_eq!(l2_reg.data[[0, 0]], 2.0, epsilon = 1e-6);
  }

  #[test]
  fn test_l1_regularization_with_bias() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::new(0.1, 0.0, true); // L1 with bias
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1_reg = reg_loss
      .config()
      .compute_l1_regularization(&weights, &biases)
      .unwrap();

    // Expected L1 = 0.1 * (|2| + |-1| + |3| + |-2| + |1| + |-1| + |0.5| + |-0.5| + |0.1|)
    //              = 0.1 * (2 + 1 + 3 + 2 + 1 + 1 + 0.5 + 0.5 + 0.1) = 0.1 * 11.1 = 1.11
    assert_abs_diff_eq!(l1_reg.data[[0, 0]], 1.11, epsilon = 1e-6);
  }

  #[test]
  fn test_elastic_net_regularization() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::elastic_net(0.05, 0.05);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1_reg = reg_loss
      .config()
      .compute_l1_regularization(&weights, &biases)
      .unwrap();
    let l2_reg = reg_loss
      .config()
      .compute_l2_regularization(&weights, &biases)
      .unwrap();

    // L1 = 0.05 * 10 = 0.5, L2 = 0.05 * 20 = 1.0
    assert_abs_diff_eq!(l1_reg.data[[0, 0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(l2_reg.data[[0, 0]], 1.0, epsilon = 1e-6);
  }

  #[test]
  fn test_regularized_loss_compute_loss() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();

    // Create mock predictions and targets
    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let total_loss = reg_loss
      .compute_loss(&predictions, &targets, &model)
      .unwrap();

    // Should be base loss + regularization term
    let base_only_loss = reg_loss
      .base_loss()
      .forward(&predictions, &targets)
      .unwrap();
    assert!(total_loss.data[[0, 0]] > base_only_loss.data[[0, 0]]);
  }

  #[test]
  fn test_l1_gradient_computation() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l1_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l1_grads = reg_loss.config().compute_l1_gradient(&weights, &biases);

    // First weight matrix: [[2.0, -1.0], [3.0, -2.0]]
    // L1 gradient = 0.1 * sign(w) = [[0.1, -0.1], [0.1, -0.1]]
    assert_eq!(l1_grads.len(), 2);
    assert_abs_diff_eq!(l1_grads[0].0.data[[0, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(l1_grads[0].0.data[[0, 1]], -0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(l1_grads[0].0.data[[1, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(l1_grads[0].0.data[[1, 1]], -0.1, epsilon = 1e-6);

    // Second weight matrix: [[1.0], [-1.0]]
    // L1 gradient = 0.1 * sign(w) = [[0.1], [-0.1]]
    assert_abs_diff_eq!(l1_grads[1].0.data[[0, 0]], 0.1, epsilon = 1e-6);
    assert_abs_diff_eq!(l1_grads[1].0.data[[1, 0]], -0.1, epsilon = 1e-6);
  }

  #[test]
  fn test_l2_gradient_computation() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();
    let weights = model.weight_tensors();
    let biases = model.bias_tensors();

    let l2_grads = reg_loss.config().compute_l2_gradient(&weights, &biases);

    // First weight matrix: [[2.0, -1.0], [3.0, -2.0]]
    // L2 gradient = 2 * 0.1 * w = [[0.4, -0.2], [0.6, -0.4]]
    assert_eq!(l2_grads.len(), 2);
    assert_abs_diff_eq!(l2_grads[0].0.data[[0, 0]], 0.4, epsilon = 1e-6);
    assert_abs_diff_eq!(l2_grads[0].0.data[[0, 1]], -0.2, epsilon = 1e-6);
    assert_abs_diff_eq!(l2_grads[0].0.data[[1, 0]], 0.6, epsilon = 1e-6);
    assert_abs_diff_eq!(l2_grads[0].0.data[[1, 1]], -0.4, epsilon = 1e-6);

    // Second weight matrix: [[1.0], [-1.0]]
    // L2 gradient = 2 * 0.1 * w = [[0.2], [-0.2]]
    assert_abs_diff_eq!(l2_grads[1].0.data[[0, 0]], 0.2, epsilon = 1e-6);
    assert_abs_diff_eq!(l2_grads[1].0.data[[1, 0]], -0.2, epsilon = 1e-6);
  }

  #[test]
  fn test_regularized_loss_no_regularization() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::new(0.0, 0.0, false);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let model = MockModel::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let total_loss = reg_loss
      .compute_loss(&predictions, &targets, &model)
      .unwrap();
    let base_only_loss = reg_loss
      .base_loss()
      .forward(&predictions, &targets)
      .unwrap();

    // Should be identical when no regularization
    assert_abs_diff_eq!(
      total_loss.data[[0, 0]],
      base_only_loss.data[[0, 0]],
      epsilon = 1e-10
    );
  }

  #[test]
  #[should_panic(expected = "RegularizedLoss::forward does not include regularization terms")]
  fn test_regularized_loss_forward_panics() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    // This should panic
    let _ = reg_loss.forward(&predictions, &targets);
  }

  #[test]
  #[should_panic(expected = "RegularizedLoss::backward does not include regularization gradients")]
  fn test_regularized_loss_backward_panics() {
    let base_loss = BinaryCrossEntropy::new();
    let config = RegularizationConfig::l2_only(0.1);
    let reg_loss = RegularizedLoss::new(base_loss, config);

    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    // This should panic
    let _ = reg_loss.backward(&predictions, &targets);
  }
}
