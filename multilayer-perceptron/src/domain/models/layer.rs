use crate::core::{ComputationGraph, Result, Tensor};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
  None,
  Sigmoid,
  ReLU,
  Tanh,
  Softmax,
}

impl Activation {
  pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
    match self {
      Activation::None => Ok(input.clone()),
      Activation::Sigmoid => input.sigmoid(),
      Activation::ReLU => input.relu(),
      Activation::Tanh => input.tanh(),
      Activation::Softmax => input.softmax(),
    }
  }

  pub fn name(&self) -> &'static str {
    match self {
      Activation::None => "none",
      Activation::Sigmoid => "sigmoid",
      Activation::ReLU => "relu",
      Activation::Tanh => "tanh",
      Activation::Softmax => "softmax",
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightInit {
  Zeros,
  Ones,
  Random,
  XavierUniform,
  HeUniform,
}

pub trait Layer: Debug {
  fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
  fn output_shape(&self, input_shape: (usize, usize)) -> (usize, usize);
  fn param_count(&self) -> usize;
  fn name(&self) -> &'static str;
  fn clone_layer(&self) -> Box<dyn Layer>;
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
  fn as_any(&self) -> &dyn std::any::Any;

  /// Get mutable references to parameters (for optimizer)
  fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    Vec::new()
  }

  /// Zero out gradients for this layer
  fn zero_gradients(&mut self) {
    for param in self.parameters_mut() {
      param.zero_grad();
    }
  }

  /// Sync gradients from computation graph
  fn sync_gradients(&mut self) -> Result<()> {
    for param in self.parameters_mut() {
      param.sync_gradient_from_graph()?;
    }
    Ok(())
  }
}

impl Clone for Box<dyn Layer> {
  fn clone(&self) -> Self {
    self.clone_layer()
  }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
  pub(crate) weights: Tensor,
  pub(crate) bias: Tensor,
  pub(crate) activation: Activation,
  pub(crate) input_size: usize,
  pub(crate) output_size: usize,
}

impl DenseLayer {
  pub fn new(
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weight_init: WeightInit,
  ) -> Self {
    let (weights, bias) = Self::initialize_params(input_size, output_size, weight_init);

    Self {
      weights,
      bias,
      activation,
      input_size,
      output_size,
    }
  }

  pub fn with_activation(input_size: usize, output_size: usize, activation: Activation) -> Self {
    Self::new(
      input_size,
      output_size,
      activation,
      WeightInit::XavierUniform,
    )
  }

  pub fn relu(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::ReLU,
      WeightInit::HeUniform,
    )
  }

  pub fn sigmoid(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::Sigmoid,
      WeightInit::XavierUniform,
    )
  }

  pub fn tanh(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::Tanh,
      WeightInit::XavierUniform,
    )
  }

  pub fn softmax(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::Softmax,
      WeightInit::XavierUniform,
    )
  }

  pub fn set_activation(&mut self, activation: Activation) {
    self.activation = activation;
  }

  fn initialize_params(
    input_size: usize,
    output_size: usize,
    init_method: WeightInit,
  ) -> (Tensor, Tensor) {
    use rand::Rng;

    let weights = match init_method {
      WeightInit::Zeros => Tensor::zeros(input_size, output_size),
      WeightInit::Ones => Tensor::ones(input_size, output_size),
      WeightInit::Random => Tensor::random(input_size, output_size),
      WeightInit::XavierUniform => {
        // Xavier/Glorot initialization: uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..input_size)
          .map(|_| {
            (0..output_size)
              .map(|_| rng.gen_range(-limit..limit))
              .collect()
          })
          .collect();
        Tensor::new(data).expect("Failed to create Xavier initialized weights")
      }
      WeightInit::HeUniform => {
        // He initialization: uniform(-√(6/fan_in), √(6/fan_in))
        let limit = (6.0 / input_size as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..input_size)
          .map(|_| {
            (0..output_size)
              .map(|_| rng.gen_range(-limit..limit))
              .collect()
          })
          .collect();
        Tensor::new(data).expect("Failed to create He initialized weights")
      }
    };

    // Bias is typically initialized to zeros
    let bias = Tensor::zeros(1, output_size);

    (weights, bias)
  }

  pub fn enable_gradients(&mut self) {
    self.weights.set_requires_grad(true);
    self.bias.set_requires_grad(true);
  }

  /// Connect layer parameters to computation graph
  pub fn connect_to_graph(&mut self, graph: std::rc::Rc<std::cell::RefCell<ComputationGraph>>) {
    // Ensure tensors require gradients before registering them with the graph so the
    // graph copies also retain the proper requires_grad flag.
    self.weights.set_requires_grad(true);
    self.bias.set_requires_grad(true);

    self.weights = self.weights.clone().with_graph(graph.clone());
    self.bias = self.bias.clone().with_graph(graph.clone());
  }

  /// Get reference to weights tensor
  pub fn weights(&self) -> &Tensor {
    &self.weights
  }

  /// Get reference to bias tensor
  pub fn bias(&self) -> &Tensor {
    &self.bias
  }

  /// Get mutable reference to weights tensor
  pub fn weights_mut(&mut self) -> &mut Tensor {
    &mut self.weights
  }

  /// Get mutable reference to bias tensor
  pub fn bias_mut(&mut self) -> &mut Tensor {
    &mut self.bias
  }
}

impl Layer for DenseLayer {
  fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
    if !self.weights.requires_grad() {
      self.enable_gradients();
    }

    let linear_output = input.matmul(&self.weights)?;
    let with_bias = linear_output.add(&self.bias)?;

    self.activation.apply(&with_bias)
  }

  fn output_shape(&self, input_shape: (usize, usize)) -> (usize, usize) {
    (input_shape.0, self.output_size)
  }

  fn param_count(&self) -> usize {
    (self.input_size * self.output_size) + self.output_size
  }

  fn name(&self) -> &'static str {
    "DenseLayer"
  }

  fn clone_layer(&self) -> Box<dyn Layer> {
    Box::new(self.clone())
  }

  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    vec![&mut self.weights, &mut self.bias]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;
  use std::cell::RefCell;
  use std::rc::Rc;

  #[test]
  fn test_activation_enum() {
    assert_eq!(Activation::Sigmoid.name(), "sigmoid");
    assert_eq!(Activation::ReLU.name(), "relu");
    assert_eq!(Activation::Softmax.name(), "softmax");
    assert_eq!(Activation::None.name(), "none");
    assert_eq!(Activation::Tanh.name(), "tanh");
  }

  #[test]
  fn test_dense_layer_creation() {
    let layer = DenseLayer::new(10, 5, Activation::ReLU, WeightInit::XavierUniform);

    assert_eq!(layer.input_size, 10);
    assert_eq!(layer.output_size, 5);
    assert_eq!(layer.activation, Activation::ReLU);
    assert_eq!(layer.weights.shape(), (10, 5));
    assert_eq!(layer.bias.shape(), (1, 5));
  }

  #[test]
  fn test_dense_layer_convenience_constructors() {
    let relu_layer = DenseLayer::relu(784, 128);
    assert_eq!(relu_layer.activation, Activation::ReLU);
    assert_eq!(relu_layer.input_size, 784);
    assert_eq!(relu_layer.output_size, 128);

    let sigmoid_layer = DenseLayer::sigmoid(128, 64);
    assert_eq!(sigmoid_layer.activation, Activation::Sigmoid);

    let softmax_layer = DenseLayer::softmax(64, 10);
    assert_eq!(softmax_layer.activation, Activation::Softmax);
  }

  #[test]
  fn test_dense_layer_forward() {
    let mut layer = DenseLayer::new(3, 2, Activation::None, WeightInit::Ones);

    // Create input tensor
    let input_data = vec![vec![1.0, 2.0, 3.0]]; // batch size 1, 3 features
    let mut input = Tensor::new(input_data).unwrap();
    input.set_requires_grad(true);

    // Set up computation graph
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let input = input.with_graph(graph);

    // Forward pass
    let output = layer.forward(&input).unwrap();

    // With ones initialization: output = [1,2,3] * [[1,1],[1,1],[1,1]] + [0,0] = [6,6]
    assert_eq!(output.shape(), (1, 2));
    assert_abs_diff_eq!(output.data[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output.data[[0, 1]], 6.0, epsilon = 1e-6);
  }

  #[test]
  fn test_layer_trait_methods() {
    let layer = DenseLayer::new(784, 128, Activation::ReLU, WeightInit::XavierUniform);

    assert_eq!(layer.output_shape((32, 784)), (32, 128));
    assert_eq!(layer.param_count(), 784 * 128 + 128);
    assert_eq!(layer.name(), "DenseLayer");
  }

  #[test]
  fn test_layer_cloning() {
    let layer = DenseLayer::new(10, 5, Activation::ReLU, WeightInit::Random);
    let boxed: Box<dyn Layer> = Box::new(layer);
    let cloned_boxed = boxed.clone();

    assert_eq!(boxed.name(), cloned_boxed.name());
    assert_eq!(boxed.param_count(), cloned_boxed.param_count());
    assert_eq!(
      boxed.output_shape((1, 10)),
      cloned_boxed.output_shape((1, 10))
    );
  }

  #[test]
  fn test_weight_initialization_shapes() {
    let layer = DenseLayer::new(100, 50, Activation::Sigmoid, WeightInit::Random);

    assert_eq!(layer.weights.shape(), (100, 50));
    assert_eq!(layer.bias.shape(), (1, 50));

    // Check that weights are not all the same (should be random)
    let first_weight = layer.weights.data[[0, 0]];
    let mut all_same = true;
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        if layer.weights.data[[i, j]] != first_weight {
          all_same = false;
          break;
        }
      }
      if !all_same {
        break;
      }
    }
    assert!(
      !all_same,
      "Weights should not all be the same with random initialization"
    );
  }

  #[test]
  fn test_xavier_uniform_initialization_range() {
    let input_size = 100;
    let output_size = 50;
    let layer = DenseLayer::new(
      input_size,
      output_size,
      Activation::Sigmoid,
      WeightInit::XavierUniform,
    );

    let expected_limit = (6.0 / (input_size + output_size) as f64).sqrt();

    // Check that all weights are within the expected range
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        let weight = layer.weights.data[[i, j]];
        assert!(
          weight >= -expected_limit && weight <= expected_limit,
          "Weight {} is outside expected range [{}, {}]",
          weight,
          -expected_limit,
          expected_limit
        );
      }
    }
  }

  #[test]
  fn test_he_uniform_initialization_range() {
    let input_size = 784;
    let output_size = 128;
    let layer = DenseLayer::new(
      input_size,
      output_size,
      Activation::ReLU,
      WeightInit::HeUniform,
    );

    let expected_limit = (6.0 / input_size as f64).sqrt();

    // Check that all weights are within the expected range
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        let weight = layer.weights.data[[i, j]];
        assert!(
          weight >= -expected_limit && weight <= expected_limit,
          "Weight {} is outside expected range [{}, {}]",
          weight,
          -expected_limit,
          expected_limit
        );
      }
    }
  }
}
