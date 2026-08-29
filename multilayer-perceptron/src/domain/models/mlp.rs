use super::layer::{Activation, DenseLayer, Layer, WeightInit};
use crate::core::{ComputationGraph, Result, Tensor};
use crate::domain::services::loss::RegularizableModel;

/// Sequential model for building neural networks layer by layer
///
/// A Sequential model is a linear stack of layers where data flows from input to output
/// through each layer in sequence. This is the most common type of neural network architecture.
#[derive(Debug, Clone)]
pub struct Sequential {
  /// List of layers in the model
  layers: Vec<Box<dyn Layer>>,
  /// Flag indicating whether model is in training mode
  training: bool,
  /// Computation graph for automatic differentiation
  graph: Option<std::rc::Rc<std::cell::RefCell<ComputationGraph>>>,
}

impl Sequential {
  /// Create a new empty Sequential model
  pub fn new() -> Self {
    Self {
      layers: Vec::new(),
      training: true,
      graph: None,
    }
  }

  /// Add a layer to the model
  ///
  /// # Arguments
  /// * `layer` - The layer to add to the model
  ///
  /// # Returns
  /// Returns self for method chaining
  #[allow(clippy::should_implement_trait)]
  pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
    self.layers.push(Box::new(layer));
    self
  }

  /// Add a dense layer with specified parameters
  ///
  /// # Arguments
  /// * `input_size` - Number of input features
  /// * `output_size` - Number of output features
  /// * `activation` - Activation function to use
  /// * `weight_init` - Weight initialization method
  ///
  /// # Returns
  /// Returns self for method chaining
  pub fn dense(
    self,
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weight_init: WeightInit,
  ) -> Self {
    self.add(DenseLayer::new(
      input_size,
      output_size,
      activation,
      weight_init,
    ))
  }

  /// Add a dense layer with ReLU activation and He initialization
  pub fn relu_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::relu(input_size, output_size))
  }

  /// Add a dense layer with sigmoid activation
  pub fn sigmoid_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::sigmoid(input_size, output_size))
  }

  /// Add a dense layer with tanh activation
  pub fn tanh_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::tanh(input_size, output_size))
  }

  /// Add a dense layer with softmax activation (typically for output layer)
  pub fn softmax_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::softmax(input_size, output_size))
  }

  /// Add a dense layer with no activation (linear layer)
  pub fn linear_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::new(
      input_size,
      output_size,
      Activation::None,
      WeightInit::XavierUniform,
    ))
  }

  pub fn set_output_activation(&mut self, activation: Activation) {
    if let Some(layer) = self.layers.last_mut() {
      if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
        dense_layer.set_activation(activation);
      }
    }
  }

  /// Set up computation graph for the model
  pub fn with_graph(mut self, graph: std::rc::Rc<std::cell::RefCell<ComputationGraph>>) -> Self {
    self.graph = Some(graph.clone());

    // Connect all layer parameters to the computation graph
    for layer in &mut self.layers {
      if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
        dense_layer.connect_to_graph(graph.clone());
      }
    }

    self
  }

  /// Get the number of layers in the model
  pub fn len(&self) -> usize {
    self.layers.len()
  }

  /// Check if the model is empty
  pub fn is_empty(&self) -> bool {
    self.layers.is_empty()
  }

  /// Set training mode
  pub fn train(&mut self) {
    self.training = true;
  }

  /// Set evaluation mode
  pub fn eval(&mut self) {
    self.training = false;
  }

  /// Check if model is in training mode
  pub fn is_training(&self) -> bool {
    self.training
  }

  /// Forward pass through the model
  pub fn forward(&mut self, mut input: Tensor) -> Result<Tensor> {
    if let Some(graph) = &self.graph {
      input = input.with_graph(graph.clone());
    }

    for layer in &mut self.layers {
      input = layer.forward(&input)?;
    }
    Ok(input)
  }

  /// Reset gradients for all layers
  pub fn zero_gradients(&mut self) {
    for layer in &mut self.layers {
      layer.zero_gradients();
    }
  }

  pub fn zero_grad(&mut self) {
    self.zero_gradients();
    if let Some(graph) = &self.graph {
      graph.borrow_mut().zero_grad();
    }
  }

  pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    let mut params = Vec::new();
    for layer in &mut self.layers {
      params.extend(layer.parameters_mut());
    }
    params
  }

  pub fn summary(&self) -> ModelSummary {
    let mut total_params = 0;
    let mut layer_info = Vec::new();

    let mut current_shape = (0, 0);

    for (i, layer) in self.layers.iter().enumerate() {
      let layer_params = layer.param_count();
      total_params += layer_params;

      if i == 0 {
        layer_info.push(LayerInfo {
          name: layer.name().to_string(),
          input_shape: (0, 0),
          output_shape: (0, 0),
          param_count: layer_params,
        });
      } else {
        let output_shape = layer.output_shape(current_shape);
        layer_info.push(LayerInfo {
          name: layer.name().to_string(),
          input_shape: current_shape,
          output_shape,
          param_count: layer_params,
        });
        current_shape = output_shape;
      }
    }

    ModelSummary {
      layers: layer_info,
      total_params,
    }
  }

  pub fn summary_with_input_shape(&self, input_shape: (usize, usize)) -> ModelSummary {
    let mut total_params = 0;
    let mut layer_info = Vec::new();
    let mut current_shape = input_shape;

    for layer in &self.layers {
      let layer_params = layer.param_count();
      total_params += layer_params;

      let output_shape = layer.output_shape(current_shape);
      layer_info.push(LayerInfo {
        name: layer.name().to_string(),
        input_shape: current_shape,
        output_shape,
        param_count: layer_params,
      });
      current_shape = output_shape;
    }

    ModelSummary {
      layers: layer_info,
      total_params,
    }
  }

  /// Snapshot dense layer weight matrices for visualization/debugging
  pub fn weight_matrices(&self) -> Vec<Vec<Vec<f64>>> {
    let mut matrices = Vec::new();

    for layer in &self.layers {
      if let Some(dense) = layer.as_any().downcast_ref::<DenseLayer>() {
        let (rows, cols) = dense.weights().shape();
        let mut matrix = Vec::with_capacity(rows);
        for r in 0..rows {
          let mut row = Vec::with_capacity(cols);
          for c in 0..cols {
            row.push(dense.weights().data[[r, c]]);
          }
          matrix.push(row);
        }
        matrices.push(matrix);
      }
    }

    matrices
  }

  /// Snapshot dense layer bias vectors for visualization/debugging
  pub fn bias_vectors(&self) -> Vec<Vec<f64>> {
    let mut vectors = Vec::new();

    for layer in &self.layers {
      if let Some(dense) = layer.as_any().downcast_ref::<DenseLayer>() {
        let (_, cols) = dense.bias().shape();
        let mut bias = Vec::with_capacity(cols);
        for c in 0..cols {
          bias.push(dense.bias().data[[0, c]]);
        }
        vectors.push(bias);
      }
    }

    vectors
  }

  /// Sync gradients from computation graph
  pub fn sync_gradients(&mut self) -> Result<()> {
    for layer in &mut self.layers {
      layer.sync_gradients()?;
    }
    Ok(())
  }
}

impl Default for Sequential {
  fn default() -> Self {
    Self::new()
  }
}

pub type MLP = Sequential;

/// Information about a single layer in the model
#[derive(Debug, Clone)]
pub struct LayerInfo {
  pub name: String,
  pub input_shape: (usize, usize),
  pub output_shape: (usize, usize),
  pub param_count: usize,
}

impl RegularizableModel for Sequential {
  fn weight_tensors(&self) -> Vec<&Tensor> {
    let mut weights = Vec::new();
    for layer in &self.layers {
      if let Some(dense_layer) = layer.as_any().downcast_ref::<DenseLayer>() {
        weights.push(dense_layer.weights());
      }
    }
    weights
  }

  fn bias_tensors(&self) -> Vec<&Tensor> {
    let mut biases = Vec::new();
    for layer in &self.layers {
      if let Some(dense_layer) = layer.as_any().downcast_ref::<DenseLayer>() {
        biases.push(dense_layer.bias());
      }
    }
    biases
  }

  fn add_regularization_gradients(&mut self, weight_grads: Vec<Tensor>, bias_grads: Vec<Tensor>) {
    let mut weight_idx = 0;
    let mut bias_idx = 0;

    for layer in &mut self.layers {
      if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
        if weight_idx < weight_grads.len() && bias_idx < bias_grads.len() {
          let accumulate = |param: &mut Tensor, grad: &Tensor| {
            // Update the parameter's local gradient storage
            if let Some(existing_grad) = param.grad.as_mut() {
              *existing_grad = existing_grad.clone() + &grad.data;
            } else {
              param.grad = Some(grad.data.clone());
            }

            // Also update the gradient stored in the computation graph node so that
            // optimizers reading from the graph see the regularization contribution.
            if let (Some(graph_weak), Some(node_id)) = (&param.graph, param.graph_id) {
              if let Some(graph_rc) = graph_weak.upgrade() {
                let node_rc_opt = { graph_rc.borrow().get_node(node_id) };
                if let Some(node_rc) = node_rc_opt {
                  let mut node_ref = node_rc.borrow_mut();
                  if let Some(existing_graph_grad) = node_ref.tensor.grad.as_mut() {
                    *existing_graph_grad = existing_graph_grad.clone() + &grad.data;
                  } else {
                    node_ref.tensor.grad = Some(grad.data.clone());
                  }
                }
              }
            }
          };

          accumulate(dense_layer.weights_mut(), &weight_grads[weight_idx]);
          accumulate(dense_layer.bias_mut(), &bias_grads[bias_idx]);

          weight_idx += 1;
          bias_idx += 1;
        }
      }
    }
  }
}

/// Summary information about the model
#[derive(Debug, Clone)]
pub struct ModelSummary {
  pub layers: Vec<LayerInfo>,
  pub total_params: usize,
}

impl std::fmt::Display for ModelSummary {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "Model Summary")?;
    writeln!(f, "=============")?;
    writeln!(
      f,
      "{:<15} {:<20} {:<20} {:<15}",
      "Layer", "Input Shape", "Output Shape", "Params"
    )?;
    writeln!(
      f,
      "-----------------------------------------------------------------------------"
    )?;

    for layer in &self.layers {
      writeln!(
        f,
        "{:<15} {:<20} {:<20} {:<15}",
        layer.name,
        format!("({}, {})", layer.input_shape.0, layer.input_shape.1),
        format!("({}, {})", layer.output_shape.0, layer.output_shape.1),
        layer.param_count
      )?;
    }

    writeln!(
      f,
      "============================================================================="
    )?;
    writeln!(f, "Total params: {}", self.total_params)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;
  use std::cell::RefCell;
  use std::rc::Rc;

  #[test]
  fn test_sequential_creation() {
    let model = Sequential::new();

    assert_eq!(model.len(), 0);
    assert!(model.is_empty());
    assert!(model.is_training());
  }

  #[test]
  fn test_sequential_add_layers() {
    let model = Sequential::new()
      .dense(784, 128, Activation::ReLU, WeightInit::HeUniform)
      .dense(128, 64, Activation::ReLU, WeightInit::HeUniform)
      .dense(64, 10, Activation::Softmax, WeightInit::XavierUniform);

    assert_eq!(model.len(), 3);
    assert!(!model.is_empty());
  }

  #[test]
  fn test_sequential_convenience_methods() {
    let model = Sequential::new()
      .relu_layer(784, 128)
      .sigmoid_layer(128, 64)
      .softmax_layer(64, 10);

    assert_eq!(model.len(), 3);
  }

  #[test]
  fn test_sequential_forward() {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut model = Sequential::new()
      .dense(3, 2, Activation::None, WeightInit::Ones) // Linear layer for predictable output
      .with_graph(graph.clone());

    // Create input: batch_size=1, features=3
    let input_data = vec![vec![1.0, 2.0, 3.0]];
    let input = Tensor::new(input_data).unwrap();

    let output = model.forward(input).unwrap();

    // With ones initialization: output = [1,2,3] * [[1,1],[1,1],[1,1]] + [0,0] = [6,6]
    assert_eq!(output.shape(), (1, 2));
    assert_abs_diff_eq!(output.data[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output.data[[0, 1]], 6.0, epsilon = 1e-6);
  }

  #[test]
  fn test_sequential_training_mode() {
    let mut model = Sequential::new();

    assert!(model.is_training());

    model.eval();
    assert!(!model.is_training());

    model.train();
    assert!(model.is_training());
  }
}
