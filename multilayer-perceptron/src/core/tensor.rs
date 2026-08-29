use super::error::{Result, TensorError};
use super::graph::{ComputationGraph, NodeId};
use super::ops::OpBuilder;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::fmt;
use std::rc::{Rc, Weak};

/// Macro to simplify graph operations by handling Weak::upgrade() patterns
macro_rules! with_graph {
  ($tensor:expr, |$graph:ident| $body:block) => {
    if let (Some(graph_weak), Some(_)) = (&$tensor.graph, $tensor.graph_id) {
      if let Some($graph) = graph_weak.upgrade() {
        return $body;
      } else {
        return Err(TensorError::GraphDropped);
      }
    }
  };

  ($tensor:expr, |$graph:ident, $node_id:ident| $body:block) => {
    if let (Some(graph_weak), Some($node_id)) = (&$tensor.graph, $tensor.graph_id) {
      if let Some($graph) = graph_weak.upgrade() {
        return $body;
      } else {
        return Err(TensorError::GraphDropped);
      }
    }
  };
}

#[derive(Clone)]
pub struct Tensor {
  pub data: Array2<f64>,

  pub grad: Option<Array2<f64>>,
  pub requires_grad: bool,

  pub graph_id: Option<NodeId>,

  pub graph: Option<Weak<RefCell<ComputationGraph>>>,
}

impl Tensor {
  pub fn new(data: Vec<Vec<f64>>) -> Result<Self> {
    let rows = data.len();
    let cols = if rows > 0 { data[0].len() } else { 0 };

    let flat_data: Vec<f64> = data.into_iter().flat_map(|row| row.into_iter()).collect();
    let array =
      Array2::from_shape_vec((rows, cols), flat_data).map_err(|e| TensorError::InvalidInput {
        message: format!("Input data has inconsistent row lengths: {}", e),
      })?;

    Ok(Self {
      data: array,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  pub fn zeros(rows: usize, cols: usize) -> Self {
    Self {
      data: Array2::zeros((rows, cols)),
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    }
  }

  pub fn ones(rows: usize, cols: usize) -> Self {
    Self {
      data: Array2::ones((rows, cols)),
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    }
  }

  pub fn from_scalar(value: f64) -> Result<Self> {
    Ok(Self {
      data: Array2::from_elem((1, 1), value),
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  pub fn zeros_like(tensor: &Tensor) -> Result<Self> {
    let shape = tensor.shape();
    Ok(Self::zeros(shape.0, shape.1))
  }

  pub fn random(rows: usize, cols: usize) -> Self {
    let mut rng = thread_rng();
    let data = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0));

    Self {
      data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    }
  }

  /// Create tensor from Array2
  pub fn from_array2(data: Array2<f64>) -> Result<Self> {
    Ok(Self {
      data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Create tensor from Array1 (as column vector)
  pub fn from_array1(data: ndarray::Array1<f64>) -> Result<Self> {
    let rows = data.len();
    let data_2d = data.to_shape((rows, 1))?.to_owned();
    Ok(Self {
      data: data_2d,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  pub fn shape(&self) -> (usize, usize) {
    let shape = self.data.dim();
    (shape.0, shape.1)
  }

  pub fn dim(&self) -> (usize, usize) {
    self.shape()
  }

  pub fn len(&self) -> usize {
    self.data.len()
  }

  pub fn is_empty(&self) -> bool {
    self.data.is_empty()
  }

  pub fn zero_grad(&mut self) {
    if let Some(ref mut grad) = self.grad {
      grad.fill(0.0);
    }
  }

  pub fn set_requires_grad(&mut self, requires_grad: bool) {
    self.requires_grad = requires_grad;
    if requires_grad && self.grad.is_none() {
      self.grad = Some(Array2::zeros(self.data.dim()));
    } else if !requires_grad {
      self.grad = None;
    }
  }

  pub fn requires_grad(&self) -> bool {
    self.requires_grad
  }

  pub fn with_graph(mut self, graph: Rc<RefCell<ComputationGraph>>) -> Self {
    let node_id = graph.borrow_mut().add_leaf_node(self.clone());
    self.graph_id = Some(node_id);
    self.graph = Some(Rc::downgrade(&graph));
    self
  }

  pub fn is_tracked(&self) -> bool {
    self.graph_id.is_some() && self.graph.is_some()
  }

  /// Get gradient value at specific position, checking computation graph if needed
  pub fn grad_at(&self, row: usize, col: usize) -> Option<f64> {
    // If tensor is tracked, always get from computation graph (most up-to-date)
    if let (Some(graph_weak), Some(node_id)) = (&self.graph, self.graph_id) {
      if let Some(graph_ref) = graph_weak.upgrade() {
        if let Some(grad) = graph_ref.borrow().get_node_gradient(node_id) {
          return Some(grad[[row, col]]);
        }
      }
    }

    // Fall back to local gradient if not tracked
    if let Some(ref grad) = self.grad {
      return Some(grad[[row, col]]);
    }

    None
  }

  /// Get the full gradient tensor, checking computation graph if needed
  pub fn grad(&self) -> Option<Array2<f64>> {
    // If tensor is tracked, always get from computation graph (most up-to-date)
    if let (Some(graph_weak), Some(node_id)) = (&self.graph, self.graph_id) {
      if let Some(graph_ref) = graph_weak.upgrade() {
        return graph_ref.borrow().get_node_gradient(node_id);
      }
    }

    // Fall back to local gradient if not tracked
    if let Some(ref grad) = self.grad {
      return Some(grad.clone());
    }

    None
  }
  pub fn set_grad_at(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
    if let Some(ref mut grad) = self.grad {
      grad[[row, col]] = value;
      Ok(())
    } else {
      Err(TensorError::NoGradient)
    }
  }

  pub fn backward(&mut self) -> Result<()> {
    with_graph!(self, |graph, node_id| {
      graph.borrow_mut().backward(node_id, None)?;
      // Note: Gradients are now directly computed in the computation graph
      // and automatically available through the graph nodes
      Ok(())
    });

    Err(TensorError::NotTracked {
      tensor_id: self.graph_id,
    })
  }

  /// Sync gradient from computation graph to this tensor
  pub fn sync_gradient_from_graph(&mut self) -> Result<()> {
    with_graph!(self, |graph, node_id| {
      // List all nodes with gradients for debugging
      if let Some(grad) = graph.borrow().get_node_gradient(node_id) {
        self.grad = Some(grad);
      }
      Ok(())
    });

    Ok(())
  }

  /// Check if two tensors belong to the same computation graph
  pub fn same_graph(a: &Tensor, b: &Tensor) -> bool {
    match (&a.graph, &b.graph) {
      (Some(a_weak), Some(b_weak)) => {
        if let (Some(a_graph), Some(b_graph)) = (a_weak.upgrade(), b_weak.upgrade()) {
          Rc::ptr_eq(&a_graph, &b_graph)
        } else {
          false
        }
      }
      _ => false,
    }
  }

  /// Add operation to computation graph with proper cross-graph handling
  fn add_operation_to_graph(
    graph: Rc<RefCell<ComputationGraph>>,
    op: super::ops::OpNode,
    input_ids: Vec<NodeId>,
    result: Tensor,
  ) -> Result<Tensor> {
    let mut result = result;
    let output_id = graph
      .borrow_mut()
      .add_operation(op, input_ids, result.clone())?;
    result.graph = Some(Rc::downgrade(&graph));
    result.graph_id = Some(output_id);
    Ok(result)
  }

  // Helper method to add operation to computation graph (refactored for readability)
  fn add_to_graph(&self, other: &Tensor, op: super::ops::OpNode, result: Tensor) -> Result<Tensor> {
    match (self.get_graph_info(), other.get_graph_info()) {
      // Both tensors are tracked
      (Some((self_graph, self_id)), Some((other_graph, other_id))) => {
        if Rc::ptr_eq(&self_graph, &other_graph) {
          // Same graph: directly add operation
          Self::add_operation_to_graph(self_graph, op, vec![self_id, other_id], result)
        } else {
          // Different graphs: merge other into self's graph
          let other_copied_id = self_graph.borrow_mut().add_leaf_node(other.clone());
          Self::add_operation_to_graph(self_graph, op, vec![self_id, other_copied_id], result)
        }
      }
      // Only self has a graph
      (Some((graph, self_id)), None) => {
        let other_id = graph.borrow_mut().add_leaf_node(other.clone());
        Self::add_operation_to_graph(graph, op, vec![self_id, other_id], result)
      }
      // Only other has a graph
      (None, Some((graph, other_id))) => {
        let self_id = graph.borrow_mut().add_leaf_node(self.clone());
        Self::add_operation_to_graph(graph, op, vec![self_id, other_id], result)
      }
      // Neither tensor is tracked
      (None, None) => Ok(result),
    }
  }

  /// Get graph and node_id as a tuple if tensor is tracked
  fn get_graph_info(&self) -> Option<(Rc<RefCell<ComputationGraph>>, NodeId)> {
    if let (Some(graph_weak), Some(node_id)) = (&self.graph, self.graph_id) {
      graph_weak.upgrade().map(|graph| (graph, node_id))
    } else {
      None
    }
  }

  // Helper method for unary operations
  fn add_unary_to_graph(&self, op: super::ops::OpNode, mut result: Tensor) -> Result<Tensor> {
    if let (Some(graph_weak), Some(self_id)) = (&self.graph, self.graph_id) {
      if let Some(graph) = graph_weak.upgrade() {
        let output_id = graph
          .borrow_mut()
          .add_operation(op, vec![self_id], result.clone())?;
        result.graph = Some(Rc::downgrade(&graph));
        result.graph_id = Some(output_id);
        return Ok(result);
      } else {
        return Err(TensorError::GraphDropped);
      }
    }
    Ok(result)
  }

  pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
    let op = OpBuilder::matmul(Rc::new(self.clone()), Rc::new(other.clone()));
    let result = op.forward()?;
    self.add_to_graph(other, op, result)
  }

  /// Create tensor from existing Array2 data
  pub fn from_data(data: Array2<f64>) -> Result<Tensor> {
    Ok(Self {
      data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Get gradient (same as grad() for compatibility)
  pub fn gradient(&self) -> Option<Tensor> {
    self.grad.as_ref().map(|grad_data| Tensor {
      data: grad_data.clone(),
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Set gradient from tensor
  pub fn set_gradient(&mut self, gradient: Option<Tensor>) {
    if let Some(grad_tensor) = gradient {
      self.grad = Some(grad_tensor.data);
    } else {
      self.grad = None;
    }
  }

  /// Element-wise natural logarithm
  pub fn log(&self) -> Result<Tensor> {
    let mut result_data = self.data.clone();
    for elem in result_data.iter_mut() {
      if *elem <= 0.0 {
        return Err(TensorError::ComputationError {
          message: "Cannot compute log of non-positive number".to_string(),
        });
      }
      *elem = elem.ln();
    }
    Ok(Tensor {
      data: result_data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Element-wise negation
  pub fn neg(&self) -> Result<Tensor> {
    Ok(Tensor {
      data: -&self.data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Element-wise multiplication
  pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
    if self.shape() != other.shape() {
      return Err(TensorError::ShapeMismatch {
        operation: "mul".to_string(),
        expected: self.shape(),
        got: other.shape(),
      });
    }

    Ok(Tensor {
      data: &self.data * &other.data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Element-wise division
  pub fn div(&self, other: &Tensor) -> Result<Tensor> {
    if self.shape() != other.shape() {
      return Err(TensorError::ShapeMismatch {
        operation: "div".to_string(),
        expected: self.shape(),
        got: other.shape(),
      });
    }

    // Check for division by zero
    for elem in other.data.iter() {
      if elem.abs() < f64::EPSILON {
        return Err(TensorError::ComputationError {
          message: "Division by zero".to_string(),
        });
      }
    }

    Ok(Tensor {
      data: &self.data / &other.data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Element-wise subtraction
  pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
    if self.shape() != other.shape() {
      return Err(TensorError::ShapeMismatch {
        operation: "sub".to_string(),
        expected: self.shape(),
        got: other.shape(),
      });
    }

    Ok(Tensor {
      data: &self.data - &other.data,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Scalar multiplication
  pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
    Ok(Tensor {
      data: &self.data * scalar,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Add scalar to all elements
  pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
    Ok(Tensor {
      data: &self.data + scalar,
      grad: None,
      requires_grad: false,
      graph_id: None,
      graph: None,
    })
  }

  /// Compute mean of all elements
  pub fn mean(&self) -> Result<Tensor> {
    let sum: f64 = self.data.iter().sum();
    let count = self.data.len() as f64;
    let mean_val = sum / count;

    let result = Tensor {
      data: Array2::from_elem((1, 1), mean_val),
      grad: None,
      requires_grad: self.requires_grad,
      graph_id: None,
      graph: None,
    };

    // If this tensor is tracked in a computation graph, create a proper mean operation
    if let (Some(graph_weak), Some(_)) = (&self.graph, self.graph_id) {
      if let Some(_graph) = graph_weak.upgrade() {
        let op = OpBuilder::mean(Rc::new(self.clone()));
        let result_from_op = op.forward()?;
        return self.add_unary_to_graph(op, result_from_op);
      }
    }

    Ok(result)
  }

  pub fn add(&self, other: &Tensor) -> Result<Tensor> {
    let op = OpBuilder::add(Rc::new(self.clone()), Rc::new(other.clone()));
    let result = op.forward()?;
    self.add_to_graph(other, op, result)
  }

  pub fn sigmoid(&self) -> Result<Tensor> {
    let op = OpBuilder::sigmoid(Rc::new(self.clone()));
    let result = op.forward()?;
    self.add_unary_to_graph(op, result)
  }

  pub fn relu(&self) -> Result<Tensor> {
    let op = OpBuilder::relu(Rc::new(self.clone()));
    let result = op.forward()?;
    self.add_unary_to_graph(op, result)
  }

  pub fn tanh(&self) -> Result<Tensor> {
    let op = OpBuilder::tanh(Rc::new(self.clone()));
    let result = op.forward()?;
    self.add_unary_to_graph(op, result)
  }

  pub fn softmax(&self) -> Result<Tensor> {
    let op = OpBuilder::softmax(Rc::new(self.clone()));
    let result = op.forward()?;
    self.add_unary_to_graph(op, result)
  }
}

impl fmt::Debug for Tensor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Tensor")
      .field("shape", &self.shape())
      .field("data", &self.data)
      .field("requires_grad", &self.requires_grad)
      .field("has_grad", &self.grad.is_some())
      .field("is_tracked", &self.is_tracked())
      .field("graph_id", &self.graph_id)
      .finish()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::graph::ComputationGraph;
  use std::cell::RefCell;
  use std::rc::Rc;

  #[test]
  fn test_new() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let tensor = Tensor::new(data).expect("Failed to create tensor");

    assert_eq!(tensor.shape(), (2, 2));
    assert_eq!(tensor.data[[0, 0]], 1.0);
    assert_eq!(tensor.data[[0, 1]], 2.0);
    assert_eq!(tensor.data[[1, 0]], 3.0);
    assert_eq!(tensor.data[[1, 1]], 4.0);
    assert!(!tensor.requires_grad());
    assert!(tensor.grad.is_none());
  }

  #[test]
  fn test_new_inconsistent_rows() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]; // Second row has 3 elements
    let result = Tensor::new(data);

    assert!(result.is_err());
    assert!(matches!(
      result.unwrap_err(),
      TensorError::InvalidInput { .. }
    ));
  }

  #[test]
  fn test_zeros() {
    let tensor = Tensor::zeros(3, 4);

    assert_eq!(tensor.shape(), (3, 4));
    assert_eq!(tensor.len(), 12);

    for i in 0..3 {
      for j in 0..4 {
        assert_eq!(tensor.data[[i, j]], 0.0);
      }
    }
  }

  #[test]
  fn test_ones() {
    let tensor = Tensor::ones(2, 3);

    assert_eq!(tensor.shape(), (2, 3));
    assert_eq!(tensor.len(), 6);

    for i in 0..2 {
      for j in 0..3 {
        assert_eq!(tensor.data[[i, j]], 1.0);
      }
    }
  }

  #[test]
  fn test_random() {
    let tensor = Tensor::random(2, 3);

    assert_eq!(tensor.shape(), (2, 3));
    assert_eq!(tensor.len(), 6);

    for i in 0..2 {
      for j in 0..3 {
        let value = tensor.data[[i, j]];
        assert!((-1.0..1.0).contains(&value));
      }
    }
  }

  #[test]
  fn test_shape_and_dim() {
    let tensor = Tensor::zeros(5, 7);
    assert_eq!(tensor.shape(), (5, 7));
    assert_eq!(tensor.dim(), (5, 7));
    assert_eq!(tensor.len(), 35);
  }

  #[test]
  fn test_is_empty() {
    let empty_tensor = Tensor::zeros(0, 0);
    assert!(empty_tensor.is_empty());

    let non_empty_tensor = Tensor::zeros(1, 1);
    assert!(!non_empty_tensor.is_empty());
  }

  #[test]
  fn test_gradient_management() {
    let mut tensor = Tensor::ones(2, 2);

    assert!(!tensor.requires_grad());
    assert!(tensor.grad.is_none());

    tensor.set_requires_grad(true);
    assert!(tensor.requires_grad());
    assert!(tensor.grad.is_some());

    // Check gradient is initialized to zeros
    if let Some(ref grad) = tensor.grad {
      for i in 0..2 {
        for j in 0..2 {
          assert_eq!(grad[[i, j]], 0.0);
        }
      }
    }

    // Disable gradient computation
    tensor.set_requires_grad(false);
    assert!(!tensor.requires_grad());
    assert!(tensor.grad.is_none());
  }

  #[test]
  fn test_zero_grad() {
    let mut tensor = Tensor::ones(2, 2);
    tensor.set_requires_grad(true);

    // Modify gradient manually to test zero_grad
    tensor.set_grad_at(0, 0, 5.0).unwrap();
    tensor.set_grad_at(1, 1, 10.0).unwrap();

    tensor.zero_grad();

    if let Some(ref grad) = tensor.grad {
      for i in 0..2 {
        for j in 0..2 {
          assert_eq!(grad[[i, j]], 0.0);
        }
      }
    }
  }

  #[test]
  fn test_clone() {
    let tensor = Tensor::random(2, 3);
    let cloned = tensor.clone();

    assert_eq!(tensor.shape(), cloned.shape());
    assert_eq!(tensor.requires_grad(), cloned.requires_grad());

    for i in 0..2 {
      for j in 0..3 {
        assert_eq!(tensor.data[[i, j]], cloned.data[[i, j]]);
      }
    }
  }

  #[test]
  fn test_debug_format() {
    let tensor = Tensor::ones(2, 2);
    let debug_str = format!("{:?}", tensor);

    assert!(debug_str.contains("Tensor"));
    assert!(debug_str.contains("shape"));
    assert!(debug_str.contains("data"));
    assert!(debug_str.contains("requires_grad"));
    assert!(debug_str.contains("has_grad"));
  }

  #[test]
  fn test_tensor_with_computation_graph() {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    // Create tracked tensors
    let mut x = Tensor::new(vec![vec![2.0, 3.0]]).unwrap();
    let mut y = Tensor::new(vec![vec![4.0], vec![5.0]]).unwrap();

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    let x = x.with_graph(graph.clone());
    let y = y.with_graph(graph.clone());

    // Verify tracking
    assert!(x.is_tracked());
    assert!(y.is_tracked());

    // Perform operations that should build the graph
    let mut result = x.matmul(&y).unwrap();

    // Result should also be tracked
    assert!(result.is_tracked());

    // Verify the result value
    assert_eq!(result.data[[0, 0]], 23.0); // 2*4 + 3*5 = 23

    // Test backward pass (automatically syncs gradients)
    result.backward().unwrap();

    // Verify gradients are computed and accessible through the new grad() method
    assert!(x.grad().is_some());
    assert!(y.grad().is_some());

    // Check gradient values using helper methods
    use approx::assert_abs_diff_eq;
    assert_abs_diff_eq!(x.grad_at(0, 0).unwrap(), 4.0, epsilon = 1e-6); // dy/dx = y[0,0]
    assert_abs_diff_eq!(x.grad_at(0, 1).unwrap(), 5.0, epsilon = 1e-6); // dy/dx = y[1,0]
    assert_abs_diff_eq!(y.grad_at(0, 0).unwrap(), 2.0, epsilon = 1e-6); // dy/dy = x[0,0]
    assert_abs_diff_eq!(y.grad_at(1, 0).unwrap(), 3.0, epsilon = 1e-6); // dy/dy = x[0,1]
  }

  #[test]
  fn test_tensor_graph_integration_with_activation() {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    // Create a simple neural network computation: output = sigmoid(input * weight + bias)
    let mut input = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let mut weight = Tensor::new(vec![vec![0.5], vec![0.3]]).unwrap();
    let mut bias = Tensor::new(vec![vec![0.1]]).unwrap();

    input.set_requires_grad(true);
    weight.set_requires_grad(true);
    bias.set_requires_grad(true);

    let input = input.with_graph(graph.clone());
    let weight = weight.with_graph(graph.clone());
    let bias = bias.with_graph(graph.clone());

    // Forward pass
    let linear = input.matmul(&weight).unwrap(); // [1*0.5 + 2*0.3] = [1.1]
    let linear_bias = linear.add(&bias).unwrap(); // [1.1 + 0.1] = [1.2]
    let mut output = linear_bias.sigmoid().unwrap(); // sigmoid(1.2)

    assert!(output.is_tracked());

    // Backward pass (automatically syncs gradients)
    output.backward().unwrap();

    // All tensors should have gradients accessible through the new grad() method
    assert!(input.grad().is_some());
    assert!(weight.grad().is_some());
    assert!(bias.grad().is_some());

    // Gradients should be non-zero
    if let Some(input_grad) = input.grad() {
      assert!(input_grad[[0, 0]] != 0.0);
      assert!(input_grad[[0, 1]] != 0.0);
    }

    if let Some(weight_grad) = weight.grad() {
      assert!(weight_grad[[0, 0]] != 0.0);
      assert!(weight_grad[[1, 0]] != 0.0);
    }

    if let Some(bias_grad) = bias.grad() {
      assert!(bias_grad[[0, 0]] != 0.0);
    }
  }

  #[test]
  fn test_cross_graph_operation() {
    let graph_a = Rc::new(RefCell::new(ComputationGraph::new()));
    let graph_b = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut x = Tensor::ones(2, 2);
    x.set_requires_grad(true);
    let x = x.with_graph(graph_a.clone());

    let mut y = Tensor::ones(2, 2);
    y.set_requires_grad(true);
    let y = y.with_graph(graph_b.clone());

    // Should merge y into x's graph
    let z = x.add(&y).unwrap();

    // z should be in x's graph
    assert!(z.is_tracked());
    assert!(Tensor::same_graph(&x, &z));

    // y's original graph should still exist
    assert_eq!(graph_b.borrow().node_count(), 1); // Only y

    // x's graph should have x, y (as leaf), and z
    assert_eq!(graph_a.borrow().node_count(), 3);
  }

  #[test]
  fn test_same_graph_operation() {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut x = Tensor::ones(2, 2);
    x.set_requires_grad(true);
    let x = x.with_graph(graph.clone());

    let mut y = Tensor::ones(2, 2);
    y.set_requires_grad(true);
    let y = y.with_graph(graph.clone());

    // Should work without merging
    let z = x.add(&y).unwrap();

    assert!(z.is_tracked());
    assert!(Tensor::same_graph(&x, &y));
    assert!(Tensor::same_graph(&x, &z));

    // Graph should have x, y, z
    assert_eq!(graph.borrow().node_count(), 3);
  }

  #[test]
  fn test_untracked_with_tracked() {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut x = Tensor::ones(2, 2);
    x.set_requires_grad(true);
    let x = x.with_graph(graph.clone());

    let y = Tensor::ones(2, 2); // Untracked

    // y should be added to x's graph
    let z = x.add(&y).unwrap();

    assert!(z.is_tracked());
    assert!(Tensor::same_graph(&x, &z));

    // Graph should have x, y (as leaf), z
    assert_eq!(graph.borrow().node_count(), 3);
  }
}
