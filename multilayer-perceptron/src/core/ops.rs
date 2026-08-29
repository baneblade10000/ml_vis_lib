use super::error::{Result, TensorError};
use super::tensor::Tensor;
use ndarray::Array2;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum OpNode {
  MatMul {
    input_a: Rc<Tensor>,
    input_b: Rc<Tensor>,
  },
  Add {
    input_a: Rc<Tensor>,
    input_b: Rc<Tensor>,
  },
  Sigmoid {
    input: Rc<Tensor>,
  },
  ReLU {
    input: Rc<Tensor>,
  },
  Tanh {
    input: Rc<Tensor>,
  },
  Softmax {
    input: Rc<Tensor>,
  },
  Mean {
    input: Rc<Tensor>,
  },
}

impl OpNode {
  pub fn forward(&self) -> Result<Tensor> {
    match self {
      OpNode::MatMul { input_a, input_b } => Self::forward_matmul(
        &input_a.data,
        &input_b.data,
        input_a.requires_grad || input_b.requires_grad,
      ),
      OpNode::Add { input_a, input_b } => Self::forward_add(
        &input_a.data,
        &input_b.data,
        input_a.requires_grad || input_b.requires_grad,
      ),
      OpNode::Sigmoid { input } => Self::forward_sigmoid(&input.data, input.requires_grad),
      OpNode::ReLU { input } => Self::forward_relu(&input.data, input.requires_grad),
      OpNode::Tanh { input } => Self::forward_tanh(&input.data, input.requires_grad),
      OpNode::Softmax { input } => Self::forward_softmax(&input.data, input.requires_grad),
      OpNode::Mean { input } => Self::forward_mean(&input.data, input.requires_grad),
    }
  }

  pub fn backward(&self, grad_output: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    match self {
      OpNode::MatMul { input_a, input_b } => {
        Self::backward_matmul(grad_output, &input_a.data, &input_b.data)
      }
      OpNode::Add { input_a, input_b } => {
        Self::backward_add(grad_output, &input_a.data, &input_b.data)
      }
      OpNode::Sigmoid { input } => Self::backward_sigmoid(grad_output, &input.data),
      OpNode::ReLU { input } => Self::backward_relu(grad_output, &input.data),
      OpNode::Tanh { input } => Self::backward_tanh(grad_output, &input.data),
      OpNode::Softmax { input } => Self::backward_softmax(grad_output, &input.data),
      OpNode::Mean { input } => Self::backward_mean(grad_output, &input.data),
    }
  }

  fn forward_matmul(a: &Array2<f64>, b: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    let a_shape = a.dim();
    let b_shape = b.dim();

    if a_shape.1 != b_shape.0 {
      return Err(TensorError::ShapeMismatch {
        operation: "Matrix multiplication".to_string(),
        expected: (a_shape.1, b_shape.0),
        got: (b_shape.0, b_shape.1),
      });
    }

    let result = a.dot(b);
    let result_dim = result.dim();
    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(result_dim))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_add(a: &Array2<f64>, b: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    let a_shape = a.dim();
    let b_shape = b.dim();

    // Check if broadcasting is possible and compute output shape
    let output_shape = Self::broadcast_shapes(a_shape, b_shape)?;

    // Perform broadcasting addition
    let result = Self::broadcast_add(a, b, output_shape)?;

    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(output_shape))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_sigmoid(input: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    // \frac{1}{1+e^{-x}}
    let result = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let result_dim = result.dim();
    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(result_dim))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_relu(input: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    let result = input.mapv(|x| x.max(0.0));
    let result_dim = result.dim();
    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(result_dim))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_tanh(input: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    // \frac{e^x - e^{-x}}{e^x+e^{-x}}
    let result = input.mapv(|x| x.tanh());
    let result_dim = result.dim();
    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(result_dim))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_softmax(input: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    // Softmax with numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    // Apply softmax along the last axis (row-wise for each sample)
    let result = Array2::from_shape_fn(input.dim(), |(i, j)| {
      let row = input.row(i);
      let max_val = row.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
      let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
      ((input[[i, j]] - max_val).exp()) / exp_sum
    });

    let result_dim = result.dim();
    Ok(Tensor {
      data: result,
      grad: if requires_grad {
        Some(Array2::zeros(result_dim))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn forward_mean(input: &Array2<f64>, requires_grad: bool) -> Result<Tensor> {
    let sum: f64 = input.iter().sum();
    let count = input.len() as f64;
    let mean_val = sum / count;

    Ok(Tensor {
      data: Array2::from_elem((1, 1), mean_val),
      grad: if requires_grad {
        Some(Array2::zeros((1, 1)))
      } else {
        None
      },
      requires_grad,
      graph_id: None,
      graph: None,
    })
  }

  fn backward_matmul(
    grad_output: &Array2<f64>,
    input_a: &Array2<f64>,
    input_b: &Array2<f64>,
  ) -> Result<Vec<Array2<f64>>> {
    // Matrix multiplication C = AB where A is (m×k), B is (k×n), C is (m×n)
    //
    // Matrix derivative definition: if L = scalar loss function
    // ∂L/∂A[i,j] = sum over all paths from A[i,j] to L
    //
    // Since C[i,p] = Σ_q A[i,q] * B[q,p], we have ∂C[i,p]/∂A[i,j] = B[j,p]
    // By chain rule: ∂L/∂A[i,j] = Σ_p (∂L/∂C[i,p]) * B[j,p] = (∂L/∂C * B^T)[i,j]
    let grad_a = grad_output.dot(&input_b.t());

    // Similarly: ∂C[i,p]/∂B[j,q] = A[i,j] * δ(j,q) * δ(p,q) = A[i,j] if q=j and p=q
    // So: ∂L/∂B[j,p] = Σ_i (∂L/∂C[i,p]) * A[i,j] = (A^T * ∂L/∂C)[j,p]
    let grad_b = input_a.t().dot(grad_output);

    Ok(vec![grad_a, grad_b])
  }

  fn backward_add(
    grad_output: &Array2<f64>,
    input_a: &Array2<f64>,
    input_b: &Array2<f64>,
  ) -> Result<Vec<Array2<f64>>> {
    // For broadcasting, we need to sum gradients along broadcasted dimensions
    let grad_a = Self::reduce_broadcasted_gradient(grad_output, input_a.dim())?;
    let grad_b = Self::reduce_broadcasted_gradient(grad_output, input_b.dim())?;
    Ok(vec![grad_a, grad_b])
  }

  fn backward_sigmoid(grad_output: &Array2<f64>, input: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    // $\frac{\partial \sigma}{\partial x} = \sigma(x)(1 - \sigma(x))$
    let sigmoid_output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let sigmoid_grad = sigmoid_output.mapv(|s| s * (1.0 - s));
    let grad_input = grad_output * &sigmoid_grad;
    Ok(vec![grad_input])
  }

  fn backward_relu(grad_output: &Array2<f64>, input: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    // $\frac{\partial \text{ReLU}}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$
    let relu_grad = input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
    let grad_input = grad_output * &relu_grad;
    Ok(vec![grad_input])
  }

  fn backward_tanh(grad_output: &Array2<f64>, input: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    // $\frac{\partial \tanh}{\partial x} = 1 - \tanh^2(x)$
    let tanh_output = input.mapv(|x| x.tanh());
    let tanh_grad = tanh_output.mapv(|t| 1.0 - t * t);
    let grad_input = grad_output * &tanh_grad;
    Ok(vec![grad_input])
  }

  fn backward_softmax(grad_output: &Array2<f64>, input: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    // Softmax backward pass is more complex due to cross-dependencies
    // For softmax S_i = exp(x_i) / sum_j(exp(x_j)), the Jacobian is:
    // ∂S_i/∂x_j = S_i * (δ_ij - S_j) where δ_ij is Kronecker delta
    let mut grad_input = Array2::zeros(input.dim());

    for i in 0..input.nrows() {
      let row = input.row(i);
      let grad_row = grad_output.row(i);

      // Compute softmax for this row
      let max_val = row.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
      let exp_vals: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
      let exp_sum: f64 = exp_vals.iter().sum();
      let softmax_vals: Vec<f64> = exp_vals.iter().map(|&exp_x| exp_x / exp_sum).collect();

      // Compute gradient for this row
      for j in 0..input.ncols() {
        let mut grad_val = 0.0;
        for k in 0..input.ncols() {
          if j == k {
            grad_val += grad_row[k] * softmax_vals[j] * (1.0 - softmax_vals[k]);
          } else {
            grad_val += grad_row[k] * softmax_vals[j] * (-softmax_vals[k]);
          }
        }
        grad_input[[i, j]] = grad_val;
      }
    }

    Ok(vec![grad_input])
  }

  fn backward_mean(grad_output: &Array2<f64>, input: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
    // For mean, the gradient is uniformly distributed across all input elements
    // ∂(mean(x))/∂x_i = 1/N where N is the total number of elements
    let total_elements = input.len() as f64;
    let grad_per_element = grad_output[[0, 0]] / total_elements;
    let grad_input = Array2::from_elem(input.dim(), grad_per_element);

    Ok(vec![grad_input])
  }

  /// Compute the output shape for broadcasting two arrays
  fn broadcast_shapes(a_shape: (usize, usize), b_shape: (usize, usize)) -> Result<(usize, usize)> {
    // Broadcasting rules (similar to NumPy):
    // 1. Compare dimensions from right to left
    // 2. Dimensions are compatible if they are equal or one of them is 1

    let (a_rows, a_cols) = a_shape;
    let (b_rows, b_cols) = b_shape;

    // Columns must match exactly for stability (bias broadcasting only occurs across rows)
    let output_cols = if a_cols == b_cols {
      a_cols
    } else {
      return Err(TensorError::ShapeMismatch {
        operation: "Broadcasting Addition".to_string(),
        expected: a_shape,
        got: b_shape,
      });
    };

    // Check row compatibility
    let output_rows = if a_rows == b_rows {
      a_rows
    } else if a_rows == 1 {
      b_rows
    } else if b_rows == 1 {
      a_rows
    } else {
      return Err(TensorError::ShapeMismatch {
        operation: "Broadcasting Addition".to_string(),
        expected: a_shape,
        got: b_shape,
      });
    };

    Ok((output_rows, output_cols))
  }

  /// Perform broadcasted addition
  fn broadcast_add(
    a: &Array2<f64>,
    b: &Array2<f64>,
    output_shape: (usize, usize),
  ) -> Result<Array2<f64>> {
    let (output_rows, output_cols) = output_shape;
    let mut result = Array2::zeros(output_shape);

    for i in 0..output_rows {
      for j in 0..output_cols {
        let a_i = if a.nrows() == 1 { 0 } else { i };
        let a_j = if a.ncols() == 1 { 0 } else { j };
        let b_i = if b.nrows() == 1 { 0 } else { i };
        let b_j = if b.ncols() == 1 { 0 } else { j };

        result[[i, j]] = a[[a_i, a_j]] + b[[b_i, b_j]];
      }
    }

    Ok(result)
  }

  /// Reduce broadcasted gradient to match original input shape
  fn reduce_broadcasted_gradient(
    grad_output: &Array2<f64>,
    input_shape: (usize, usize),
  ) -> Result<Array2<f64>> {
    let (input_rows, input_cols) = input_shape;
    let (grad_rows, grad_cols) = grad_output.dim();

    // If shapes match, no reduction needed
    if input_rows == grad_rows && input_cols == grad_cols {
      return Ok(grad_output.clone());
    }

    let mut result = Array2::zeros(input_shape);

    // Sum along broadcasted dimensions
    for i in 0..input_rows {
      for j in 0..input_cols {
        let mut grad_sum = 0.0;

        // Sum over all positions that this input element affected
        let row_start = if input_rows == 1 { 0 } else { i };
        let row_end = if input_rows == 1 { grad_rows } else { i + 1 };
        let col_start = if input_cols == 1 { 0 } else { j };
        let col_end = if input_cols == 1 { grad_cols } else { j + 1 };

        for gi in row_start..row_end {
          for gj in col_start..col_end {
            grad_sum += grad_output[[gi, gj]];
          }
        }

        result[[i, j]] = grad_sum;
      }
    }

    Ok(result)
  }
}

pub struct OpBuilder;

impl OpBuilder {
  pub fn matmul(input_a: Rc<Tensor>, input_b: Rc<Tensor>) -> OpNode {
    OpNode::MatMul { input_a, input_b }
  }

  pub fn add(input_a: Rc<Tensor>, input_b: Rc<Tensor>) -> OpNode {
    OpNode::Add { input_a, input_b }
  }

  pub fn sigmoid(input: Rc<Tensor>) -> OpNode {
    OpNode::Sigmoid { input }
  }

  pub fn relu(input: Rc<Tensor>) -> OpNode {
    OpNode::ReLU { input }
  }

  pub fn tanh(input: Rc<Tensor>) -> OpNode {
    OpNode::Tanh { input }
  }

  pub fn softmax(input: Rc<Tensor>) -> OpNode {
    OpNode::Softmax { input }
  }

  pub fn mean(input: Rc<Tensor>) -> OpNode {
    OpNode::Mean { input }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::tensor::Tensor;
  use approx::assert_abs_diff_eq;
  use std::rc::Rc;

  #[test]
  fn test_matmul_forward() {
    let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

    let op = OpBuilder::matmul(Rc::new(a), Rc::new(b));
    let result = op.forward().unwrap();

    assert_eq!(result.data[[0, 0]], 19.0);
    assert_eq!(result.data[[0, 1]], 22.0);
    assert_eq!(result.data[[1, 0]], 43.0);
    assert_eq!(result.data[[1, 1]], 50.0);
  }

  #[test]
  fn test_matmul_dimension_mismatch() {
    let a = Tensor::new(vec![vec![1.0, 2.0]]).unwrap(); // 1x2
    let b = Tensor::new(vec![vec![1.0], vec![2.0], vec![3.0]]).unwrap(); // 3x1

    let op = OpBuilder::matmul(Rc::new(a), Rc::new(b));
    let result = op.forward();

    assert!(result.is_err());
    assert!(matches!(
      result.unwrap_err(),
      TensorError::ShapeMismatch { .. }
    ));
  }

  #[test]
  fn test_matmul_backward() {
    let a_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b_data = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let a = Tensor::new(a_data).unwrap();
    let b = Tensor::new(b_data).unwrap();

    let op = OpBuilder::matmul(Rc::new(a.clone()), Rc::new(b.clone()));

    let grad_output = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 2); // Two inputs

    assert_eq!(gradients[0][[0, 0]], 11.0);
    assert_eq!(gradients[0][[0, 1]], 15.0);
    assert_eq!(gradients[0][[1, 0]], 11.0);
    assert_eq!(gradients[0][[1, 1]], 15.0);

    // a^T = [[1, 3], [2, 4]]
    // a^T * grad_output = [[1*1+3*1, 1*1+3*1], [2*1+4*1, 2*1+4*1]] = [[4, 4], [6, 6]]
    assert_eq!(gradients[1][[0, 0]], 4.0);
    assert_eq!(gradients[1][[0, 1]], 4.0);
    assert_eq!(gradients[1][[1, 0]], 6.0);
    assert_eq!(gradients[1][[1, 1]], 6.0);
  }

  #[test]
  fn test_add_forward() {
    let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

    let op = OpBuilder::add(Rc::new(a), Rc::new(b));
    let result = op.forward().unwrap();

    assert_eq!(result.data[[0, 0]], 6.0); // 1 + 5
    assert_eq!(result.data[[0, 1]], 8.0); // 2 + 6
    assert_eq!(result.data[[1, 0]], 10.0); // 3 + 7
    assert_eq!(result.data[[1, 1]], 12.0); // 4 + 8
  }

  #[test]
  fn test_add_shape_mismatch() {
    let a = Tensor::new(vec![vec![1.0, 2.0]]).unwrap(); // 1x2
    let b = Tensor::new(vec![vec![1.0], vec![2.0]]).unwrap(); // 2x1

    let op = OpBuilder::add(Rc::new(a), Rc::new(b));
    let result = op.forward();

    assert!(result.is_err());
    assert!(matches!(
      result.unwrap_err(),
      TensorError::ShapeMismatch { .. }
    ));
  }

  #[test]
  fn test_add_backward() {
    let a = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let b = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();

    let op = OpBuilder::add(Rc::new(a), Rc::new(b));
    let grad_output = Array2::from_shape_vec((1, 2), vec![5.0, 6.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 2);

    // For addition, gradients pass through unchanged
    assert_eq!(gradients[0], grad_output);
    assert_eq!(gradients[1], grad_output);
  }

  #[test]
  fn test_sigmoid_forward() {
    let input = Tensor::new(vec![vec![0.0, 1.0], vec![-1.0, 2.0]]).unwrap();

    let op = OpBuilder::sigmoid(Rc::new(input));
    let result = op.forward().unwrap();

    // Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.268, Sigmoid(2) ≈ 0.881
    assert_abs_diff_eq!(result.data[[0, 0]], 0.5, epsilon = 1e-3);
    assert_abs_diff_eq!(result.data[[0, 1]], 0.7310585786300049, epsilon = 1e-3);
    assert_abs_diff_eq!(result.data[[1, 0]], 0.2689414213699951, epsilon = 1e-3);
    assert_abs_diff_eq!(result.data[[1, 1]], 0.8807970779778823, epsilon = 1e-3);
  }

  #[test]
  fn test_sigmoid_backward() {
    let input = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();

    let op = OpBuilder::sigmoid(Rc::new(input));
    let grad_output = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 1);

    // Sigmoid derivative at 0: 0.5 * (1 - 0.5) = 0.25
    // Sigmoid derivative at 1: sigmoid(1) * (1 - sigmoid(1)) ≈ 0.731 * 0.269 ≈ 0.196
    assert_abs_diff_eq!(gradients[0][[0, 0]], 0.25, epsilon = 1e-3);
    assert_abs_diff_eq!(gradients[0][[0, 1]], 0.19661193324148185, epsilon = 1e-3);
  }

  #[test]
  fn test_relu_forward() {
    let input = Tensor::new(vec![vec![-1.0, 0.0], vec![1.0, 2.0]]).unwrap();

    let op = OpBuilder::relu(Rc::new(input));
    let result = op.forward().unwrap();

    assert_eq!(result.data[[0, 0]], 0.0); // max(-1, 0) = 0
    assert_eq!(result.data[[0, 1]], 0.0); // max(0, 0) = 0
    assert_eq!(result.data[[1, 0]], 1.0); // max(1, 0) = 1
    assert_eq!(result.data[[1, 1]], 2.0); // max(2, 0) = 2
  }

  #[test]
  fn test_relu_backward() {
    let input = Tensor::new(vec![vec![-1.0, 0.0], vec![1.0, 2.0]]).unwrap();

    let op = OpBuilder::relu(Rc::new(input));
    let grad_output = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 1);

    // ReLU derivative: 1 if x > 0, 0 otherwise
    assert_eq!(gradients[0][[0, 0]], 0.0); // input was -1
    assert_eq!(gradients[0][[0, 1]], 0.0); // input was 0
    assert_eq!(gradients[0][[1, 0]], 1.0); // input was 1
    assert_eq!(gradients[0][[1, 1]], 1.0); // input was 2
  }

  #[test]
  fn test_tanh_forward() {
    let input = Tensor::new(vec![vec![-1.0, 0.0], vec![1.0, 2.0]]).unwrap();

    let op = OpBuilder::tanh(Rc::new(input));
    let result = op.forward().unwrap();

    assert_abs_diff_eq!(result.data[[0, 0]], (-1.0f64).tanh(), epsilon = 1e-6);
    assert_abs_diff_eq!(result.data[[0, 1]], 0.0f64.tanh(), epsilon = 1e-6);
    assert_abs_diff_eq!(result.data[[1, 0]], 1.0f64.tanh(), epsilon = 1e-6);
    assert_abs_diff_eq!(result.data[[1, 1]], 2.0f64.tanh(), epsilon = 1e-6);
  }

  #[test]
  fn test_tanh_backward() {
    let input = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();

    let op = OpBuilder::tanh(Rc::new(input));
    let grad_output = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 1);

    // tanh derivative: 1 - tanh^2(x)
    let tanh_0 = 0.0f64.tanh();
    let tanh_1 = 1.0f64.tanh();

    assert_abs_diff_eq!(gradients[0][[0, 0]], 1.0 - tanh_0 * tanh_0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradients[0][[0, 1]], 1.0 - tanh_1 * tanh_1, epsilon = 1e-6);
  }

  #[test]
  fn test_tensor_operations() {
    // Test tensor operation methods
    let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

    // Test matmul
    let result = a.matmul(&b).unwrap();
    assert_eq!(result.data[[0, 0]], 19.0);
    assert_eq!(result.data[[1, 1]], 50.0);

    // Test add
    let result = a.add(&b).unwrap();
    assert_eq!(result.data[[0, 0]], 6.0);
    assert_eq!(result.data[[1, 1]], 12.0);

    // Test sigmoid
    let input = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();
    let result = input.sigmoid().unwrap();
    assert_abs_diff_eq!(result.data[[0, 0]], 0.5, epsilon = 1e-3);

    // Test relu
    let input = Tensor::new(vec![vec![-1.0, 2.0]]).unwrap();
    let result = input.relu().unwrap();
    assert_eq!(result.data[[0, 0]], 0.0);
    assert_eq!(result.data[[0, 1]], 2.0);

    // Test tanh
    let input = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();
    let result = input.tanh().unwrap();
    assert_abs_diff_eq!(result.data[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.data[[0, 1]], 1.0f64.tanh(), epsilon = 1e-6);

    // Test softmax
    let input = Tensor::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
    let result = input.softmax().unwrap();

    // Softmax([1,2,3]) should give probabilities that sum to 1
    let sum: f64 = (0..3).map(|i| result.data[[0, i]]).sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

    // Values should be in ascending order since input is [1,2,3]
    assert!(result.data[[0, 0]] < result.data[[0, 1]]);
    assert!(result.data[[0, 1]] < result.data[[0, 2]]);
  }

  #[test]
  fn test_softmax_forward() {
    let input = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]]).unwrap();

    let op = OpBuilder::softmax(Rc::new(input));
    let result = op.forward().unwrap();

    // Check that each row sums to 1
    for i in 0..result.data.nrows() {
      let row_sum: f64 = (0..result.data.ncols()).map(|j| result.data[[i, j]]).sum();
      assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
    }

    // For uniform input [0,0,0], softmax should give [1/3, 1/3, 1/3]
    for j in 0..3 {
      assert_abs_diff_eq!(result.data[[1, j]], 1.0 / 3.0, epsilon = 1e-6);
    }
  }

  #[test]
  fn test_softmax_backward() {
    let input = Tensor::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();

    let op = OpBuilder::softmax(Rc::new(input.clone()));
    let grad_output = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();

    let gradients = op.backward(&grad_output).unwrap();

    assert_eq!(gradients.len(), 1);

    // The gradient should have the correct shape
    assert_eq!(gradients[0].dim(), input.data.dim());

    // For softmax, gradients should sum to 0 along each row (due to constraint that softmax sums to 1)
    let grad_sum: f64 = (0..3).map(|j| gradients[0][[0, j]]).sum();
    assert_abs_diff_eq!(grad_sum, 0.0, epsilon = 1e-6);
  }
}
