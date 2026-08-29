//! Error types for the multilayer perceptron library

use std::fmt;

/// Comprehensive error type for tensor operations and computation graph operations
#[derive(Debug)]
pub enum TensorError {
  /// Shape mismatch during tensor operations
  ShapeMismatch {
    operation: String,
    expected: (usize, usize),
    got: (usize, usize),
  },

  /// Tensor is not tracked in a computation graph when gradient computation is required
  NotTracked { tensor_id: Option<usize> },

  /// Computation graph has been dropped (Weak reference upgrade failed)
  GraphDropped,

  /// Invalid input data (e.g., inconsistent row lengths)
  InvalidInput { message: String },

  /// Gradient computation failed
  GradientError { message: String },

  /// No gradient available for tensor
  NoGradient,

  /// Dimension error for array operations
  DimensionError { message: String },

  /// Generic computational error
  ComputationError { message: String },

  /// Invalid value encountered during data processing
  InvalidValue(String),

  /// Dimension mismatch in array operations
  DimensionMismatch(String),

  /// IO error (file not found, read error, etc.)
  IoError(std::io::Error),

  /// CSV parsing error
  CsvError(csv::Error),
}

impl fmt::Display for TensorError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      TensorError::ShapeMismatch {
        operation,
        expected,
        got,
      } => {
        write!(
          f,
          "Shape mismatch in {}: expected {:?}, got {:?}",
          operation, expected, got
        )
      }
      TensorError::NotTracked { tensor_id } => {
        if let Some(id) = tensor_id {
          write!(
            f,
            "Tensor with ID {} is not tracked in a computation graph",
            id
          )
        } else {
          write!(f, "Tensor is not tracked in a computation graph")
        }
      }
      TensorError::GraphDropped => {
        write!(
          f,
          "Computation graph has been dropped and is no longer accessible"
        )
      }
      TensorError::InvalidInput { message } => {
        write!(f, "Invalid input data: {}", message)
      }
      TensorError::GradientError { message } => {
        write!(f, "Gradient computation error: {}", message)
      }
      TensorError::NoGradient => {
        write!(f, "No gradient available for this tensor")
      }
      TensorError::DimensionError { message } => {
        write!(f, "Dimension error: {}", message)
      }
      TensorError::ComputationError { message } => {
        write!(f, "Computation error: {}", message)
      }
      TensorError::InvalidValue(message) => {
        write!(f, "Invalid value: {}", message)
      }
      TensorError::DimensionMismatch(message) => {
        write!(f, "Dimension mismatch: {}", message)
      }
      TensorError::IoError(error) => {
        write!(f, "IO error: {}", error)
      }
      TensorError::CsvError(error) => {
        write!(f, "CSV error: {}", error)
      }
    }
  }
}

impl std::error::Error for TensorError {}

/// Convert ndarray shape errors into TensorError
impl From<ndarray::ShapeError> for TensorError {
  fn from(error: ndarray::ShapeError) -> Self {
    TensorError::DimensionError {
      message: format!("Array shape error: {}", error),
    }
  }
}

/// Convert String errors into TensorError for backward compatibility
impl From<String> for TensorError {
  fn from(message: String) -> Self {
    TensorError::ComputationError { message }
  }
}

/// Convert IO errors into TensorError
impl From<std::io::Error> for TensorError {
  fn from(error: std::io::Error) -> Self {
    TensorError::IoError(error)
  }
}

/// Convert CSV errors into TensorError
impl From<csv::Error> for TensorError {
  fn from(error: csv::Error) -> Self {
    TensorError::CsvError(error)
  }
}

/// Result type alias for operations that can fail with TensorError
pub type Result<T> = std::result::Result<T, TensorError>;
