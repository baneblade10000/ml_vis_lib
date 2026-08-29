//! WebAssembly bindings for multilayer perceptron
//!
//! This module provides JavaScript-compatible wrappers around the core
//! tensor operations and neural network functionality.

use crate::prelude::*;
use js_sys::Float64Array;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

type JsResult<T> = std::result::Result<T, JsValue>;

/// Log helper functions for WASM environment
fn log_info(message: &str) {
  console::log_1(&JsValue::from_str(message));
}

fn log_error(message: &str) {
  console::error_1(&JsValue::from_str(message));
}

fn log_debug(message: &str) {
  console::debug_1(&JsValue::from_str(message));
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum JsOptimizerType {
  GD,
  SGD,
  SGDMomentum,
  RMSProp,
  Adam,
}

#[derive(Debug, Clone)]
struct OptimizerConfigData {
  optimizer_type: JsOptimizerType,
  learning_rate: f64,
}

impl OptimizerConfigData {
  fn ensure_valid(&self) -> JsResult<()> {
    if self.learning_rate <= 0.0 {
      return Err(JsValue::from_str("Learning rate must be positive"));
    }
    Ok(())
  }

  fn build_optimizer(&self) -> Box<dyn Optimizer> {
    match self.optimizer_type {
      JsOptimizerType::GD => Box::new(GradientDescent::new(self.learning_rate)),
      JsOptimizerType::SGD => Box::new(SGD::new(self.learning_rate)),
      JsOptimizerType::SGDMomentum => Box::new(SGDMomentum::new(self.learning_rate, 0.9)),
      JsOptimizerType::RMSProp => Box::new(RMSProp::new(self.learning_rate)),
      JsOptimizerType::Adam => Box::new(Adam::new(self.learning_rate)),
    }
  }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct JsOptimizerConfig {
  inner: OptimizerConfigData,
}

impl Clone for JsOptimizerConfig {
  fn clone(&self) -> Self {
    Self {
      inner: self.inner.clone(),
    }
  }
}

#[wasm_bindgen]
impl JsOptimizerConfig {
  #[wasm_bindgen(constructor)]
  pub fn new(optimizer_type: JsOptimizerType, learning_rate: f64) -> JsResult<JsOptimizerConfig> {
    let config = OptimizerConfigData {
      optimizer_type,
      learning_rate,
    };
    config.ensure_valid()?;
    Ok(Self { inner: config })
  }

  #[wasm_bindgen(getter)]
  pub fn optimizer_type(&self) -> JsOptimizerType {
    self.inner.optimizer_type
  }

  #[wasm_bindgen(getter)]
  pub fn learning_rate(&self) -> f64 {
    self.inner.learning_rate
  }

  pub(crate) fn inner(&self) -> &OptimizerConfigData {
    &self.inner
  }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum JsRegularizationType {
  None,
  L1,
  L2,
  ElasticNet,
}

#[derive(Debug, Clone)]
struct RegularizationConfigData {
  regularization_type: JsRegularizationType,
  l1_lambda: f64,
  l2_lambda: f64,
}

impl RegularizationConfigData {
  fn ensure_valid(&self) -> JsResult<()> {
    if self.l1_lambda < 0.0 {
      return Err(JsValue::from_str("L1 lambda must be non-negative"));
    }
    if self.l2_lambda < 0.0 {
      return Err(JsValue::from_str("L2 lambda must be non-negative"));
    }
    Ok(())
  }

  fn to_domain(&self) -> Option<RegularizationConfig> {
    match self.regularization_type {
      JsRegularizationType::None => None,
      JsRegularizationType::L1 => Some(RegularizationConfig::l1_only(self.l1_lambda)),
      JsRegularizationType::L2 => Some(RegularizationConfig::l2_only(self.l2_lambda)),
      JsRegularizationType::ElasticNet => Some(RegularizationConfig::elastic_net(
        self.l1_lambda,
        self.l2_lambda,
      )),
    }
  }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct JsRegularizationConfig {
  inner: RegularizationConfigData,
}

impl Clone for JsRegularizationConfig {
  fn clone(&self) -> Self {
    Self {
      inner: self.inner.clone(),
    }
  }
}

#[wasm_bindgen]
impl JsRegularizationConfig {
  #[wasm_bindgen(constructor)]
  pub fn new(
    reg_type: JsRegularizationType,
    l1_lambda: f64,
    l2_lambda: f64,
  ) -> JsResult<JsRegularizationConfig> {
    let config = RegularizationConfigData {
      regularization_type: reg_type,
      l1_lambda,
      l2_lambda,
    };
    config.ensure_valid()?;
    Ok(Self { inner: config })
  }

  pub(crate) fn to_domain(&self) -> Option<RegularizationConfig> {
    self.inner.to_domain()
  }

  pub(crate) fn inner(&self) -> &RegularizationConfigData {
    &self.inner
  }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsTaskType {
  BinaryClassification,
  MultiClassification,
  Regression,
}

impl From<JsTaskType> for TaskKind {
  fn from(value: JsTaskType) -> Self {
    match value {
      JsTaskType::BinaryClassification => TaskKind::BinaryClassification,
      JsTaskType::MultiClassification => TaskKind::MultiClassification,
      JsTaskType::Regression => TaskKind::Regression,
    }
  }
}

#[derive(Debug, Clone)]
struct ModelConfigData {
  layers: Vec<usize>,
  activation_fn: String,
  task_type: JsTaskType,
}

#[wasm_bindgen]
pub struct JsModelConfig {
  inner: ModelConfigData,
}

impl Clone for JsModelConfig {
  fn clone(&self) -> Self {
    Self {
      inner: self.inner.clone(),
    }
  }
}

#[wasm_bindgen]
impl JsModelConfig {
  #[wasm_bindgen(constructor)]
  pub fn new(
    layers: js_sys::Array,
    activation_fn: String,
    task_type: JsTaskType,
  ) -> JsResult<JsModelConfig> {
    if layers.length() < 2 {
      return Err(JsValue::from_str(
        "Model must contain at least an input and an output layer",
      ));
    }

    let mut parsed_layers = Vec::with_capacity(layers.length() as usize);
    for idx in 0..layers.length() {
      let value = layers.get(idx);
      let layer_size = value
        .as_f64()
        .ok_or_else(|| JsValue::from_str("Layer sizes must be numeric"))?;
      if layer_size <= 0.0 {
        return Err(JsValue::from_str("Layer sizes must be positive"));
      }
      parsed_layers.push(layer_size as usize);
    }

    Ok(Self {
      inner: ModelConfigData {
        layers: parsed_layers,
        activation_fn: activation_fn.to_lowercase(),
        task_type,
      },
    })
  }

  #[wasm_bindgen(getter)]
  pub fn task_type(&self) -> JsTaskType {
    self.inner.task_type
  }

  pub(crate) fn layers(&self) -> &[usize] {
    &self.inner.layers
  }

  pub(crate) fn activation(&self) -> &str {
    &self.inner.activation_fn
  }

  pub(crate) fn to_inner(&self) -> ModelConfigData {
    self.inner.clone()
  }
}

#[derive(Debug, Clone)]
struct TrainingConfigData {
  epochs: usize,
  batch_size: usize,
  validation_split: f64,
  optimizer_config: JsOptimizerConfig,
  regularization_config: Option<JsRegularizationConfig>,
  early_stopping_patience: usize,
  early_stopping_min_delta: f64,
  enable_early_stopping: bool,
}

#[wasm_bindgen]
pub struct JsTrainingConfig {
  inner: TrainingConfigData,
}

impl Clone for JsTrainingConfig {
  fn clone(&self) -> Self {
    Self {
      inner: self.inner.clone(),
    }
  }
}

#[wasm_bindgen]
impl JsTrainingConfig {
  #[wasm_bindgen(constructor)]
  pub fn new(
    epochs: usize,
    batch_size: usize,
    validation_split: f64,
    optimizer_config: JsOptimizerConfig,
    regularization_config: Option<JsRegularizationConfig>,
  ) -> JsResult<JsTrainingConfig> {
    if epochs == 0 {
      return Err(JsValue::from_str("Epoch count must be at least 1"));
    }
    if batch_size == 0 {
      return Err(JsValue::from_str("Batch size must be at least 1"));
    }
    if validation_split < 0.0 || validation_split >= 1.0 {
      return Err(JsValue::from_str(
        "Validation split must be in the range [0.0, 1.0)",
      ));
    }

    Ok(Self {
      inner: TrainingConfigData {
        epochs,
        batch_size,
        validation_split,
        optimizer_config,
        regularization_config,
        early_stopping_patience: 0,
        early_stopping_min_delta: 0.0001,
        enable_early_stopping: false,
      },
    })
  }

  #[wasm_bindgen(js_name = "newWithEarlyStopping")]
  pub fn new_with_early_stopping(
    epochs: usize,
    batch_size: usize,
    validation_split: f64,
    optimizer_config: JsOptimizerConfig,
    regularization_config: Option<JsRegularizationConfig>,
    enable_early_stopping: bool,
    early_stopping_patience: usize,
    early_stopping_min_delta: f64,
  ) -> JsResult<JsTrainingConfig> {
    if epochs == 0 {
      return Err(JsValue::from_str("Epoch count must be at least 1"));
    }
    if batch_size == 0 {
      return Err(JsValue::from_str("Batch size must be at least 1"));
    }
    if validation_split < 0.0 || validation_split >= 1.0 {
      return Err(JsValue::from_str(
        "Validation split must be in the range [0.0, 1.0)",
      ));
    }
    if early_stopping_min_delta < 0.0 {
      return Err(JsValue::from_str("Early stopping min delta must be >= 0"));
    }

    Ok(Self {
      inner: TrainingConfigData {
        epochs,
        batch_size,
        validation_split,
        optimizer_config,
        regularization_config,
        early_stopping_patience,
        early_stopping_min_delta,
        enable_early_stopping,
      },
    })
  }

  pub(crate) fn epochs(&self) -> usize {
    self.inner.epochs
  }

  pub(crate) fn batch_size(&self) -> usize {
    self.inner.batch_size
  }

  pub(crate) fn validation_split(&self) -> f64 {
    self.inner.validation_split
  }

  pub(crate) fn optimizer_config(&self) -> &JsOptimizerConfig {
    &self.inner.optimizer_config
  }

  pub(crate) fn regularization_config(&self) -> Option<&JsRegularizationConfig> {
    self.inner.regularization_config.as_ref()
  }

  pub(crate) fn early_stopping_patience(&self) -> usize {
    self.inner.early_stopping_patience
  }

  pub(crate) fn early_stopping_min_delta(&self) -> f64 {
    self.inner.early_stopping_min_delta
  }

  pub(crate) fn enable_early_stopping(&self) -> bool {
    self.inner.enable_early_stopping
  }

  pub(crate) fn to_inner(&self) -> TrainingConfigData {
    self.inner.clone()
  }
}

// Enable better panic messages in debug mode
#[cfg(feature = "console_error_panic_hook")]
#[wasm_bindgen(start)]
pub fn main() {
  console_error_panic_hook::set_once();
}

/// JavaScript-compatible wrapper for Tensor
#[wasm_bindgen]
pub struct JsTensor {
  inner: Tensor,
}

#[wasm_bindgen]
impl JsTensor {
  /// Create a new tensor from a JavaScript Float64Array
  ///
  /// # Arguments
  /// * `data` - Flattened tensor data as Float64Array
  /// * `rows` - Number of rows
  /// * `cols` - Number of columns
  #[wasm_bindgen(constructor)]
  pub fn new(data: Float64Array, rows: usize, cols: usize) -> JsResult<JsTensor> {
    let vec_data: Vec<f64> = data.to_vec();

    if vec_data.len() != rows * cols {
      return Err(JsValue::from_str(&format!(
        "Data length {} doesn't match dimensions {}x{}",
        vec_data.len(),
        rows,
        cols
      )));
    }

    let mut tensor_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        row.push(vec_data[i * cols + j]);
      }
      tensor_data.push(row);
    }

    let tensor = Tensor::new(tensor_data)
      .map_err(|e| JsValue::from_str(&format!("Failed to create tensor: {}", e)))?;

    Ok(JsTensor { inner: tensor })
  }

  /// Create a tensor filled with zeros
  #[wasm_bindgen]
  pub fn zeros(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::zeros(rows, cols),
    }
  }

  /// Create a tensor filled with ones
  #[wasm_bindgen]
  pub fn ones(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::ones(rows, cols),
    }
  }

  /// Create a tensor with random values between -1 and 1
  #[wasm_bindgen]
  pub fn random(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::random(rows, cols),
    }
  }

  /// Get the shape of the tensor as [rows, cols]
  #[wasm_bindgen]
  pub fn shape(&self) -> js_sys::Array {
    let (rows, cols) = self.inner.shape();
    let array = js_sys::Array::new();
    array.push(&JsValue::from_f64(rows as f64));
    array.push(&JsValue::from_f64(cols as f64));
    array
  }

  /// Get the tensor data as a flattened Float64Array
  #[wasm_bindgen]
  pub fn data(&self) -> Float64Array {
    let (rows, cols) = self.inner.shape();
    let mut flat_data = Vec::with_capacity(rows * cols);

    for i in 0..rows {
      for j in 0..cols {
        flat_data.push(self.inner.data[[i, j]]);
      }
    }

    Float64Array::from(&flat_data[..])
  }

  /// Set whether this tensor requires gradients
  #[wasm_bindgen]
  pub fn set_requires_grad(&mut self, requires_grad: bool) {
    self.inner.set_requires_grad(requires_grad);
  }

  /// Check if this tensor requires gradients
  #[wasm_bindgen]
  pub fn requires_grad(&self) -> bool {
    self.inner.requires_grad()
  }

  /// Get the gradient as a JsTensor (if available)
  #[wasm_bindgen]
  pub fn gradient(&self) -> Option<JsTensor> {
    self.inner.gradient().map(|grad| JsTensor { inner: grad })
  }

  /// Zero out the gradients
  #[wasm_bindgen]
  pub fn zero_grad(&mut self) {
    self.inner.zero_grad();
  }

  /// Perform matrix multiplication
  #[wasm_bindgen]
  pub fn matmul(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .matmul(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Matrix multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Add two tensors
  #[wasm_bindgen]
  pub fn add(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .add(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Addition failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Subtract two tensors
  #[wasm_bindgen]
  pub fn sub(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .sub(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Subtraction failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Element-wise multiplication
  #[wasm_bindgen]
  pub fn mul(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mul(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Scalar multiplication
  #[wasm_bindgen]
  pub fn mul_scalar(&self, scalar: f64) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mul_scalar(scalar)
      .map_err(|e| JsValue::from_str(&format!("Scalar multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply sigmoid activation
  #[wasm_bindgen]
  pub fn sigmoid(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .sigmoid()
      .map_err(|e| JsValue::from_str(&format!("Sigmoid failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply ReLU activation
  #[wasm_bindgen]
  pub fn relu(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .relu()
      .map_err(|e| JsValue::from_str(&format!("ReLU failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply tanh activation
  #[wasm_bindgen]
  pub fn tanh(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .tanh()
      .map_err(|e| JsValue::from_str(&format!("Tanh failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply softmax activation
  #[wasm_bindgen]
  pub fn softmax(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .softmax()
      .map_err(|e| JsValue::from_str(&format!("Softmax failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Compute mean of all elements
  #[wasm_bindgen]
  pub fn mean(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mean()
      .map_err(|e| JsValue::from_str(&format!("Mean failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Perform backward pass (compute gradients)
  #[wasm_bindgen]
  pub fn backward(&mut self) -> JsResult<()> {
    self
      .inner
      .backward()
      .map_err(|e| JsValue::from_str(&format!("Backward pass failed: {}", e)))?;
    Ok(())
  }

  /// Clone the tensor
  #[wasm_bindgen]
  pub fn clone(&self) -> JsTensor {
    JsTensor {
      inner: self.inner.clone(),
    }
  }

  /// Get a string representation of the tensor
  #[wasm_bindgen]
  pub fn to_string(&self) -> String {
    format!("{:?}", self.inner)
  }

  /// Log the tensor to browser console (for debugging)
  #[wasm_bindgen]
  pub fn log(&self) {
    console::log_1(&format!("Tensor: {:?}", self.inner).into());
  }
}

/// JavaScript-compatible wrapper for Neural Network Model
#[wasm_bindgen]
pub struct JsModel {
  inner: Sequential,
  graph: Rc<RefCell<ComputationGraph>>,
}

#[wasm_bindgen]
impl JsModel {
  /// Create a new empty model
  #[wasm_bindgen(constructor)]
  pub fn new() -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    JsModel {
      inner: Sequential::new().with_graph(graph.clone()),
      graph,
    }
  }

  /// Add a dense layer with ReLU activation
  #[wasm_bindgen]
  pub fn add_dense_relu(&mut self, input_size: usize, output_size: usize) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .relu_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Add a dense layer with sigmoid activation
  #[wasm_bindgen]
  pub fn add_dense_sigmoid(&mut self, input_size: usize, output_size: usize) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .sigmoid_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Add a dense layer with softmax activation (typically for output layer)
  #[wasm_bindgen]
  pub fn add_dense_softmax(&mut self, input_size: usize, output_size: usize) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .softmax_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Forward pass through the model
  #[wasm_bindgen]
  pub fn forward(&mut self, input: &JsTensor) -> JsResult<JsTensor> {
    // Ensure input tensor is tracked in our graph
    let tracked_input = input.inner.clone().with_graph(self.graph.clone());

    let output = self
      .inner
      .forward(tracked_input)
      .map_err(|e| JsValue::from_str(&format!("Forward pass failed: {}", e)))?;

    Ok(JsTensor { inner: output })
  }

  /// Get model summary as a string
  #[wasm_bindgen]
  pub fn summary(&self) -> String {
    format!("{:#?}", self.inner.summary())
  }

  /// Get total number of parameters
  #[wasm_bindgen]
  pub fn param_count(&self) -> usize {
    self.inner.summary().total_params
  }
}

fn activation_from_name(name: &str) -> JsResult<Activation> {
  match name {
    "relu" => Ok(Activation::ReLU),
    "sigmoid" => Ok(Activation::Sigmoid),
    "tanh" => Ok(Activation::Tanh),
    "linear" | "none" => Ok(Activation::None),
    "softmax" => Ok(Activation::Softmax),
    other => Err(JsValue::from_str(&format!(
      "Unsupported activation function: {}",
      other
    ))),
  }
}

fn output_activation_for_task(task: JsTaskType) -> Activation {
  match task {
    JsTaskType::BinaryClassification => Activation::Sigmoid,
    JsTaskType::MultiClassification => Activation::Softmax,
    JsTaskType::Regression => Activation::None,
  }
}

fn build_sequential_model(
  config: &ModelConfigData,
  graph: Rc<RefCell<ComputationGraph>>,
) -> JsResult<Sequential> {
  let hidden_activation = activation_from_name(&config.activation_fn)?;
  let output_activation = output_activation_for_task(config.task_type);

  let mut model = Sequential::new();

  for (idx, window) in config.layers.windows(2).enumerate() {
    let input_size = window[0];
    let output_size = window[1];
    let activation = if idx == config.layers.len() - 2 {
      output_activation
    } else {
      hidden_activation
    };
    model = model.dense(
      input_size,
      output_size,
      activation,
      WeightInit::XavierUniform,
    );
  }

  Ok(model.with_graph(graph))
}

fn create_loss_for_task(task: JsTaskType) -> Box<dyn Loss> {
  match task {
    JsTaskType::BinaryClassification => Box::new(BinaryCrossEntropy::new()),
    JsTaskType::MultiClassification => Box::new(CrossEntropy::new()),
    JsTaskType::Regression => Box::new(MeanSquaredError::new()),
  }
}

fn create_metric_sets_for_task(task: JsTaskType) -> (Vec<Box<dyn Metric>>, Vec<Box<dyn Metric>>) {
  match task {
    JsTaskType::BinaryClassification => (
      vec![
        Box::new(Accuracy::default()) as Box<dyn Metric>,
        Box::new(Precision::default()),
        Box::new(Recall::default()),
        Box::new(F1Score::default()),
      ],
      vec![
        Box::new(Accuracy::default()) as Box<dyn Metric>,
        Box::new(Precision::default()),
        Box::new(Recall::default()),
        Box::new(F1Score::default()),
      ],
    ),
    JsTaskType::MultiClassification => (
      vec![Box::new(CategoricalAccuracy) as Box<dyn Metric>],
      vec![Box::new(CategoricalAccuracy) as Box<dyn Metric>],
    ),
    JsTaskType::Regression => (
      vec![Box::new(MeanSquaredErrorMetric) as Box<dyn Metric>],
      vec![Box::new(MeanSquaredErrorMetric) as Box<dyn Metric>],
    ),
  }
}

fn metric_lookup(metrics: &HashMap<String, f64>, key: &str) -> Option<f64> {
  metrics.get(key).copied()
}

#[wasm_bindgen]
pub struct JsTrainer {
  model_config: ModelConfigData,
  training_config: TrainingConfigData,
  task_type: JsTaskType,
  graph: Rc<RefCell<ComputationGraph>>,
  model: RefCell<Sequential>,
}

#[wasm_bindgen]
impl JsTrainer {
  #[wasm_bindgen(constructor)]
  pub fn new(
    model_config: &JsModelConfig,
    training_config: &JsTrainingConfig,
  ) -> JsResult<JsTrainer> {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let model_config_inner = model_config.to_inner();
    let training_config_inner = training_config.to_inner();
    let model = build_sequential_model(&model_config_inner, graph.clone())?;

    Ok(Self {
      model_config: model_config_inner.clone(),
      training_config: training_config_inner.clone(),
      task_type: model_config_inner.task_type,
      graph,
      model: RefCell::new(model),
    })
  }

  pub async fn train(&mut self, dataset: &JsDataset) -> JsResult<JsTrainingResult> {
    if dataset.task_type() != self.task_type {
      return Err(JsValue::from_str(
        "Dataset task type does not match trainer configuration",
      ));
    }

    if dataset.len() == 0 {
      return Err(JsValue::from_str("Dataset is empty"));
    }

    let output_size = *self
      .model_config
      .layers
      .last()
      .ok_or_else(|| JsValue::from_str("Model configuration has no layers"))?;

    if dataset.label_dim() != output_size {
      return Err(JsValue::from_str(&format!(
        "Dataset label dimension ({}) does not match model output size ({})",
        dataset.label_dim(),
        output_size
      )));
    }

    let total_samples = dataset.len();
    let validation_ratio = self.training_config.validation_split;
    let mut val_count = if validation_ratio > 0.0 {
      ((total_samples as f64) * validation_ratio).round() as usize
    } else {
      0
    };

    if val_count >= total_samples {
      val_count = total_samples.saturating_sub(1);
    }

    let train_count = total_samples - val_count;
    let (train_features, val_features) = dataset.split_features(train_count);
    let (train_labels, val_labels) = dataset.split_labels(train_count);

    let mut train_x = Tensor::new(train_features.clone()).map_err(|e| {
      JsValue::from_str(&format!("Failed to create training feature tensor: {}", e))
    })?;
    train_x = train_x.with_graph(self.graph.clone());
    train_x.set_requires_grad(true);

    let train_y = Tensor::new(train_labels.clone())
      .map_err(|e| JsValue::from_str(&format!("Failed to create training label tensor: {}", e)))?;

    let (val_x_tensor, val_y_tensor) = if val_count > 0 {
      let val_x = Tensor::new(val_features.clone()).map_err(|e| {
        JsValue::from_str(&format!(
          "Failed to create validation feature tensor: {}",
          e
        ))
      })?;
      let val_y = Tensor::new(val_labels.clone()).map_err(|e| {
        JsValue::from_str(&format!("Failed to create validation label tensor: {}", e))
      })?;
      (Some(val_x), Some(val_y))
    } else {
      (None, None)
    };

    let mut optimizer = self
      .training_config
      .optimizer_config
      .inner()
      .build_optimizer();

    let domain_config = TrainingConfig {
      epochs: self.training_config.epochs,
      batch_size: self.training_config.batch_size,
      shuffle: true,
      validation_frequency: if val_count > 0 { 1 } else { 0 },
      verbose: true, // Enable verbose output for WASM
      early_stopping_patience: self.training_config.early_stopping_patience,
      early_stopping_min_delta: self.training_config.early_stopping_min_delta,
      enable_early_stopping: self.training_config.enable_early_stopping,
      learning_rate: self.training_config.optimizer_config.inner().learning_rate,
      regularization: self
        .training_config
        .regularization_config
        .as_ref()
        .and_then(|cfg| cfg.to_domain()),
      #[cfg(not(target_arch = "wasm32"))]
      show_gui_plots: false,
    };

    let mut model_ref = self.model.get_mut();

    // Create trainer with specific optimizer type
    let mut trainer = match self.training_config.optimizer_config.inner().optimizer_type {
      JsOptimizerType::GD => {
        let opt = GradientDescent::new(self.training_config.optimizer_config.inner().learning_rate);
        match self.task_type {
          JsTaskType::BinaryClassification => {
            Trainer::new(&mut model_ref, BinaryCrossEntropy::new(), opt)
          }
          JsTaskType::MultiClassification => Trainer::new(&mut model_ref, CrossEntropy::new(), opt),
          JsTaskType::Regression => Trainer::new(&mut model_ref, MeanSquaredError::new(), opt),
        }
      }
      JsOptimizerType::SGD => {
        let opt = SGD::new(self.training_config.optimizer_config.inner().learning_rate);
        match self.task_type {
          JsTaskType::BinaryClassification => {
            Trainer::new(&mut model_ref, BinaryCrossEntropy::new(), opt)
          }
          JsTaskType::MultiClassification => Trainer::new(&mut model_ref, CrossEntropy::new(), opt),
          JsTaskType::Regression => Trainer::new(&mut model_ref, MeanSquaredError::new(), opt),
        }
      }
      JsOptimizerType::SGDMomentum => {
        let opt = SGDMomentum::new(
          self.training_config.optimizer_config.inner().learning_rate,
          0.9,
        );
        match self.task_type {
          JsTaskType::BinaryClassification => {
            Trainer::new(&mut model_ref, BinaryCrossEntropy::new(), opt)
          }
          JsTaskType::MultiClassification => Trainer::new(&mut model_ref, CrossEntropy::new(), opt),
          JsTaskType::Regression => Trainer::new(&mut model_ref, MeanSquaredError::new(), opt),
        }
      }
      JsOptimizerType::RMSProp => {
        let opt = RMSProp::new(self.training_config.optimizer_config.inner().learning_rate);
        match self.task_type {
          JsTaskType::BinaryClassification => {
            Trainer::new(&mut model_ref, BinaryCrossEntropy::new(), opt)
          }
          JsTaskType::MultiClassification => Trainer::new(&mut model_ref, CrossEntropy::new(), opt),
          JsTaskType::Regression => Trainer::new(&mut model_ref, MeanSquaredError::new(), opt),
        }
      }
      JsOptimizerType::Adam => {
        let opt = Adam::new(self.training_config.optimizer_config.inner().learning_rate);
        match self.task_type {
          JsTaskType::BinaryClassification => {
            Trainer::new(&mut model_ref, BinaryCrossEntropy::new(), opt)
          }
          JsTaskType::MultiClassification => Trainer::new(&mut model_ref, CrossEntropy::new(), opt),
          JsTaskType::Regression => Trainer::new(&mut model_ref, MeanSquaredError::new(), opt),
        }
      }
    }
    .with_config(domain_config);

    let (train_metrics, val_metrics) = create_metric_sets_for_task(self.task_type);
    for metric in train_metrics.into_iter() {
      trainer = trainer.with_train_metric_box(metric);
    }
    for metric in val_metrics.into_iter() {
      trainer = trainer.with_val_metric_box(metric);
    }

    let history = trainer
      .fit(
        &train_x,
        &train_y,
        val_x_tensor.as_ref(),
        val_y_tensor.as_ref(),
      )
      .map_err(|e| JsValue::from_str(&format!("Training failed: {}", e)))?
      .clone();

    drop(trainer);

    Ok(JsTrainingResult::from_history(history, self.task_type))
  }

  pub fn predict(&self, input: &JsTensor) -> JsResult<JsTensor> {
    let mut model_ref = self.model.borrow_mut();
    let mut tensor = input.inner.clone().with_graph(self.graph.clone());
    tensor.set_requires_grad(false);
    let output = model_ref
      .forward(tensor)
      .map_err(|e| JsValue::from_str(&format!("Prediction failed: {}", e)))?;
    Ok(JsTensor { inner: output })
  }

  pub fn weight_matrices(&self) -> js_sys::Array {
    let model_ref = self.model.borrow();
    let matrices = model_ref.weight_matrices();
    let outer = js_sys::Array::new();

    for layer_matrix in matrices {
      let js_layer = js_sys::Array::new();
      for row in layer_matrix {
        let js_row = js_sys::Array::new();
        for value in row {
          js_row.push(&JsValue::from_f64(value));
        }
        js_layer.push(&js_row);
      }
      outer.push(&js_layer);
    }

    outer
  }

  pub fn bias_vectors(&self) -> js_sys::Array {
    let model_ref = self.model.borrow();
    let vectors = model_ref.bias_vectors();
    let outer = js_sys::Array::new();

    for bias in vectors {
      let js_bias = js_sys::Array::new();
      for value in bias {
        js_bias.push(&JsValue::from_f64(value));
      }
      outer.push(&js_bias);
    }

    outer
  }
}

#[wasm_bindgen]
pub struct JsTrainingResult {
  loss_history: Vec<f64>,
  accuracy_history: Vec<f64>,
  val_loss_history: Vec<f64>,
  val_accuracy_history: Vec<f64>,
  metrics: JsMetrics,
}

impl JsTrainingResult {
  fn from_history(history: TrainingHistory, task: JsTaskType) -> JsTrainingResult {
    let loss_history = history.train_losses();
    let val_loss_history = history
      .epochs
      .iter()
      .map(|epoch| epoch.val_loss.unwrap_or(f64::NAN))
      .collect();

    let accuracy_key = match task {
      JsTaskType::BinaryClassification => "accuracy",
      JsTaskType::MultiClassification => "categorical_accuracy",
      JsTaskType::Regression => "mse",
    };

    let accuracy_history = history
      .epochs
      .iter()
      .map(|epoch| metric_lookup(&epoch.train_metrics, accuracy_key).unwrap_or(f64::NAN))
      .collect();

    let val_accuracy_history = history
      .epochs
      .iter()
      .map(|epoch| metric_lookup(&epoch.val_metrics, accuracy_key).unwrap_or(f64::NAN))
      .collect();

    let metrics = JsMetrics::from_epoch(history.epochs.last(), task);

    JsTrainingResult {
      loss_history,
      accuracy_history,
      val_loss_history,
      val_accuracy_history,
      metrics,
    }
  }
}

#[wasm_bindgen]
impl JsTrainingResult {
  #[wasm_bindgen(getter)]
  pub fn loss_history(&self) -> js_sys::Array {
    self
      .loss_history
      .iter()
      .map(|v| JsValue::from_f64(*v))
      .collect()
  }

  #[wasm_bindgen(getter)]
  pub fn accuracy_history(&self) -> js_sys::Array {
    self
      .accuracy_history
      .iter()
      .map(|v| JsValue::from_f64(*v))
      .collect()
  }

  #[wasm_bindgen(getter)]
  pub fn validation_loss_history(&self) -> js_sys::Array {
    self
      .val_loss_history
      .iter()
      .map(|v| JsValue::from_f64(*v))
      .collect()
  }

  #[wasm_bindgen(getter)]
  pub fn validation_accuracy_history(&self) -> js_sys::Array {
    self
      .val_accuracy_history
      .iter()
      .map(|v| JsValue::from_f64(*v))
      .collect()
  }

  #[wasm_bindgen(getter)]
  pub fn final_metrics(&self) -> JsMetrics {
    self.metrics.clone()
  }
}

#[derive(Debug, Clone)]
struct MetricsData {
  loss: f64,
  accuracy: Option<f64>,
  precision: Option<f64>,
  recall: Option<f64>,
  f1_score: Option<f64>,
  mse: Option<f64>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct JsMetrics {
  inner: MetricsData,
}

impl JsMetrics {
  fn from_epoch(epoch: Option<&EpochHistory>, task: JsTaskType) -> JsMetrics {
    if let Some(epoch) = epoch {
      let loss = epoch.val_loss.unwrap_or(epoch.train_loss);
      let accuracy_key = match task {
        JsTaskType::BinaryClassification => "accuracy",
        JsTaskType::MultiClassification => "categorical_accuracy",
        JsTaskType::Regression => "mse",
      };

      let accuracy = metric_lookup(&epoch.val_metrics, accuracy_key)
        .or_else(|| metric_lookup(&epoch.train_metrics, accuracy_key));

      JsMetrics {
        inner: MetricsData {
          loss,
          accuracy,
          precision: metric_lookup(&epoch.train_metrics, "precision"),
          recall: metric_lookup(&epoch.train_metrics, "recall"),
          f1_score: metric_lookup(&epoch.train_metrics, "f1_score"),
          mse: metric_lookup(&epoch.train_metrics, "mse"),
        },
      }
    } else {
      JsMetrics {
        inner: MetricsData {
          loss: 0.0,
          accuracy: None,
          precision: None,
          recall: None,
          f1_score: None,
          mse: None,
        },
      }
    }
  }
}

#[wasm_bindgen]
impl JsMetrics {
  #[wasm_bindgen(getter)]
  pub fn accuracy(&self) -> f64 {
    self.inner.accuracy.unwrap_or(0.0)
  }

  #[wasm_bindgen(getter)]
  pub fn loss(&self) -> f64 {
    self.inner.loss
  }

  #[wasm_bindgen(getter)]
  pub fn precision(&self) -> Option<f64> {
    self.inner.precision
  }

  #[wasm_bindgen(getter)]
  pub fn recall(&self) -> Option<f64> {
    self.inner.recall
  }

  #[wasm_bindgen(getter)]
  pub fn f1_score(&self) -> Option<f64> {
    self.inner.f1_score
  }

  #[wasm_bindgen(getter)]
  pub fn mse(&self) -> Option<f64> {
    self.inner.mse
  }
}

#[wasm_bindgen]
pub struct JsDataPoint {
  x: f64,
  y: f64,
  label: f64,
}

#[wasm_bindgen]
impl JsDataPoint {
  #[wasm_bindgen(constructor)]
  pub fn new(x: f64, y: f64, label: f64) -> JsDataPoint {
    JsDataPoint { x, y, label }
  }

  #[wasm_bindgen(getter)]
  pub fn x(&self) -> f64 {
    self.x
  }

  #[wasm_bindgen(getter)]
  pub fn y(&self) -> f64 {
    self.y
  }

  #[wasm_bindgen(getter)]
  pub fn label(&self) -> f64 {
    self.label
  }
}

/// Dataset wrapper for JavaScript
#[wasm_bindgen]
pub struct JsDataset {
  features: Vec<Vec<f64>>,
  labels: Vec<Vec<f64>>,
  task_type: JsTaskType,
}

#[wasm_bindgen]
impl JsDataset {
  /// Create a new dataset from features and labels (defaulting to binary classification)
  #[wasm_bindgen(constructor)]
  pub fn new(features: js_sys::Array, labels: Float64Array) -> JsResult<JsDataset> {
    let mut feature_vec = Vec::new();

    for i in 0..features.length() {
      let row = features.get(i);
      if let Ok(row_array) = row.dyn_into::<Float64Array>() {
        feature_vec.push(row_array.to_vec());
      } else {
        return Err(JsValue::from_str(
          "Features must be an array of Float64Array",
        ));
      }
    }

    let mut label_matrix = Vec::new();
    let label_vec = labels.to_vec();
    for value in label_vec {
      label_matrix.push(vec![value]);
    }

    JsDataset::from_data(feature_vec, label_matrix, JsTaskType::BinaryClassification)
  }

  #[wasm_bindgen(getter)]
  pub fn task_type(&self) -> JsTaskType {
    self.task_type
  }

  /// Get the number of samples
  #[wasm_bindgen]
  pub fn len(&self) -> usize {
    self.features.len()
  }

  /// Get the number of features per sample
  #[wasm_bindgen]
  pub fn feature_count(&self) -> usize {
    self.features.first().map(|row| row.len()).unwrap_or(0)
  }

  /// Get features as a tensor
  #[wasm_bindgen]
  pub fn features_tensor(&self) -> JsResult<JsTensor> {
    if self.features.is_empty() {
      return Err(JsValue::from_str("Dataset is empty"));
    }

    match Tensor::new(self.features.clone()) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create features tensor: {}",
        e
      ))),
    }
  }

  /// Get labels as a tensor
  #[wasm_bindgen]
  pub fn labels_tensor(&self) -> JsResult<JsTensor> {
    if self.labels.is_empty() {
      return Err(JsValue::from_str("Dataset is empty"));
    }

    match Tensor::new(self.labels.clone()) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create labels tensor: {}",
        e
      ))),
    }
  }
}

impl JsDataset {
  pub(crate) fn from_data(
    features: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
    task_type: JsTaskType,
  ) -> JsResult<JsDataset> {
    if features.len() != labels.len() {
      return Err(JsValue::from_str(
        "Features and labels must contain the same number of samples",
      ));
    }

    if let Some(expected) = features.first().map(|row| row.len()) {
      if !features.iter().all(|row| row.len() == expected) {
        return Err(JsValue::from_str("Inconsistent feature dimensions"));
      }
    }

    if let Some(expected) = labels.first().map(|row| row.len()) {
      if !labels.iter().all(|row| row.len() == expected) {
        return Err(JsValue::from_str("Inconsistent label dimensions"));
      }
    }

    Ok(JsDataset {
      features,
      labels,
      task_type,
    })
  }

  pub(crate) fn label_dim(&self) -> usize {
    self.labels.first().map(|row| row.len()).unwrap_or(0)
  }

  pub(crate) fn split_features(&self, train_count: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let total = self.features.len();
    let split = train_count.min(total);
    let train = self.features[..split].to_vec();
    let val = if split < total {
      self.features[split..].to_vec()
    } else {
      Vec::new()
    };
    (train, val)
  }

  pub(crate) fn split_labels(&self, train_count: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let total = self.labels.len();
    let split = train_count.min(total);
    let train = self.labels[..split].to_vec();
    let val = if split < total {
      self.labels[split..].to_vec()
    } else {
      Vec::new()
    };
    (train, val)
  }
}

#[wasm_bindgen]
pub fn generate_dataset_from_points(points: js_sys::Array) -> JsResult<JsDataset> {
  if points.length() == 0 {
    return Err(JsValue::from_str("Point collection must not be empty"));
  }

  let mut features = Vec::with_capacity(points.length() as usize);
  let mut labels = Vec::with_capacity(points.length() as usize);

  for value in points.iter() {
    // Extract data point properties directly from JS object
    let x = js_sys::Reflect::get(&value, &JsValue::from_str("x"))
      .map_err(|_| JsValue::from_str("Data point missing 'x' property"))?
      .as_f64()
      .ok_or_else(|| JsValue::from_str("Data point 'x' must be numeric"))?;
    let y = js_sys::Reflect::get(&value, &JsValue::from_str("y"))
      .map_err(|_| JsValue::from_str("Data point missing 'y' property"))?
      .as_f64()
      .ok_or_else(|| JsValue::from_str("Data point 'y' must be numeric"))?;
    let label = js_sys::Reflect::get(&value, &JsValue::from_str("label"))
      .map_err(|_| JsValue::from_str("Data point missing 'label' property"))?
      .as_f64()
      .ok_or_else(|| JsValue::from_str("Data point 'label' must be numeric"))?;

    features.push(vec![x, y]);
    labels.push(vec![label]);
  }

  JsDataset::from_data(features, labels, JsTaskType::BinaryClassification)
}

/// Data conversion utilities between JavaScript and Rust types
#[wasm_bindgen]
pub struct DataConverter;

#[wasm_bindgen]
impl DataConverter {
  /// Convert JavaScript 2D array to JsTensor
  #[wasm_bindgen]
  pub fn array_to_tensor(array: js_sys::Array) -> JsResult<JsTensor> {
    let mut data = Vec::new();

    for i in 0..array.length() {
      let row_val = array.get(i);

      // Handle both Array and Float64Array
      let row_data = if let Ok(float_array) = row_val.clone().dyn_into::<Float64Array>() {
        float_array.to_vec()
      } else if let Ok(js_array) = row_val.dyn_into::<js_sys::Array>() {
        let mut row = Vec::new();
        for j in 0..js_array.length() {
          let val = js_array.get(j);
          if let Some(num) = val.as_f64() {
            row.push(num);
          } else {
            return Err(JsValue::from_str("Array elements must be numbers"));
          }
        }
        row
      } else {
        return Err(JsValue::from_str(
          "Array rows must be arrays or Float64Array",
        ));
      };

      data.push(row_data);
    }

    if data.is_empty() {
      return Err(JsValue::from_str("Array cannot be empty"));
    }

    match Tensor::new(data) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create tensor: {}",
        e
      ))),
    }
  }

  /// Convert JsTensor to JavaScript 2D array
  #[wasm_bindgen]
  pub fn tensor_to_array(tensor: &JsTensor) -> js_sys::Array {
    let (rows, cols) = tensor.inner.shape();
    let result = js_sys::Array::new();

    for i in 0..rows {
      let row = js_sys::Array::new();
      for j in 0..cols {
        row.push(&JsValue::from_f64(tensor.inner.data[[i, j]]));
      }
      result.push(&row);
    }

    result
  }

  /// Convert flat JavaScript array to JsTensor with specified shape
  #[wasm_bindgen]
  pub fn flat_array_to_tensor(
    flat_array: Float64Array,
    rows: usize,
    cols: usize,
  ) -> JsResult<JsTensor> {
    JsTensor::new(flat_array, rows, cols)
  }

  /// Convert JsTensor to flat JavaScript array
  #[wasm_bindgen]
  pub fn tensor_to_flat_array(tensor: &JsTensor) -> Float64Array {
    tensor.data()
  }

  /// Create tensor from CSV-like string data
  #[wasm_bindgen]
  pub fn csv_to_tensor(csv_string: &str, has_header: bool, delimiter: &str) -> JsResult<JsTensor> {
    let lines: Vec<&str> = csv_string.lines().collect();

    if lines.is_empty() {
      return Err(JsValue::from_str("CSV string is empty"));
    }

    let start_row = if has_header { 1 } else { 0 };
    let mut data = Vec::new();

    for line in lines.iter().skip(start_row) {
      let values: std::result::Result<Vec<f64>, std::num::ParseFloatError> = line
        .split(delimiter)
        .map(|s| s.trim().parse::<f64>())
        .collect();

      match values {
        Ok(row) => data.push(row),
        Err(_) => return Err(JsValue::from_str("Failed to parse numeric values from CSV")),
      }
    }

    if data.is_empty() {
      return Err(JsValue::from_str("No data rows found in CSV"));
    }

    match Tensor::new(data) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create tensor from CSV: {}",
        e
      ))),
    }
  }

  /// Convert tensor to CSV-like string
  #[wasm_bindgen]
  pub fn tensor_to_csv(tensor: &JsTensor, delimiter: &str) -> String {
    let (rows, cols) = tensor.inner.shape();
    let mut lines = Vec::new();

    for i in 0..rows {
      let mut row_values = Vec::new();
      for j in 0..cols {
        row_values.push(tensor.inner.data[[i, j]].to_string());
      }
      lines.push(row_values.join(delimiter));
    }

    lines.join("\n")
  }

  /// Normalize tensor values to [0, 1] range
  #[wasm_bindgen]
  pub fn normalize_min_max(tensor: &JsTensor) -> JsResult<JsTensor> {
    let (rows, cols) = tensor.inner.shape();
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    // Find min and max values
    for i in 0..rows {
      for j in 0..cols {
        let val = tensor.inner.data[[i, j]];
        min_val = min_val.min(val);
        max_val = max_val.max(val);
      }
    }

    let range = max_val - min_val;
    if range == 0.0 {
      return Ok(tensor.clone());
    }

    // Normalize
    let mut normalized_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        let normalized = (tensor.inner.data[[i, j]] - min_val) / range;
        row.push(normalized);
      }
      normalized_data.push(row);
    }

    match Tensor::new(normalized_data) {
      Ok(normalized_tensor) => Ok(JsTensor {
        inner: normalized_tensor,
      }),
      Err(e) => Err(JsValue::from_str(&format!("Normalization failed: {}", e))),
    }
  }

  /// Standardize tensor values (z-score normalization)
  #[wasm_bindgen]
  pub fn standardize(tensor: &JsTensor) -> JsResult<JsTensor> {
    let (rows, cols) = tensor.inner.shape();
    let total_elements = (rows * cols) as f64;

    // Calculate mean
    let mut sum = 0.0;
    for i in 0..rows {
      for j in 0..cols {
        sum += tensor.inner.data[[i, j]];
      }
    }
    let mean = sum / total_elements;

    // Calculate standard deviation
    let mut sum_sq_diff = 0.0;
    for i in 0..rows {
      for j in 0..cols {
        let diff = tensor.inner.data[[i, j]] - mean;
        sum_sq_diff += diff * diff;
      }
    }
    let std_dev = (sum_sq_diff / total_elements).sqrt();

    if std_dev == 0.0 {
      return Ok(tensor.clone());
    }

    // Standardize
    let mut standardized_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        let standardized = (tensor.inner.data[[i, j]] - mean) / std_dev;
        row.push(standardized);
      }
      standardized_data.push(row);
    }

    match Tensor::new(standardized_data) {
      Ok(standardized_tensor) => Ok(JsTensor {
        inner: standardized_tensor,
      }),
      Err(e) => Err(JsValue::from_str(&format!("Standardization failed: {}", e))),
    }
  }
}

/// Utility functions
#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
  /// Create a simple 2-layer neural network for binary classification
  #[wasm_bindgen]
  pub fn create_binary_classifier(input_size: usize, hidden_size: usize) -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let model = Sequential::new()
      .relu_layer(input_size, hidden_size)
      .sigmoid_layer(hidden_size, 1)
      .with_graph(graph.clone());

    JsModel {
      inner: model,
      graph,
    }
  }

  /// Create a multi-class classifier with softmax output
  #[wasm_bindgen]
  pub fn create_multiclass_classifier(
    input_size: usize,
    hidden_size: usize,
    num_classes: usize,
  ) -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let model = Sequential::new()
      .relu_layer(input_size, hidden_size)
      .relu_layer(hidden_size, hidden_size / 2)
      .softmax_layer(hidden_size / 2, num_classes)
      .with_graph(graph.clone());

    JsModel {
      inner: model,
      graph,
    }
  }

  /// Log a message to the browser console
  #[wasm_bindgen]
  pub fn log(message: &str) {
    console::log_1(&message.into());
  }

  /// Calculate binary cross-entropy loss
  #[wasm_bindgen]
  pub fn binary_cross_entropy(predictions: &JsTensor, targets: &JsTensor) -> JsResult<f64> {
    let loss_fn = BinaryCrossEntropy::new();
    match loss_fn.forward(&predictions.inner, &targets.inner) {
      Ok(loss) => Ok(loss.data[[0, 0]]),
      Err(e) => Err(JsValue::from_str(&format!(
        "Loss computation failed: {}",
        e
      ))),
    }
  }

  /// Calculate accuracy for binary classification
  #[wasm_bindgen]
  pub fn binary_accuracy(predictions: &JsTensor, targets: &JsTensor) -> JsResult<f64> {
    let accuracy_metric = Accuracy::new(0.5); // Default threshold of 0.5
    match accuracy_metric.compute(&predictions.inner, &targets.inner) {
      Ok(acc) => Ok(acc),
      Err(e) => Err(JsValue::from_str(&format!(
        "Accuracy computation failed: {}",
        e
      ))),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use wasm_bindgen_test::*;

  #[wasm_bindgen_test]
  fn test_js_tensor_creation() {
    let data = Float64Array::from(&[1.0, 2.0, 3.0, 4.0][..]);
    let tensor = JsTensor::new(data, 2, 2).unwrap();

    let shape = tensor.shape();
    assert_eq!(shape.get(0), JsValue::from_f64(2.0));
    assert_eq!(shape.get(1), JsValue::from_f64(2.0));
  }

  #[wasm_bindgen_test]
  fn test_js_tensor_operations() {
    let a = JsTensor::ones(2, 2);
    let b = JsTensor::ones(2, 2);
    let c = a.add(&b).unwrap();

    let data = c.data();
    // Each element should be 2.0 (1.0 + 1.0)
    for i in 0..4 {
      assert_eq!(data.get_index(i), 2.0);
    }
  }

  #[wasm_bindgen_test]
  fn test_js_model_creation() {
    let mut model = JsModel::new();
    model.add_dense_relu(4, 8).unwrap();
    model.add_dense_sigmoid(8, 1).unwrap();

    assert!(model.param_count() > 0);
  }
}
