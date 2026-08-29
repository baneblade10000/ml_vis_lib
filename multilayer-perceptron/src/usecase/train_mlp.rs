//! Neural network training system
//!
//! This module provides the core training infrastructure for neural networks,
//! including the Trainer struct that orchestrates the training loop with
//! forward propagation, backward propagation, and parameter optimization.

use super::preprocess::build_pipeline;
#[cfg(not(target_arch = "wasm32"))]
use crate::adapters::presentation::NativeVisualizer;
use crate::core::{Result, Tensor, TensorError};
use crate::domain::models::{Activation, Sequential};
use crate::domain::ports::DataRepository;
use crate::domain::services::loss::{
  BinaryCrossEntropy, Loss, MeanSquaredError, RegularizationConfig,
};
use crate::domain::services::metrics::{
  Accuracy, CategoricalAccuracy, F1Score, MeanSquaredErrorMetric, Metric, Precision, Recall,
};
use crate::domain::services::optimizer::Optimizer;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! console_log {
  ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
  /// Number of training epochs
  pub epochs: usize,
  /// Batch size for training
  pub batch_size: usize,
  /// Whether to shuffle training data between epochs
  pub shuffle: bool,
  /// Validation frequency (validate every N epochs, 0 to disable)
  pub validation_frequency: usize,
  /// Whether to print training progress
  pub verbose: bool,
  /// Early stopping patience (stop if no improvement for N epochs, 0 to disable)
  pub early_stopping_patience: usize,
  /// Minimum improvement threshold for early stopping
  pub early_stopping_min_delta: f64,
  /// Whether to enable early stopping logic
  pub enable_early_stopping: bool,
  /// Base learning rate to apply to the optimizer
  pub learning_rate: f64,
  /// Optional regularization configuration applied during training
  pub regularization: Option<RegularizationConfig>,
  /// Whether to display native GUI plots (only available on non-wasm targets)
  #[cfg(not(target_arch = "wasm32"))]
  pub show_gui_plots: bool,
}

impl Default for TrainingConfig {
  fn default() -> Self {
    Self {
      epochs: 100,
      batch_size: 32,
      shuffle: true,
      validation_frequency: 1,
      verbose: true,
      early_stopping_patience: 0, // Disabled by default
      early_stopping_min_delta: 0.0001,
      enable_early_stopping: true,
      learning_rate: 0.01,
      regularization: None,
      #[cfg(not(target_arch = "wasm32"))]
      show_gui_plots: false,
    }
  }
}

/// Training history for a single epoch
#[derive(Debug, Clone)]
pub struct EpochHistory {
  /// Epoch number (0-indexed)
  pub epoch: usize,
  /// Training loss
  pub train_loss: f64,
  /// Validation loss (if validation data provided)
  pub val_loss: Option<f64>,
  /// Training metrics
  pub train_metrics: HashMap<String, f64>,
  /// Validation metrics (if validation data provided)
  pub val_metrics: HashMap<String, f64>,
  /// Training time for this epoch
  pub training_time: Duration,
}

/// Complete training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
  /// History for each epoch
  pub epochs: Vec<EpochHistory>,
  /// Total training time
  pub total_time: Duration,
  /// Whether training was stopped early
  pub stopped_early: bool,
  /// Best epoch (based on validation loss if available, otherwise training loss)
  pub best_epoch: Option<usize>,
}

impl TrainingHistory {
  /// Create new empty training history
  pub fn new() -> Self {
    Self {
      epochs: Vec::new(),
      total_time: Duration::new(0, 0),
      stopped_early: false,
      best_epoch: None,
    }
  }

  /// Add epoch history
  pub fn add_epoch(&mut self, epoch_history: EpochHistory) {
    self.epochs.push(epoch_history);
  }

  /// Get training losses as a vector
  pub fn train_losses(&self) -> Vec<f64> {
    self.epochs.iter().map(|e| e.train_loss).collect()
  }

  /// Get validation losses as a vector (skipping epochs without validation)
  pub fn val_losses(&self) -> Vec<f64> {
    self.epochs.iter().filter_map(|e| e.val_loss).collect()
  }

  /// Get training metric values for a specific metric
  pub fn train_metric_values(&self, metric_name: &str) -> Vec<f64> {
    self
      .epochs
      .iter()
      .filter_map(|e| e.train_metrics.get(metric_name))
      .copied()
      .collect()
  }

  /// Get validation metric values for a specific metric
  pub fn val_metric_values(&self, metric_name: &str) -> Vec<f64> {
    self
      .epochs
      .iter()
      .filter_map(|e| e.val_metrics.get(metric_name))
      .copied()
      .collect()
  }

  /// Find the best epoch based on validation loss (or training loss if no validation)
  pub fn find_best_epoch(&mut self) {
    if self.epochs.is_empty() {
      return;
    }

    // Use validation loss if available, otherwise training loss
    let (best_idx, _) = self
      .epochs
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| {
        let loss_a = a.val_loss.unwrap_or(a.train_loss);
        let loss_b = b.val_loss.unwrap_or(b.train_loss);
        loss_a
          .partial_cmp(&loss_b)
          .unwrap_or(std::cmp::Ordering::Equal)
      })
      .unwrap();

    self.best_epoch = Some(best_idx);
  }
}

impl Default for TrainingHistory {
  fn default() -> Self {
    Self::new()
  }
}

pub struct TrainRequest {
  pub task: TaskKind,
  pub data_config: DataConfig,
  pub training_config: TrainingConfig,
  pub validation_split: Option<f64>,
  pub model: Sequential,
  pub optimizer: Box<dyn Optimizer>,
  pub loss_fn: Option<Box<dyn Loss>>,
  pub train_metrics: Vec<Box<dyn Metric>>,
  pub val_metrics: Vec<Box<dyn Metric>>,
}

pub struct TrainResponse {
  pub task: TaskKind,
  pub model: Sequential,
  pub history: TrainingHistory,
  pub train_dataset: Dataset,
  pub validation_dataset: Option<Dataset>,
}

struct TaskStrategy;

impl TaskStrategy {
  fn output_activation(task: TaskKind) -> Activation {
    match task {
      TaskKind::BinaryClassification => Activation::Sigmoid,
      TaskKind::MultiClassification => Activation::Softmax,
      TaskKind::Regression => Activation::None,
    }
  }

  fn default_loss(task: TaskKind) -> Box<dyn Loss> {
    match task {
      TaskKind::BinaryClassification | TaskKind::MultiClassification => {
        Box::new(BinaryCrossEntropy::new())
      }
      TaskKind::Regression => Box::new(MeanSquaredError::new()),
    }
  }

  fn default_train_metrics(task: TaskKind) -> Vec<Box<dyn Metric>> {
    match task {
      TaskKind::BinaryClassification => vec![
        Box::new(Accuracy::default()),
        Box::new(Precision::default()),
        Box::new(Recall::default()),
        Box::new(F1Score::default()),
      ],
      TaskKind::MultiClassification => vec![Box::new(CategoricalAccuracy)],
      TaskKind::Regression => vec![Box::new(MeanSquaredErrorMetric)],
    }
  }

  fn default_val_metrics(task: TaskKind) -> Vec<Box<dyn Metric>> {
    Self::default_train_metrics(task)
  }

  fn configure_model(task: TaskKind, model: &mut Sequential) {
    model.set_output_activation(Self::output_activation(task));
  }
}

pub struct TrainMLPUsecase<R: DataRepository> {
  data_repo: Arc<R>,
}

impl<R: DataRepository> TrainMLPUsecase<R> {
  pub fn new(data_repo: Arc<R>) -> Self {
    Self { data_repo }
  }

  pub fn execute(&self, request: TrainRequest) -> Result<TrainResponse> {
    let TrainRequest {
      task,
      data_config,
      training_config,
      validation_split,
      mut model,
      optimizer,
      loss_fn,
      mut train_metrics,
      mut val_metrics,
    } = request;

    TaskStrategy::configure_model(task, &mut model);

    let dataset = self.data_repo.load_dataset(&data_config)?;

    let validation_split = validation_split.unwrap_or(0.0);
    let (mut train_dataset, mut val_dataset) = if validation_split > 0.0 {
      let (train, val) = dataset.train_test_split(validation_split)?;
      (train, Some(val))
    } else {
      (dataset, None)
    };

    let mut pipeline = build_pipeline(&data_config);
    if !pipeline.is_empty() {
      pipeline.fit(&train_dataset)?;
      pipeline.apply(&mut train_dataset)?;
      if let Some(val) = val_dataset.as_mut() {
        pipeline.apply(val)?;
      }
    }

    let (train_x, train_y) = train_dataset.to_tensors()?;
    let (val_x, val_y) = match val_dataset.as_ref() {
      Some(dataset) => {
        let (vx, vy) = dataset.to_tensors()?;
        (Some(vx), Some(vy))
      }
      None => (None, None),
    };

    let loss_fn = loss_fn.unwrap_or_else(|| TaskStrategy::default_loss(task));

    if train_metrics.is_empty() {
      train_metrics = TaskStrategy::default_train_metrics(task);
    }

    if val_metrics.is_empty() {
      val_metrics = TaskStrategy::default_val_metrics(task);
    }

    let mut trainer =
      Trainer::new_boxed(&mut model, loss_fn, optimizer).with_config(training_config);

    for metric in train_metrics.into_iter() {
      trainer = trainer.with_train_metric_box(metric);
    }

    for metric in val_metrics.into_iter() {
      trainer = trainer.with_val_metric_box(metric);
    }

    let history_ref = trainer.fit(&train_x, &train_y, val_x.as_ref(), val_y.as_ref())?;
    let history = history_ref.clone();

    drop(trainer);

    Ok(TrainResponse {
      task,
      model,
      history,
      train_dataset,
      validation_dataset: val_dataset,
    })
  }
}

/// Neural network trainer
///
/// The Trainer orchestrates the training process, handling:
/// - Forward and backward propagation
/// - Loss computation and optimization
/// - Metric evaluation
/// - Training history tracking
/// - Early stopping
pub struct Trainer<'a> {
  /// The model to train
  model: &'a mut Sequential,
  /// Loss function
  loss_fn: Box<dyn Loss>,
  /// Optimizer
  optimizer: Box<dyn Optimizer>,
  /// Training metrics to evaluate
  train_metrics: Vec<Box<dyn Metric>>,
  /// Validation metrics to evaluate
  val_metrics: Vec<Box<dyn Metric>>,
  /// Training configuration
  config: TrainingConfig,
  /// Training history
  history: TrainingHistory,
}

impl<'a> Trainer<'a> {
  /// Create a new trainer
  ///
  /// # Arguments
  /// * `model` - Mutable reference to the model to train
  /// * `loss_fn` - Loss function to use
  /// * `optimizer` - Optimizer for parameter updates
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  /// use multilayer_perceptron::prelude::*;
  /// use std::cell::RefCell;
  /// use std::rc::Rc;
  ///
  /// let graph = Rc::new(RefCell::new(ComputationGraph::new()));
  /// let mut model = Sequential::new()
  ///     .relu_layer(10, 5)
  ///     .sigmoid_layer(5, 1)
  ///     .with_graph(graph);
  ///
  /// let loss_fn = BinaryCrossEntropy::new();
  /// let optimizer = SGD::new(0.01);
  ///
  /// let trainer = Trainer::new(&mut model, loss_fn, optimizer);
  /// ```
  pub fn new<L, O>(model: &'a mut Sequential, loss_fn: L, optimizer: O) -> Self
  where
    L: Loss + 'static,
    O: Optimizer + 'static,
  {
    let optimizer: Box<dyn Optimizer> = Box::new(optimizer);
    let default_config = TrainingConfig {
      learning_rate: optimizer.learning_rate(),
      ..TrainingConfig::default()
    };

    Self {
      model,
      loss_fn: Box::new(loss_fn),
      optimizer,
      train_metrics: Vec::new(),
      val_metrics: Vec::new(),
      config: default_config,
      history: TrainingHistory::new(),
    }
  }

  pub fn new_boxed(
    model: &'a mut Sequential,
    loss_fn: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
  ) -> Self {
    let default_config = TrainingConfig {
      learning_rate: optimizer.learning_rate(),
      ..TrainingConfig::default()
    };

    Self {
      model,
      loss_fn,
      optimizer,
      train_metrics: Vec::new(),
      val_metrics: Vec::new(),
      config: default_config,
      history: TrainingHistory::new(),
    }
  }

  /// Set training configuration
  pub fn with_config(mut self, config: TrainingConfig) -> Self {
    self.optimizer.set_learning_rate(config.learning_rate);
    self.config = config;
    self
  }

  /// Add a training metric
  pub fn with_train_metric<M: Metric + 'static>(mut self, metric: M) -> Self {
    self.train_metrics.push(Box::new(metric));
    self
  }

  pub fn with_train_metric_box(mut self, metric: Box<dyn Metric>) -> Self {
    self.train_metrics.push(metric);
    self
  }

  /// Add a validation metric
  pub fn with_val_metric<M: Metric + 'static>(mut self, metric: M) -> Self {
    self.val_metrics.push(Box::new(metric));
    self
  }

  pub fn with_val_metric_box(mut self, metric: Box<dyn Metric>) -> Self {
    self.val_metrics.push(metric);
    self
  }

  /// Train the model
  ///
  /// # Arguments
  /// * `train_x` - Training input data
  /// * `train_y` - Training target data
  /// * `val_x` - Optional validation input data
  /// * `val_y` - Optional validation target data
  ///
  /// # Returns
  /// Training history
  pub fn fit(
    &mut self,
    train_x: &Tensor,
    train_y: &Tensor,
    val_x: Option<&Tensor>,
    val_y: Option<&Tensor>,
  ) -> Result<&TrainingHistory> {
    #[cfg(not(target_arch = "wasm32"))]
    let train_start = Instant::now();

    // Validate inputs
    if train_x.shape().0 != train_y.shape().0 {
      return Err(TensorError::ShapeMismatch {
        operation: "training data validation".to_string(),
        expected: (train_x.shape().0, train_y.shape().1),
        got: train_y.shape(),
      });
    }

    if let (Some(vx), Some(vy)) = (val_x, val_y) {
      if vx.shape().0 != vy.shape().0 {
        return Err(TensorError::ShapeMismatch {
          operation: "validation data validation".to_string(),
          expected: (vx.shape().0, vy.shape().1),
          got: vy.shape(),
        });
      }
    }

    let num_samples = train_x.shape().0;
    let mut indices: Vec<usize> = (0..num_samples).collect();

    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    // Training loop
    for epoch in 0..self.config.epochs {
      #[cfg(not(target_arch = "wasm32"))]
      let epoch_start = Instant::now();

      // Shuffle data if configured
      if self.config.shuffle {
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());
      }

      // Training phase
      self.model.train();
      let (train_loss, train_metrics) = self.train_epoch(train_x, train_y, &indices)?;

      // Validation phase (if validation data provided and frequency matches)
      let (val_loss, val_metrics) = if let (Some(vx), Some(vy)) = (val_x, val_y) {
        if self.config.validation_frequency > 0
          && (epoch + 1) % self.config.validation_frequency == 0
        {
          self.model.eval();
          let (loss, metrics) = self.validate(vx, vy)?;
          (Some(loss), metrics)
        } else {
          (None, HashMap::new())
        }
      } else {
        (None, HashMap::new())
      };

      #[cfg(not(target_arch = "wasm32"))]
      let epoch_time = epoch_start.elapsed();
      #[cfg(target_arch = "wasm32")]
      let epoch_time = std::time::Duration::from_secs(0);

      // Record epoch history
      let epoch_history = EpochHistory {
        epoch,
        train_loss,
        val_loss,
        train_metrics,
        val_metrics,
        training_time: epoch_time,
      };

      self.history.add_epoch(epoch_history);

      // Print progress
      if self.config.verbose {
        self.print_epoch_progress(
          epoch + 1,
          self.config.epochs,
          self.history.epochs.last().unwrap(),
        );
      }

      // Early stopping check
      if self.config.enable_early_stopping && self.config.early_stopping_patience > 0 {
        let current_val_loss = val_loss.unwrap_or(train_loss);

        if current_val_loss < best_val_loss - self.config.early_stopping_min_delta {
          best_val_loss = current_val_loss;
          epochs_without_improvement = 0;
        } else {
          epochs_without_improvement += 1;

          if epochs_without_improvement >= self.config.early_stopping_patience {
            if self.config.verbose {
              #[cfg(not(target_arch = "wasm32"))]
              println!(
                "Early stopping triggered after {} epochs without improvement",
                epochs_without_improvement
              );

              #[cfg(target_arch = "wasm32")]
              console_log!(
                "Early stopping triggered after {} epochs without improvement",
                epochs_without_improvement
              );
            }
            self.history.stopped_early = true;
            break;
          }
        }
      }
    }

    // Finalize training history
    #[cfg(not(target_arch = "wasm32"))]
    {
      self.history.total_time = train_start.elapsed();
    }
    #[cfg(target_arch = "wasm32")]
    {
      self.history.total_time = std::time::Duration::from_secs(0);
    }
    self.history.find_best_epoch();

    #[cfg(not(target_arch = "wasm32"))]
    {
      if self.config.show_gui_plots {
        if let Err(e) = NativeVisualizer::show_curves(&self.history) {
          if self.config.verbose {
            println!("Warning: Failed to show GUI plots: {}", e);
          }
        }
      }
    }

    if self.config.verbose {
      #[cfg(not(target_arch = "wasm32"))]
      {
        println!("\nTraining completed in {:.2?}", self.history.total_time);
        if let Some(best_epoch) = self.history.best_epoch {
          println!("Best epoch: {} (0-indexed)", best_epoch);
        }
      }

      #[cfg(target_arch = "wasm32")]
      {
        console_log!("Training completed in {:.2?}", self.history.total_time);
        if let Some(best_epoch) = self.history.best_epoch {
          console_log!("Best epoch: {} (0-indexed)", best_epoch);
        }
      }
    }

    Ok(&self.history)
  }

  /// Train for one epoch
  fn train_epoch(
    &mut self,
    train_x: &Tensor,
    train_y: &Tensor,
    indices: &[usize],
  ) -> Result<(f64, HashMap<String, f64>)> {
    if indices.is_empty() {
      // Return zero loss and empty metrics if no indices are provided
      return Ok((0.0, HashMap::new()));
    }
    let mut total_loss = 0.0;
    let mut total_predictions = Vec::with_capacity(indices.len());
    let mut total_targets = Vec::with_capacity(indices.len());
    let use_full_batch = self.optimizer.requires_full_batch();
    let effective_batch_size = if use_full_batch {
      indices.len()
    } else {
      std::cmp::max(self.config.batch_size, 1)
    };
    let num_batches = indices.len().div_ceil(effective_batch_size);

    for batch_idx in 0..num_batches {
      let start_idx = batch_idx * effective_batch_size;
      let end_idx = std::cmp::min(start_idx + effective_batch_size, indices.len());
      let batch_indices: &[usize] = if use_full_batch {
        indices
      } else {
        &indices[start_idx..end_idx]
      };

      // Create batch tensors
      let batch_x = self.extract_batch(train_x, batch_indices, true)?;
      let batch_y = self.extract_batch(train_y, batch_indices, false)?;

      // Forward pass
      let predictions = self.model.forward(batch_x)?;

      // Compute loss
      let loss = self.loss_fn.forward(&predictions, &batch_y)?;
      total_loss += loss.data[[0, 0]];

      // Compute gradient of loss with respect to predictions
      let loss_grad = self.loss_fn.backward(&predictions, &batch_y)?;

      // Backpropagate loss gradients through the computation graph
      if predictions.graph.is_some() && predictions.graph_id.is_some() {
        if let (Some(graph_weak), Some(node_id)) = (&predictions.graph, predictions.graph_id) {
          if let Some(graph) = graph_weak.upgrade() {
            graph
              .borrow_mut()
              .backward(node_id, Some(loss_grad.data.clone()))?;
          }
        }
      } else {
        return Err(TensorError::ComputationError {
          message: "Predictions not connected to computation graph".to_string(),
        });
      }

      // Update parameters
      self.update_parameters()?;

      // Collect predictions and targets for metric computation
      total_predictions.extend(self.tensor_to_vec(&predictions));
      total_targets.extend(self.tensor_to_vec(&batch_y));
    }

    // Compute average loss
    let avg_loss = total_loss / num_batches as f64;

    // Compute training metrics
    let pred_tensor = self.vec_to_tensor(total_predictions)?;
    let target_tensor = self.vec_to_tensor(total_targets)?;
    let train_metrics = self.compute_metrics(&self.train_metrics, &pred_tensor, &target_tensor)?;

    Ok((avg_loss, train_metrics))
  }

  /// Validate the model
  fn validate(&mut self, val_x: &Tensor, val_y: &Tensor) -> Result<(f64, HashMap<String, f64>)> {
    let mut total_loss = 0.0;
    let capacity = val_x.shape().0;
    let mut total_predictions = Vec::with_capacity(capacity);
    let mut total_targets = Vec::with_capacity(capacity);
    let num_samples = val_x.shape().0;
    let num_batches = num_samples.div_ceil(self.config.batch_size);

    for batch_idx in 0..num_batches {
      let start_idx = batch_idx * self.config.batch_size;
      let end_idx = std::cmp::min(start_idx + self.config.batch_size, num_samples);
      let indices: Vec<usize> = (start_idx..end_idx).collect();

      // Create batch tensors
      let batch_x = self.extract_batch(val_x, &indices, false)?;
      let batch_y = self.extract_batch(val_y, &indices, false)?;

      // Forward pass (no gradients needed for validation)
      let predictions = self.model.forward(batch_x)?;

      // Compute loss
      let loss = self.loss_fn.forward(&predictions, &batch_y)?;
      total_loss += loss.data[[0, 0]];

      // Collect predictions and targets for metric computation
      total_predictions.extend(self.tensor_to_vec(&predictions));
      total_targets.extend(self.tensor_to_vec(&batch_y));
    }

    // Compute average loss
    let avg_loss = total_loss / num_batches as f64;

    // Compute validation metrics
    let pred_tensor = self.vec_to_tensor(total_predictions)?;
    let target_tensor = self.vec_to_tensor(total_targets)?;
    let val_metrics = self.compute_metrics(&self.val_metrics, &pred_tensor, &target_tensor)?;

    Ok((avg_loss, val_metrics))
  }

  /// Extract a batch of samples from a tensor
  fn extract_batch(
    &self,
    tensor: &Tensor,
    indices: &[usize],
    requires_grad: bool,
  ) -> Result<Tensor> {
    let num_features = tensor.shape().1;
    let data = indices
      .iter()
      .map(|&sample_idx| {
        (0..num_features)
          .map(|feature_idx| tensor.data[[sample_idx, feature_idx]])
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    let mut batch_tensor = Tensor::new(data)?;

    if requires_grad {
      if let Some(weak_graph) = &tensor.graph {
        if let Some(graph) = weak_graph.upgrade() {
          batch_tensor = batch_tensor.with_graph(graph);
        }
      }
      batch_tensor.set_requires_grad(true);
    }

    Ok(batch_tensor)
  }

  /// Convert tensor to flat vector for metrics computation
  fn tensor_to_vec(&self, tensor: &Tensor) -> Vec<f64> {
    let mut vec = Vec::new();
    for i in 0..tensor.shape().0 {
      for j in 0..tensor.shape().1 {
        vec.push(tensor.data[[i, j]]);
      }
    }
    vec
  }

  /// Convert flat vector back to tensor
  fn vec_to_tensor(&self, vec: Vec<f64>) -> Result<Tensor> {
    if vec.is_empty() {
      return Err(TensorError::ComputationError {
        message: "Cannot convert empty vector to tensor".to_string(),
      });
    }

    // Reshape as column vector
    let data: Vec<Vec<f64>> = vec.into_iter().map(|x| vec![x]).collect();
    Tensor::new(data)
  }

  /// Compute metrics for given predictions and targets
  fn compute_metrics(
    &self,
    metrics: &[Box<dyn Metric>],
    predictions: &Tensor,
    targets: &Tensor,
  ) -> Result<HashMap<String, f64>> {
    let mut metric_values = HashMap::new();

    for metric in metrics {
      let value = metric.compute(predictions, targets)?;
      metric_values.insert(metric.name().to_string(), value);
    }

    Ok(metric_values)
  }

  /// Print epoch progress
  fn print_epoch_progress(&self, epoch: usize, total_epochs: usize, epoch_history: &EpochHistory) {
    #[cfg(not(target_arch = "wasm32"))]
    {
      print!(
        "Epoch {}/{} - loss: {:.4}",
        epoch, total_epochs, epoch_history.train_loss
      );

      // Print training metrics
      for (name, value) in &epoch_history.train_metrics {
        print!(" - {}: {:.4}", name, value);
      }

      // Print validation metrics if available
      if let Some(val_loss) = epoch_history.val_loss {
        print!(" - val_loss: {:.4}", val_loss);
        for (name, value) in &epoch_history.val_metrics {
          print!(" - val_{}: {:.4}", name, value);
        }
      }

      print!(" - time: {:.2?}", epoch_history.training_time);
      println!();
    }

    #[cfg(target_arch = "wasm32")]
    {
      let mut message = format!(
        "Epoch {}/{} - loss: {:.4}",
        epoch, total_epochs, epoch_history.train_loss
      );

      // Add training metrics
      for (name, value) in &epoch_history.train_metrics {
        message.push_str(&format!(" - {}: {:.4}", name, value));
      }

      // Add validation metrics if available
      if let Some(val_loss) = epoch_history.val_loss {
        message.push_str(&format!(" - val_loss: {:.4}", val_loss));
        for (name, value) in &epoch_history.val_metrics {
          message.push_str(&format!(" - val_{}: {:.4}", name, value));
        }
      }

      message.push_str(&format!(" - time: {:.2?}", epoch_history.training_time));
      console_log!("{}", message);
    }
  }

  /// Get the training history
  pub fn history(&self) -> &TrainingHistory {
    &self.history
  }

  /// Update model parameters using the optimizer
  fn update_parameters(&mut self) -> Result<()> {
    // Ensure gradients from the computation graph are available on the parameters
    self.model.sync_gradients()?;

    if let Some(reg_config) = &self.config.regularization {
      reg_config.add_regularization_gradients(self.model)?;
    }

    let mut params = self.model.parameters_mut();
    self.optimizer.step(params.as_mut_slice())?;
    self.optimizer.zero_grad(params.as_mut_slice());

    // Clear any remaining gradients in the model (e.g., intermediate tensors)
    self.model.zero_grad();

    Ok(())
  }
}
#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::ComputationGraph;
  use crate::domain::models::Sequential;
  use crate::domain::services::loss::BinaryCrossEntropy;
  use crate::domain::services::metrics::Accuracy;
  use crate::domain::services::optimizer::SGD;
  use std::cell::RefCell;
  use std::rc::Rc;

  fn create_test_model() -> Sequential {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    Sequential::new()
      .relu_layer(2, 4)
      .sigmoid_layer(4, 1)
      .with_graph(graph)
  }

  fn create_test_data() -> (Tensor, Tensor) {
    let x = Tensor::new(vec![
      vec![0.0, 0.0],
      vec![0.0, 1.0],
      vec![1.0, 0.0],
      vec![1.0, 1.0],
    ])
    .unwrap();

    let y = Tensor::new(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]).unwrap();

    (x, y)
  }

  #[test]
  fn test_training_config_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.epochs, 100);
    assert_eq!(config.batch_size, 32);
    assert!(config.shuffle);
    assert_eq!(config.validation_frequency, 1);
    assert!(config.verbose);
    assert_eq!(config.early_stopping_patience, 0);
    assert!((config.learning_rate - 0.01).abs() < f64::EPSILON);
  }

  #[test]
  fn test_training_history_creation() {
    let mut history = TrainingHistory::new();
    assert!(history.epochs.is_empty());
    assert_eq!(history.total_time, Duration::new(0, 0));
    assert!(!history.stopped_early);
    assert!(history.best_epoch.is_none());

    let epoch_history = EpochHistory {
      epoch: 0,
      train_loss: 0.5,
      val_loss: Some(0.6),
      train_metrics: HashMap::new(),
      val_metrics: HashMap::new(),
      training_time: Duration::from_millis(100),
    };

    history.add_epoch(epoch_history);
    assert_eq!(history.epochs.len(), 1);
    assert_eq!(history.train_losses(), vec![0.5]);
    assert_eq!(history.val_losses(), vec![0.6]);
  }

  #[test]
  fn test_trainer_creation() {
    let mut model = create_test_model();
    let loss_fn = BinaryCrossEntropy::new();
    let optimizer = SGD::new(0.01);

    let trainer = Trainer::new(&mut model, loss_fn, optimizer)
      .with_config(TrainingConfig {
        epochs: 5,
        batch_size: 2,
        verbose: false,
        ..Default::default()
      })
      .with_train_metric(Accuracy::default());

    assert_eq!(trainer.config.epochs, 5);
    assert_eq!(trainer.config.batch_size, 2);
    assert!(!trainer.config.verbose);
    assert_eq!(trainer.train_metrics.len(), 1);
    assert!((trainer.config.learning_rate - 0.01).abs() < f64::EPSILON);
  }

  #[test]
  fn test_extract_batch() {
    let mut model = create_test_model();
    let loss_fn = BinaryCrossEntropy::new();
    let optimizer = SGD::new(0.01);
    let trainer = Trainer::new(&mut model, loss_fn, optimizer);

    let (train_x, _) = create_test_data();
    let indices = vec![0, 2];

    let batch = trainer.extract_batch(&train_x, &indices, true).unwrap();
    assert_eq!(batch.shape(), (2, 2));

    // Should contain rows 0 and 2 from original tensor
    assert_eq!(batch.data[[0, 0]], 0.0); // Row 0
    assert_eq!(batch.data[[0, 1]], 0.0);
    assert_eq!(batch.data[[1, 0]], 1.0); // Row 2
    assert_eq!(batch.data[[1, 1]], 0.0);
  }

  #[test]
  fn test_tensor_vec_conversion() {
    let mut model = create_test_model();
    let loss_fn = BinaryCrossEntropy::new();
    let optimizer = SGD::new(0.01);
    let trainer = Trainer::new(&mut model, loss_fn, optimizer);

    let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let vec = trainer.tensor_to_vec(&tensor);
    assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);

    let reconstructed = trainer.vec_to_tensor(vec).unwrap();
    assert_eq!(reconstructed.shape(), (4, 1));
    assert_eq!(reconstructed.data[[0, 0]], 1.0);
    assert_eq!(reconstructed.data[[1, 0]], 2.0);
    assert_eq!(reconstructed.data[[2, 0]], 3.0);
    assert_eq!(reconstructed.data[[3, 0]], 4.0);
  }

  #[test]
  fn test_training_history_find_best_epoch() {
    let mut history = TrainingHistory::new();

    // Add epochs with decreasing then increasing loss
    let epoch1 = EpochHistory {
      epoch: 0,
      train_loss: 1.0,
      val_loss: Some(1.2),
      train_metrics: HashMap::new(),
      val_metrics: HashMap::new(),
      training_time: Duration::from_millis(100),
    };

    let epoch2 = EpochHistory {
      epoch: 1,
      train_loss: 0.8,
      val_loss: Some(0.9), // Best validation loss
      train_metrics: HashMap::new(),
      val_metrics: HashMap::new(),
      training_time: Duration::from_millis(100),
    };

    let epoch3 = EpochHistory {
      epoch: 2,
      train_loss: 0.6,
      val_loss: Some(1.1),
      train_metrics: HashMap::new(),
      val_metrics: HashMap::new(),
      training_time: Duration::from_millis(100),
    };

    history.add_epoch(epoch1);
    history.add_epoch(epoch2);
    history.add_epoch(epoch3);

    history.find_best_epoch();
    assert_eq!(history.best_epoch, Some(1)); // Epoch 1 has best validation loss
  }
}
