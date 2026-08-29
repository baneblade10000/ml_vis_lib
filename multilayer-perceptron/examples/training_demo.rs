//! Training Demo
//!
//! This example demonstrates how to train a neural network using the multilayer perceptron library.
//! It creates synthetic data for binary classification and trains a simple neural network.

use multilayer_perceptron::prelude::*;
use multilayer_perceptron::usecase::preprocess::build_pipeline;
use std::cell::RefCell;
use std::rc::Rc;

fn gui_plots_enabled() -> bool {
  #[cfg(not(target_arch = "wasm32"))]
  {
    matches!(std::env::var("SHOW_GUI_PLOTS"), Ok(value) if value != "0")
  }
  #[cfg(target_arch = "wasm32")]
  {
    false
  }
}

fn main() -> Result<()> {
  println!("🚀 Multilayer Perceptron Training Demo");
  println!("======================================");

  if gui_plots_enabled() {
    println!("🖥️  Native learning curve visualizer enabled (SHOW_GUI_PLOTS ≠ 0)");
  }

  println!("\n=== Binary Classification Demo ===");
  demo_binary_classification()?;

  println!("\n=== Regression Demo ===");
  demo_regression()?;

  println!("\n=== Multi-Class Classification Demo ===");
  demo_multiclass()?;

  Ok(())
}

/// Demonstrate binary classification using the breast cancer dataset
fn demo_binary_classification() -> Result<()> {
  // Load breast cancer dataset
  println!("\n📊 Loading breast cancer dataset...");
  let (train_x, train_y, val_x, val_y) = load_breast_cancer_dataset()?;

  println!("Training set shape: {:?}", train_x.shape());
  println!("Validation set shape: {:?}", val_x.shape());

  // Create computation graph
  let graph = Rc::new(RefCell::new(ComputationGraph::new()));

  // Connect training data to computation graph
  let train_x = train_x.with_graph(graph.clone());
  let train_y = train_y.with_graph(graph.clone());
  let val_x = val_x.with_graph(graph.clone());
  let val_y = val_y.with_graph(graph.clone());

  // Build neural network model
  println!("\n🧠 Building neural network model...");
  let mut model = Sequential::new()
    .relu_layer(30, 24) // Input layer: 30 features -> 24 hidden units
    .relu_layer(24, 16) // Hidden layer: 24 -> 16 units
    .relu_layer(16, 8) // Hidden layer: 16 -> 8 units
    .sigmoid_layer(8, 1) // Output layer: 8 -> 1 unit (binary classification)
    .with_graph(graph);

  // Print model summary
  let summary = model.summary_with_input_shape(train_x.shape());
  println!("Model Architecture:");
  println!("{}", summary);

  // Setup training components
  println!("\n⚙️  Setting up training components...");
  let full_batch_size = train_x.shape().0;
  let mut config = TrainingConfig {
    epochs: 500,
    batch_size: full_batch_size,
    shuffle: false,
    validation_frequency: 5,
    verbose: true,
    early_stopping_patience: 10,
    early_stopping_min_delta: 0.001,
    enable_early_stopping: true,
    learning_rate: 0.01,
    regularization: None,
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: false,
  };
  #[cfg(not(target_arch = "wasm32"))]
  {
    let gui_enabled = gui_plots_enabled();
    config.show_gui_plots = gui_enabled;
  }
  let loss_fn = BinaryCrossEntropy::new();
  let optimizer = SGD::new(config.learning_rate);
  println!(
    "Using optimizer: {} (learning rate = {:.4})",
    optimizer.name(),
    config.learning_rate
  );

  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer)
    .with_config(config)
    .with_train_metric(Accuracy::default())
    .with_train_metric(Precision::default())
    .with_train_metric(Recall::default())
    .with_val_metric(Accuracy::default())
    .with_val_metric(F1Score::default());

  // Start training
  println!("\n🎯 Starting training...");
  println!("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1");
  println!("------|------------|----------|-----------|---------|-------");

  let history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  // Training summary
  println!("\n📈 Training Summary");
  println!("==================");
  println!("Total training time: {:.2?}", history.total_time);
  println!("Total epochs: {}", history.epochs.len());

  if let Some(best_epoch) = history.best_epoch {
    let best = &history.epochs[best_epoch];
    println!("Best epoch: {} (0-indexed)", best_epoch);
    println!(
      "Best validation loss: {:.4}",
      best.val_loss.unwrap_or(best.train_loss)
    );
  }

  if history.stopped_early {
    println!("Training stopped early due to no improvement.");
  }

  // Final evaluation
  println!("\n🔍 Final Evaluation");
  println!("===================");
  model.eval();
  let final_predictions = model.forward(val_x.clone())?;

  let final_loss = BinaryCrossEntropy::new().forward(&final_predictions, &val_y)?;
  let accuracy_metric = Accuracy::default();
  let final_accuracy = accuracy_metric.compute(&final_predictions, &val_y)?;

  println!("Final validation loss: {:.4}", final_loss.data[[0, 0]]);
  println!("Final validation accuracy: {:.2}%", final_accuracy * 100.0);

  // Show some example predictions
  println!("\n🔮 Sample Predictions");
  println!("====================");
  for i in 0..5.min(val_x.shape().0) {
    let prediction = final_predictions.data[[i, 0]];
    let true_label = val_y.data[[i, 0]];
    let predicted_class = if prediction >= 0.5 { 1 } else { 0 };
    let true_class = if true_label >= 0.5 {
      "Malignant"
    } else {
      "Benign"
    };
    let predicted_diagnosis = if predicted_class == 1 {
      "Malignant"
    } else {
      "Benign"
    };

    println!(
      "Sample {}: Prediction: {:.3} ({}) | True: {} | Correct: {}",
      i + 1,
      prediction,
      predicted_diagnosis,
      true_class,
      predicted_class == (true_label as i32)
    );
  }

  println!("\n✅ Binary classification demo completed successfully!");

  Ok(())
}

/// Demonstrate regression using the Boston housing dataset
fn demo_regression() -> Result<()> {
  // Load Boston housing dataset
  println!("\n📊 Loading Boston housing dataset...");
  let (train_x_raw, train_y_raw, val_x_raw, val_y_raw) = load_boston_housing_dataset()?;

  println!("Training set shape: {:?}", train_x_raw.shape());
  println!("Validation set shape: {:?}", val_x_raw.shape());

  // Compute target scaling (z-score) for stability
  let target_column = train_y_raw.data.column(0);
  let target_mean = target_column.mean().unwrap_or(0.0);
  let target_var = target_column
    .iter()
    .map(|v| (v - target_mean).powi(2))
    .sum::<f64>()
    / target_column.len().max(1) as f64;
  let target_std = target_var.sqrt().max(1e-8);

  let train_y_scaled = train_y_raw.data.mapv(|v| (v - target_mean) / target_std);
  let val_y_scaled = val_y_raw.data.mapv(|v| (v - target_mean) / target_std);

  let val_y_original = Tensor::from_array2(val_y_raw.data.clone())?;

  // Create computation graph
  let graph = Rc::new(RefCell::new(ComputationGraph::new()));

  // Connect training data to computation graph
  let train_x = train_x_raw.with_graph(graph.clone());
  let train_y = Tensor::from_array2(train_y_scaled)?.with_graph(graph.clone());
  let val_x = val_x_raw.with_graph(graph.clone());
  let val_y = Tensor::from_array2(val_y_scaled)?.with_graph(graph.clone());

  // Build neural network model for regression
  println!("\n🧠 Building regression neural network model...");
  let mut model = Sequential::new()
    .tanh_layer(13, 32) // Input layer with smooth activation
    .tanh_layer(32, 16) // Hidden layer
    .linear_layer(16, 1) // Output layer: linear activation for regression
    .with_graph(graph);

  // Print model summary
  let summary = model.summary_with_input_shape(train_x.shape());
  println!("Model Architecture:");
  println!("{}", summary);

  // Setup training components for regression
  println!("\n⚙️  Setting up regression training components...");
  let mut config = TrainingConfig {
    epochs: 400,
    batch_size: 32,
    shuffle: true,
    validation_frequency: 10,
    verbose: true,
    early_stopping_patience: 20,
    early_stopping_min_delta: 0.05,
    enable_early_stopping: true,
    learning_rate: 0.0005,
    regularization: Some(RegularizationConfig::l2_only(0.02)),
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: false,
  };
  #[cfg(not(target_arch = "wasm32"))]
  {
    let gui_enabled = gui_plots_enabled();
    config.show_gui_plots = gui_enabled;
  }

  let loss_fn = MeanSquaredError::new(); // MSE for regression
  let optimizer = SGD::new(config.learning_rate);
  println!(
    "Using optimizer: {} (learning rate = {:.4})",
    optimizer.name(),
    config.learning_rate
  );
  println!("Using MSE loss for regression task");

  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer).with_config(config);
  // Note: We could add regression-specific metrics here like MAE, R²

  // Start training
  println!("\n🎯 Starting regression training...");
  println!("Epoch | Train Loss | Val Loss");
  println!("------|------------|----------");

  let history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  // Training summary
  println!("\n📈 Regression Training Summary");
  println!("=============================");
  println!("Total training time: {:.2?}", history.total_time);
  println!("Total epochs: {}", history.epochs.len());

  if let Some(best_epoch) = history.best_epoch {
    let best = &history.epochs[best_epoch];
    println!("Best epoch: {} (0-indexed)", best_epoch);
    println!(
      "Best validation MSE: {:.4}",
      best.val_loss.unwrap_or(best.train_loss)
    );
  }

  if history.stopped_early {
    println!("Training stopped early due to no improvement.");
  }

  // Final evaluation
  println!("\n🔍 Final Regression Evaluation");
  println!("==============================");
  model.eval();
  let final_predictions_scaled = model.forward(val_x.clone())?;
  let mut final_predictions = final_predictions_scaled.clone();
  final_predictions.data = final_predictions
    .data
    .mapv(|v| v * target_std + target_mean);

  let final_mse = MeanSquaredError::new().forward(&final_predictions, &val_y_original)?;
  let final_rmse = final_mse.data[[0, 0]].sqrt();

  println!("Final validation MSE: {:.4}", final_mse.data[[0, 0]]);
  println!("Final validation RMSE: {:.4}", final_rmse);

  // Show some example predictions
  println!("\n🔮 Sample Regression Predictions");
  println!("===============================");
  for i in 0..5.min(val_x.shape().0) {
    let prediction = final_predictions.data[[i, 0]];
    let true_value = val_y_original.data[[i, 0]];
    let error = (prediction - true_value).abs();

    println!(
      "Sample {}: Predicted: ${:.1}k | True: ${:.1}k | Error: ${:.1}k",
      i + 1,
      prediction,
      true_value,
      error
    );
  }

  println!("\n✅ Regression demo completed successfully!");

  Ok(())
}

/// Demonstrate multi-class classification using the Iris dataset
fn demo_multiclass() -> Result<()> {
  // Load Iris dataset
  println!("\n📊 Loading Iris dataset...");
  let (train_x, train_y, val_x, val_y) = load_iris_dataset()?;

  println!("Training set shape: {:?}", train_x.shape());
  println!("Validation set shape: {:?}", val_x.shape());

  // Create computation graph
  let graph = Rc::new(RefCell::new(ComputationGraph::new()));

  // Connect training data to computation graph
  let train_x = train_x.with_graph(graph.clone());
  let train_y = train_y.with_graph(graph.clone());
  let val_x = val_x.with_graph(graph.clone());
  let val_y = val_y.with_graph(graph.clone());

  // Build neural network model for multi-class classification
  println!("\n🧠 Building multi-class classification neural network model...");
  let mut model = Sequential::new()
    .relu_layer(4, 16) // Input layer: 4 features -> 16 hidden units
    .relu_layer(16, 8) // Hidden layer: 16 -> 8 units
    .softmax_layer(8, 3) // Output layer: 8 -> 3 classes (softmax for multi-class)
    .with_graph(graph);

  // Print model summary
  let summary = model.summary_with_input_shape(train_x.shape());
  println!("Model Architecture:");
  println!("{}", summary);

  // Setup training components for multi-class classification
  println!("\n⚙️  Setting up multi-class training components...");
  let mut config = TrainingConfig {
    epochs: 200,
    batch_size: 16,
    shuffle: true,
    validation_frequency: 10,
    verbose: true,
    early_stopping_patience: 20,
    early_stopping_min_delta: 0.001,
    enable_early_stopping: true,
    learning_rate: 0.01,
    regularization: None,
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: false,
  };
  #[cfg(not(target_arch = "wasm32"))]
  {
    let gui_enabled = gui_plots_enabled();
    config.show_gui_plots = gui_enabled;
  }

  let loss_fn = CrossEntropy::new(); // Cross-entropy for multi-class
  let optimizer = SGD::new(config.learning_rate);
  println!(
    "Using optimizer: {} (learning rate = {:.4})",
    optimizer.name(),
    config.learning_rate
  );
  println!("Using Cross-entropy loss with softmax for multi-class classification");

  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer)
    .with_config(config)
    .with_val_metric(CategoricalAccuracy); // Categorical accuracy

  // Start training
  println!("\n🎯 Starting multi-class training...");
  println!("Epoch | Train Loss | Val Loss | Val Acc");
  println!("------|------------|----------|--------");

  let history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  // Training summary
  println!("\n📈 Multi-class Training Summary");
  println!("==============================");
  println!("Total training time: {:.2?}", history.total_time);
  println!("Total epochs: {}", history.epochs.len());

  if let Some(best_epoch) = history.best_epoch {
    let best = &history.epochs[best_epoch];
    println!("Best epoch: {} (0-indexed)", best_epoch);
    println!(
      "Best validation loss: {:.4}",
      best.val_loss.unwrap_or(best.train_loss)
    );
  }

  if history.stopped_early {
    println!("Training stopped early due to no improvement.");
  }

  // Final evaluation
  println!("\n🔍 Final Multi-class Evaluation");
  println!("===============================");
  model.eval();
  let final_predictions = model.forward(val_x.clone())?;

  let final_loss = CrossEntropy::new().forward(&final_predictions, &val_y)?;
  let accuracy_metric = CategoricalAccuracy;
  let final_accuracy = accuracy_metric.compute(&final_predictions, &val_y)?;

  println!("Final validation loss: {:.4}", final_loss.data[[0, 0]]);
  println!("Final validation accuracy: {:.2}%", final_accuracy * 100.0);

  // Show some example predictions
  println!("\n🔮 Sample Multi-class Predictions");
  println!("=================================");
  let class_names = ["Setosa", "Versicolor", "Virginica"];

  for i in 0..5.min(val_x.shape().0) {
    // Get predicted class (argmax of softmax output)
    let mut max_idx = 0;
    let mut max_val = final_predictions.data[[i, 0]];
    for j in 1..3 {
      if final_predictions.data[[i, j]] > max_val {
        max_val = final_predictions.data[[i, j]];
        max_idx = j;
      }
    }

    let true_class_idx = val_y
      .data
      .row(i)
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
      .map(|(idx, _)| idx)
      .unwrap_or(0);
    let predicted_class = class_names[max_idx];
    let true_class = class_names[true_class_idx];

    println!(
      "Sample {}: Predicted: {} ({:.3}) | True: {} | Correct: {}",
      i + 1,
      predicted_class,
      max_val,
      true_class,
      max_idx == true_class_idx
    );
  }

  println!("\n✅ Multi-class classification demo completed successfully!");

  Ok(())
}

/// Load breast cancer dataset from CSV file
///
/// This function loads the Wisconsin breast cancer dataset from the CSV file,
/// preprocesses it, and splits it into training and validation sets.
fn load_breast_cancer_dataset() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
  // Configure preprocessing
  let config = PreprocessConfig {
    standardize: true,     // Apply z-score normalization
    normalize: false,      // Don't apply min-max normalization (already standardizing)
    random_seed: Some(42), // For reproducible splits
  };

  // Load dataset from CSV via repository
  let repo = BreastCancerRepository::new("./data/data.csv");
  let dataset = repo.load_dataset(&config)?;

  println!(
    "Loaded {} samples with {} features",
    dataset.len(),
    dataset.n_features()
  );

  // Split into train and validation sets (80/20 split)
  let (mut train_dataset, mut val_dataset) = dataset.train_test_split(0.2)?;

  // Build preprocessing pipeline and apply it consistently
  let mut pipeline = build_pipeline(&config);
  if !pipeline.is_empty() {
    pipeline.fit(&train_dataset)?;
    pipeline.apply(&mut train_dataset)?;
    pipeline.apply(&mut val_dataset)?;
  }

  println!("Train set: {} samples", train_dataset.len());
  println!("Validation set: {} samples", val_dataset.len());

  // Convert to tensors
  let (train_x, train_y) = train_dataset.to_tensors()?;
  let (val_x, val_y) = val_dataset.to_tensors()?;

  Ok((train_x, train_y, val_x, val_y))
}

/// Load Boston housing dataset from CSV file
///
/// This function loads the Boston housing dataset from the CSV file,
/// preprocesses it, and splits it into training and validation sets.
fn load_boston_housing_dataset() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
  // Configure preprocessing for regression
  let config = PreprocessConfig {
    standardize: true, // Apply z-score normalization for regression
    normalize: false,
    random_seed: Some(42), // For reproducible splits
  };

  // Load dataset from CSV via Boston Housing specific repository
  let repo = BostonHousingRepository::new("data/boston_housing_formatted.csv");
  let dataset = repo.load_dataset(&config)?;

  println!(
    "Loaded {} samples with {} features",
    dataset.len(),
    dataset.n_features()
  );

  // Split into train and validation sets (80/20 split)
  let (mut train_dataset, mut val_dataset) = dataset.train_test_split(0.2)?;

  // Build preprocessing pipeline and apply it consistently
  let mut pipeline = build_pipeline(&config);
  if !pipeline.is_empty() {
    pipeline.fit(&train_dataset)?;
    pipeline.apply(&mut train_dataset)?;
    pipeline.apply(&mut val_dataset)?;
  }

  println!("Train set: {} samples", train_dataset.len());
  println!("Validation set: {} samples", val_dataset.len());

  // Convert to tensors
  let (train_x, train_y) = train_dataset.to_tensors()?;
  let (val_x, val_y) = val_dataset.to_tensors()?;

  Ok((train_x, train_y, val_x, val_y))
}

/// Load Iris dataset from CSV file
///
/// This function loads the Iris dataset from the CSV file,
/// preprocesses it, and splits it into training and validation sets.
fn load_iris_dataset() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
  // Configure preprocessing for multi-class classification
  let config = PreprocessConfig {
    standardize: true, // Apply z-score normalization
    normalize: false,
    random_seed: Some(42), // For reproducible splits
  };

  // Load dataset from CSV via repository
  let repo = IrisRepository::new("./data/iris.csv");
  let dataset = repo.load_dataset(&config)?;

  println!(
    "Loaded {} samples with {} features",
    dataset.len(),
    dataset.n_features()
  );

  // Split into train and validation sets (70/30 split, smaller dataset)
  let (mut train_dataset, mut val_dataset) = dataset.train_test_split(0.3)?;

  // Build preprocessing pipeline and apply it consistently
  let mut pipeline = build_pipeline(&config);
  if !pipeline.is_empty() {
    pipeline.fit(&train_dataset)?;
    pipeline.apply(&mut train_dataset)?;
    pipeline.apply(&mut val_dataset)?;
  }

  println!("Train set: {} samples", train_dataset.len());
  println!("Validation set: {} samples", val_dataset.len());

  // Convert to tensors
  let (train_x, train_y_raw) = train_dataset.to_tensors()?;
  let (val_x, val_y_raw) = val_dataset.to_tensors()?;

  let train_y = to_one_hot(&train_y_raw, 3)?;
  let val_y = to_one_hot(&val_y_raw, 3)?;

  Ok((train_x, train_y, val_x, val_y))
}

fn to_one_hot(labels: &Tensor, num_classes: usize) -> Result<Tensor> {
  let (rows, cols) = labels.shape();
  if cols != 1 {
    return Err(TensorError::InvalidValue(format!(
      "Expected labels with shape (n, 1) for one-hot encoding, got {:?}",
      labels.shape()
    )));
  }

  let mut encoded = Tensor::zeros(rows, num_classes);

  for i in 0..rows {
    let class_value = labels.data[[i, 0]];
    if !class_value.is_finite() {
      return Err(TensorError::InvalidValue(format!(
        "Non-finite class label at index {}: {}",
        i, class_value
      )));
    }

    let class_idx = class_value.round() as isize;
    if class_idx < 0 || class_idx >= num_classes as isize {
      return Err(TensorError::InvalidValue(format!(
        "Class index {} out of range for {} classes",
        class_idx, num_classes
      )));
    }

    encoded.data[[i, class_idx as usize]] = 1.0;
  }

  Ok(encoded)
}
