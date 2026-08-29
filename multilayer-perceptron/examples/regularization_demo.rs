//! Minimal Regularization Demo
//!
//! Shows how adding regularization only requires changing 3 lines of code.

use multilayer_perceptron::domain::services::loss::RegularizationConfig;
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
  println!("🎯 Minimal Regularization Demo");
  println!("==============================");

  let (train_x, train_y, val_x, val_y) = load_dataset()?;
  println!(
    "📊 Boston Housing: {} train, {} val samples\n",
    train_x.shape().0,
    val_x.shape().0
  );

  // Without regularization
  let rmse1 = train_without_regularization(&train_x, &train_y, &val_x, &val_y)?;
  println!("   ✓ No regularization: RMSE {:.4}", rmse1);

  // L2 regularization
  let l2_config = RegularizationConfig::l2_only(0.001);
  let rmse2 = train_with_regularization(&train_x, &train_y, &val_x, &val_y, l2_config)?;
  println!("   ✓ L2 regularization (λ=0.001): RMSE {:.4}", rmse2);

  // L1 regularization
  let l1_config = RegularizationConfig::l1_only(0.001);
  let rmse3 = train_with_regularization(&train_x, &train_y, &val_x, &val_y, l1_config)?;
  println!("   ✓ L1 regularization (λ=0.001): RMSE {:.4}", rmse3);

  // Elastic Net regularization
  let elastic_config = RegularizationConfig::elastic_net(0.0001, 0.0001);
  let rmse4 = train_with_regularization(&train_x, &train_y, &val_x, &val_y, elastic_config)?;
  println!("   ✓ Elastic Net (λ1=0.0001, λ2=0.0001): RMSE {:.4}", rmse4);

  Ok(())
}

fn train_without_regularization(
  train_x_raw: &Tensor,
  train_y_raw: &Tensor,
  val_x_raw: &Tensor,
  val_y_raw: &Tensor,
) -> Result<f64> {
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
  let train_x = train_x_raw.clone().with_graph(graph.clone());
  let train_y = Tensor::from_array2(train_y_scaled)?.with_graph(graph.clone());
  let val_x = val_x_raw.clone().with_graph(graph.clone());
  let val_y = Tensor::from_array2(val_y_scaled)?.with_graph(graph.clone());

  // Build neural network model for regression
  let mut model = Sequential::new()
    .tanh_layer(13, 32) // Input layer with smooth activation
    .tanh_layer(32, 16) // Hidden layer
    .linear_layer(16, 1) // Output layer: linear activation for regression
    .with_graph(graph);

  // Setup training components for regression
  let mut config = TrainingConfig {
    epochs: 400,
    batch_size: 32,
    shuffle: true,
    validation_frequency: 10,
    verbose: false,
    early_stopping_patience: 20,
    early_stopping_min_delta: 0.05,
    enable_early_stopping: true,
    learning_rate: 0.0005,
    regularization: None,
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: false,
  };
  #[cfg(not(target_arch = "wasm32"))]
  {
    config.show_gui_plots = gui_plots_enabled();
  }

  let loss_fn = MeanSquaredError::new(); // MSE for regression
  let optimizer = SGD::new(config.learning_rate);

  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer).with_config(config);

  let _history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  // Final evaluation
  model.eval();
  let final_predictions_scaled = model.forward(val_x.clone())?;
  let mut final_predictions = final_predictions_scaled.clone();
  final_predictions.data = final_predictions
    .data
    .mapv(|v| v * target_std + target_mean);

  let final_mse = MeanSquaredError::new().forward(&final_predictions, &val_y_original)?;
  let final_rmse = final_mse.data[[0, 0]].sqrt();

  Ok(final_rmse)
}

fn train_with_regularization(
  train_x_raw: &Tensor,
  train_y_raw: &Tensor,
  val_x_raw: &Tensor,
  val_y_raw: &Tensor,
  reg_config: RegularizationConfig,
) -> Result<f64> {
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

  let graph = Rc::new(RefCell::new(ComputationGraph::new()));
  let train_x = train_x_raw.clone().with_graph(graph.clone());
  let train_y = Tensor::from_array2(train_y_scaled)?.with_graph(graph.clone());
  let val_x = val_x_raw.clone().with_graph(graph.clone());
  let val_y = Tensor::from_array2(val_y_scaled)?.with_graph(graph.clone());

  let mut model = Sequential::new()
    .tanh_layer(13, 32)
    .tanh_layer(32, 16)
    .linear_layer(16, 1)
    .with_graph(graph);

  let config = TrainingConfig {
    epochs: 400,
    batch_size: 32,
    shuffle: true,
    validation_frequency: 10,
    verbose: false,
    early_stopping_patience: 20,
    early_stopping_min_delta: 0.05,
    enable_early_stopping: true,
    learning_rate: 0.0005,
    regularization: Some(reg_config),
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: gui_plots_enabled(),
  };

  let loss_fn = MeanSquaredError::new();
  let optimizer = SGD::new(config.learning_rate);
  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer).with_config(config);
  let _history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  model.eval();
  let final_predictions_scaled = model.forward(val_x.clone())?;
  let mut final_predictions = final_predictions_scaled.clone();
  final_predictions.data = final_predictions
    .data
    .mapv(|v| v * target_std + target_mean);

  let final_mse = MeanSquaredError::new().forward(&final_predictions, &val_y_original)?;
  let final_rmse = final_mse.data[[0, 0]].sqrt();

  Ok(final_rmse)
}

fn load_dataset() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
  let config = PreprocessConfig {
    standardize: true,
    normalize: false,
    random_seed: Some(42),
  };
  let repo = BostonHousingRepository::new("data/boston_housing_formatted.csv");
  let dataset = repo.load_dataset(&config)?;
  let (mut train_dataset, mut val_dataset) = dataset.train_test_split(0.2)?;

  let mut pipeline = build_pipeline(&config);
  if !pipeline.is_empty() {
    pipeline.fit(&train_dataset)?;
    pipeline.apply(&mut train_dataset)?;
    pipeline.apply(&mut val_dataset)?;
  }

  train_dataset.to_tensors().and_then(|(train_x, train_y)| {
    val_dataset
      .to_tensors()
      .map(|(val_x, val_y)| (train_x, train_y, val_x, val_y))
  })
}
