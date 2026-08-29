//! Optimizer Comparison Demo (Regression)
//!
//! Train the same regression model on the Boston Housing dataset using multiple
//! optimizers to compare their convergence behaviour and validation performance.

use multilayer_perceptron::prelude::*;
use multilayer_perceptron::usecase::preprocess::build_pipeline;
use std::cell::RefCell;
use std::rc::Rc;

type OptimizerFactory = Box<dyn Fn(f64) -> Box<dyn Optimizer>>;

struct OptimizerSetup {
  name: &'static str,
  learning_rate: f64,
  factory: OptimizerFactory,
}

struct OptimizerReport {
  name: String,
  best_epoch: usize,
  best_train_rmse: f64,
  best_val_rmse: f64,
  final_val_rmse: f64,
  train_time_secs: f64,
}

struct DatasetBundle {
  train_x: Tensor,
  train_y: Tensor,
  val_x: Tensor,
  val_y: Tensor,
  val_y_original: Tensor,
  target_mean: f64,
  target_std: f64,
}

fn main() -> Result<()> {
  println!("🚀 Optimizer Comparison Demo (Regression)");
  println!("========================================");

  let dataset = load_boston_housing_dataset()?;
  println!(
    "Dataset prepared: train {} samples, validation {} samples",
    dataset.train_x.shape().0,
    dataset.val_x.shape().0
  );
  println!(
    "Target (median home value) mean {:.3}, std {:.3}\n",
    dataset.target_mean, dataset.target_std
  );

  let optimizers = vec![
    OptimizerSetup {
      name: "SGD",
      learning_rate: 0.001,
      factory: Box::new(|lr| Box::new(SGD::new(lr))),
    },
    OptimizerSetup {
      name: "SGD + Momentum",
      learning_rate: 0.001,
      factory: Box::new(|lr| Box::new(SGDMomentum::new(lr, 0.9))),
    },
    OptimizerSetup {
      name: "Gradient Descent",
      learning_rate: 0.001,
      factory: Box::new(|lr| Box::new(GradientDescent::new(lr))),
    },
    OptimizerSetup {
      name: "RMSProp",
      learning_rate: 0.001,
      factory: Box::new(|lr| Box::new(RMSProp::new(lr))),
    },
    OptimizerSetup {
      name: "Adam",
      learning_rate: 0.001,
      factory: Box::new(|lr| Box::new(Adam::new(lr))),
    },
  ];

  println!("Running experiments...\n");
  let mut reports = Vec::new();

  for setup in optimizers {
    let report = run_experiment(&setup, &dataset)?;
    println!(
      "{:<16} -> Best epoch {:>3}, best val RMSE {:.4}, final val RMSE {:.4}, time {:.2}s",
      report.name,
      report.best_epoch,
      report.best_val_rmse,
      report.final_val_rmse,
      report.train_time_secs
    );
    reports.push(report);
  }

  println!("\nSummary:");
  println!("========");
  for report in reports {
    println!(
      "{:<16} | Best epoch {:>3} | Train RMSE {:.4} | Val RMSE {:.4} | Final RMSE {:.4} | Time {:.2}s",
      report.name,
      report.best_epoch,
      report.best_train_rmse,
      report.best_val_rmse,
      report.final_val_rmse,
      report.train_time_secs
    );
  }

  Ok(())
}

fn run_experiment(setup: &OptimizerSetup, data: &DatasetBundle) -> Result<OptimizerReport> {
  let graph = Rc::new(RefCell::new(ComputationGraph::new()));

  let train_x = data.train_x.clone().with_graph(graph.clone());
  let train_y = data.train_y.clone().with_graph(graph.clone());
  let val_x = data.val_x.clone().with_graph(graph.clone());
  let val_y = data.val_y.clone().with_graph(graph.clone());

  let mut model = Sequential::new()
    .tanh_layer(train_x.shape().1, 32)
    .tanh_layer(32, 16)
    .linear_layer(16, 1)
    .with_graph(graph);

  let config = TrainingConfig {
    epochs: 60,
    batch_size: 32,
    shuffle: true,
    validation_frequency: 1,
    verbose: true,
    early_stopping_patience: 30,
    early_stopping_min_delta: 0.0005,
    enable_early_stopping: true,
    learning_rate: setup.learning_rate,
    regularization: None,
    #[cfg(not(target_arch = "wasm32"))]
    show_gui_plots: false,
  };

  let loss_fn: Box<dyn Loss> = Box::new(MeanSquaredError::new());
  let optimizer = (setup.factory)(setup.learning_rate);

  let mut trainer = Trainer::new_boxed(&mut model, loss_fn, optimizer);
  trainer = trainer.with_config(config);
  trainer = trainer.with_train_metric(MeanSquaredErrorMetric);
  trainer = trainer.with_val_metric(MeanSquaredErrorMetric);

  let history_ref = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;
  let history = history_ref.clone();

  drop(trainer); // release mutable borrow on model

  model.eval();
  let predictions_scaled = model.forward(val_x.clone())?;
  let mut predictions = predictions_scaled.clone();
  predictions.data = predictions
    .data
    .mapv(|v| v * data.target_std + data.target_mean);

  let final_mse = MeanSquaredError::new().forward(&predictions, &data.val_y_original)?;
  let final_rmse = final_mse.data[[0, 0]].sqrt();

  let best_epoch = history
    .best_epoch
    .unwrap_or_else(|| history.epochs.len().saturating_sub(1));
  let best_epoch_entry = &history.epochs[best_epoch];

  let best_train_rmse = best_epoch_entry.train_loss.sqrt() * data.target_std;
  let best_val_mse_scaled = best_epoch_entry
    .val_loss
    .unwrap_or(best_epoch_entry.train_loss);
  let best_val_rmse = best_val_mse_scaled.sqrt() * data.target_std;
  let train_time_secs = history.total_time.as_secs_f64();

  Ok(OptimizerReport {
    name: setup.name.to_string(),
    best_epoch,
    best_train_rmse,
    best_val_rmse,
    final_val_rmse: final_rmse,
    train_time_secs,
  })
}

fn load_boston_housing_dataset() -> Result<DatasetBundle> {
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

  let (train_x_raw, train_y_raw) = train_dataset.to_tensors()?;
  let (val_x_raw, val_y_raw) = val_dataset.to_tensors()?;

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

  Ok(DatasetBundle {
    train_x: train_x_raw,
    train_y: Tensor::from_array2(train_y_scaled)?,
    val_x: val_x_raw,
    val_y: Tensor::from_array2(val_y_scaled)?,
    val_y_original: Tensor::from_array2(val_y_raw.data.clone())?,
    target_mean,
    target_std,
  })
}
