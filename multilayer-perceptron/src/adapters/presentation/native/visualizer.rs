//! Native learning curve visualizer.
//!
//! Provides an egui-based window for displaying training progress when running
//! on native targets. For WebAssembly builds we expose a stub implementation
//! that reports the feature as unavailable.

use crate::core::{Result, TensorError};
use crate::usecase::train_mlp::TrainingHistory;

/// Native learning curve visualizer using egui (stubbed out on wasm targets).
pub struct NativeVisualizer;

#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Line, Plot, PlotPoints};

#[cfg(not(target_arch = "wasm32"))]
impl NativeVisualizer {
  /// Create a new native visualizer.
  pub fn new() -> Self {
    Self
  }

  /// Display learning curves in a native GUI window.
  ///
  /// This call blocks until the user closes the window.
  pub fn show_curves(history: &TrainingHistory) -> Result<()> {
    let options = eframe::NativeOptions {
      viewport: egui::ViewportBuilder::default().with_inner_size([880.0, 660.0]),
      ..Default::default()
    };

    let app = LearningCurveApp::new(history);

    eframe::run_native(
      "Learning Curves - Multilayer Perceptron",
      options,
      Box::new(|_cc| Box::new(app)),
    )
    .map_err(|e| TensorError::ComputationError {
      message: format!("Failed to create GUI window: {}", e),
    })?;

    Ok(())
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for NativeVisualizer {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(target_arch = "wasm32")]
impl NativeVisualizer {
  /// Stubbed constructor for wasm targets.
  pub fn new() -> Self {
    Self
  }

  /// Stubbed visualization entry point that reports unavailability.
  pub fn show_curves(_history: &TrainingHistory) -> Result<()> {
    Err(TensorError::ComputationError {
      message: "Native visualization is not available on wasm targets".to_string(),
    })
  }
}

#[cfg(target_arch = "wasm32")]
impl Default for NativeVisualizer {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(not(target_arch = "wasm32"))]
struct LearningCurveApp {
  train_losses: Vec<[f64; 2]>,
  val_losses: Vec<[f64; 2]>,
  metric_series: Vec<MetricSeries>,
  show_loss: bool,
  show_metrics: bool,
}

#[cfg(not(target_arch = "wasm32"))]
struct MetricSeries {
  name: String,
  train: Vec<[f64; 2]>,
  val: Vec<[f64; 2]>,
}

#[cfg(not(target_arch = "wasm32"))]
impl LearningCurveApp {
  fn new(history: &TrainingHistory) -> Self {
    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();
    let mut metric_names = std::collections::BTreeSet::new();

    for epoch in &history.epochs {
      let epoch_idx = epoch.epoch as f64;
      train_losses.push([epoch_idx, epoch.train_loss]);
      if let Some(val_loss) = epoch.val_loss {
        val_losses.push([epoch_idx, val_loss]);
      }

      for metric_name in epoch.train_metrics.keys() {
        metric_names.insert(metric_name.to_string());
      }
      for metric_name in epoch.val_metrics.keys() {
        metric_names.insert(metric_name.to_string());
      }
    }

    let mut metric_series = Vec::new();
    for name in metric_names {
      let mut train_points = Vec::new();
      let mut val_points = Vec::new();

      for epoch in &history.epochs {
        let epoch_idx = epoch.epoch as f64;
        if let Some(value) = epoch.train_metrics.get(&name) {
          train_points.push([epoch_idx, *value]);
        }
        if let Some(value) = epoch.val_metrics.get(&name) {
          val_points.push([epoch_idx, *value]);
        }
      }

      if !train_points.is_empty() || !val_points.is_empty() {
        metric_series.push(MetricSeries {
          name,
          train: train_points,
          val: val_points,
        });
      }
    }

    let has_metrics = !metric_series.is_empty();

    Self {
      train_losses,
      val_losses,
      metric_series,
      show_loss: true,
      show_metrics: has_metrics,
    }
  }
}

#[cfg(not(target_arch = "wasm32"))]
impl eframe::App for LearningCurveApp {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.heading("Learning Curves");

      ui.horizontal(|ui| {
        ui.checkbox(&mut self.show_loss, "Show Loss");
        if !self.metric_series.is_empty() {
          ui.checkbox(&mut self.show_metrics, "Show Metrics");
        }
      });

      ui.separator();

      if self.show_loss && (!self.train_losses.is_empty() || !self.val_losses.is_empty()) {
        ui.heading("Loss");
        Plot::new("loss_plot")
          .legend(egui_plot::Legend::default())
          .allow_zoom(true)
          .allow_drag(true)
          .show(ui, |plot_ui| {
            if !self.train_losses.is_empty() {
              plot_ui.line(
                Line::new(PlotPoints::from(self.train_losses.clone()))
                  .color(egui::Color32::from_rgb(54, 125, 255))
                  .name("Training Loss")
                  .width(2.0),
              );
            }
            if !self.val_losses.is_empty() {
              plot_ui.line(
                Line::new(PlotPoints::from(self.val_losses.clone()))
                  .color(egui::Color32::from_rgb(231, 76, 60))
                  .name("Validation Loss")
                  .width(2.0),
              );
            }
          });
      }

      if self.show_metrics && !self.metric_series.is_empty() {
        ui.heading("Metrics");
        for series in &self.metric_series {
          Plot::new(format!("metric_plot_{}", series.name))
            .legend(egui_plot::Legend::default())
            .allow_zoom(true)
            .allow_drag(true)
            .show(ui, |plot_ui| {
              if !series.train.is_empty() {
                plot_ui.line(
                  Line::new(PlotPoints::from(series.train.clone()))
                    .color(egui::Color32::from_rgb(46, 204, 113))
                    .name(format!("Train {}", series.name))
                    .width(2.0),
                );
              }
              if !series.val.is_empty() {
                plot_ui.line(
                  Line::new(PlotPoints::from(series.val.clone()))
                    .color(egui::Color32::from_rgb(241, 196, 15))
                    .name(format!("Val {}", series.name))
                    .width(2.0),
                );
              }
            });
        }
      }

      ui.separator();

      if ui.button("Close Window").clicked() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
      }
    });
  }
}
