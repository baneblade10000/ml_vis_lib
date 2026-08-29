use crate::core::{Result, Tensor};
use crate::domain::services::metrics::Metric;

pub fn evaluate_model(metric: &dyn Metric, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
  metric.compute(predictions, targets)
}
