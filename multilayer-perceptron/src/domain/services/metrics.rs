//! Evaluation metrics for model performance assessment
//!
//! This module provides various metrics to evaluate the performance of
//! trained models, particularly for binary classification tasks.

use crate::core::{Result, Tensor};
use std::cmp::Ordering;

/// Binary classification metrics
///
/// Contains the four basic components of the confusion matrix:
/// - True Positives (TP): Correctly predicted positive cases
/// - True Negatives (TN): Correctly predicted negative cases  
/// - False Positives (FP): Incorrectly predicted positive cases (Type I error)
/// - False Negatives (FN): Incorrectly predicted negative cases (Type II error)
#[derive(Debug, Clone, Copy)]
pub struct BinaryClassificationMetrics {
  /// True Positives
  pub tp: usize,
  /// True Negatives
  pub tn: usize,
  /// False Positives
  pub fp: usize,
  /// False Negatives
  pub fn_: usize,
}

impl BinaryClassificationMetrics {
  /// Create new binary classification metrics from confusion matrix components
  ///
  /// # Arguments
  /// * `tp` - True positives
  /// * `tn` - True negatives
  /// * `fp` - False positives
  /// * `fn_` - False negatives
  pub fn new(tp: usize, tn: usize, fp: usize, fn_: usize) -> Self {
    Self { tp, tn, fp, fn_ }
  }

  /// Compute metrics from predictions and targets
  ///
  /// # Arguments
  /// * `predictions` - Model predictions (probabilities between 0 and 1)
  /// * `targets` - True binary labels (0 or 1)
  /// * `threshold` - Decision threshold (default: 0.5)
  ///
  /// # Returns
  /// Binary classification metrics
  pub fn from_predictions(predictions: &Tensor, targets: &Tensor, threshold: f64) -> Result<Self> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );
    assert!(
      (0.0..=1.0).contains(&threshold),
      "Threshold must be between 0 and 1"
    );

    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for i in 0..predictions.shape().0 {
      for j in 0..predictions.shape().1 {
        let pred_prob = predictions.data[[i, j]];
        let true_label = targets.data[[i, j]];

        let predicted_label = if pred_prob >= threshold { 1.0 } else { 0.0 };

        match (true_label as u8, predicted_label as u8) {
          (1, 1) => tp += 1,  // True Positive
          (0, 0) => tn += 1,  // True Negative
          (0, 1) => fp += 1,  // False Positive
          (1, 0) => fn_ += 1, // False Negative
          _ => panic!("Invalid label values. Expected 0 or 1."),
        }
      }
    }

    Ok(Self::new(tp, tn, fp, fn_))
  }

  /// Total number of samples
  pub fn total(&self) -> usize {
    self.tp + self.tn + self.fp + self.fn_
  }

  /// Total number of positive samples (actual positives)
  pub fn actual_positives(&self) -> usize {
    self.tp + self.fn_
  }

  /// Total number of negative samples (actual negatives)
  pub fn actual_negatives(&self) -> usize {
    self.tn + self.fp
  }

  /// Total number of predicted positives
  pub fn predicted_positives(&self) -> usize {
    self.tp + self.fp
  }

  /// Total number of predicted negatives
  pub fn predicted_negatives(&self) -> usize {
    self.tn + self.fn_
  }

  /// Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
  ///
  /// Accuracy measures the proportion of correct predictions
  pub fn accuracy(&self) -> f64 {
    let total = self.total();
    if total == 0 {
      return 1.0; // Perfect accuracy for empty set
    }
    (self.tp + self.tn) as f64 / total as f64
  }

  /// Compute precision: TP / (TP + FP)
  ///
  /// Precision measures the proportion of positive predictions that were correct
  pub fn precision(&self) -> f64 {
    let predicted_positive = self.predicted_positives();
    if predicted_positive == 0 {
      return 1.0; // Perfect precision when no positive predictions
    }
    self.tp as f64 / predicted_positive as f64
  }

  /// Compute recall (sensitivity): TP / (TP + FN)
  ///
  /// Recall measures the proportion of actual positives that were correctly identified
  pub fn recall(&self) -> f64 {
    let actual_positive = self.actual_positives();
    if actual_positive == 0 {
      return 1.0; // Perfect recall when no actual positives
    }
    self.tp as f64 / actual_positive as f64
  }

  /// Compute specificity: TN / (TN + FP)
  ///
  /// Specificity measures the proportion of actual negatives that were correctly identified
  pub fn specificity(&self) -> f64 {
    let actual_negative = self.actual_negatives();
    if actual_negative == 0 {
      return 1.0; // Perfect specificity when no actual negatives
    }
    self.tn as f64 / actual_negative as f64
  }

  /// Compute F1 score: 2 * (precision * recall) / (precision + recall)
  ///
  /// F1 score is the harmonic mean of precision and recall
  pub fn f1_score(&self) -> f64 {
    let precision = self.precision();
    let recall = self.recall();

    if precision + recall == 0.0 {
      return 0.0;
    }

    2.0 * (precision * recall) / (precision + recall)
  }

  /// Compute Matthews Correlation Coefficient (MCC)
  ///
  /// MCC is a balanced measure that considers all four confusion matrix categories
  /// Returns a value between -1 and +1, where +1 represents perfect prediction,
  /// 0 represents random prediction, and -1 represents total disagreement
  pub fn matthews_correlation_coefficient(&self) -> f64 {
    let numerator = (self.tp * self.tn) as i64 - (self.fp * self.fn_) as i64;
    let denominator =
      ((self.tp + self.fp) * (self.tp + self.fn_) * (self.tn + self.fp) * (self.tn + self.fn_))
        as f64;

    if denominator == 0.0 {
      return 0.0;
    }

    numerator as f64 / denominator.sqrt()
  }
}

/// Trait for computing evaluation metrics
pub trait Metric {
  /// Compute the metric value
  ///
  /// # Arguments
  /// * `predictions` - Model predictions
  /// * `targets` - True target labels
  ///
  /// # Returns
  /// Metric value as f64
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64>;

  /// Get the name of the metric
  fn name(&self) -> &'static str;

  /// Whether higher values indicate better performance
  fn higher_is_better(&self) -> bool {
    true
  }
}

/// Accuracy metric for binary classification
#[derive(Debug, Clone)]
pub struct Accuracy {
  threshold: f64,
}

impl Accuracy {
  /// Create new accuracy metric
  ///
  /// # Arguments
  /// * `threshold` - Decision threshold (default: 0.5)
  pub fn new(threshold: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&threshold),
      "Threshold must be between 0 and 1"
    );
    Self { threshold }
  }

  /// Create accuracy metric with default threshold of 0.5
  pub fn default_threshold() -> Self {
    Self::new(0.5)
  }
}

impl Default for Accuracy {
  fn default() -> Self {
    Self::default_threshold()
  }
}

impl Metric for Accuracy {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    let metrics =
      BinaryClassificationMetrics::from_predictions(predictions, targets, self.threshold)?;
    Ok(metrics.accuracy())
  }

  fn name(&self) -> &'static str {
    "accuracy"
  }
}

/// Precision metric for binary classification
#[derive(Debug, Clone)]
pub struct Precision {
  threshold: f64,
}

impl Precision {
  /// Create new precision metric
  pub fn new(threshold: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&threshold),
      "Threshold must be between 0 and 1"
    );
    Self { threshold }
  }

  /// Create precision metric with default threshold of 0.5
  pub fn default_threshold() -> Self {
    Self::new(0.5)
  }
}

impl Default for Precision {
  fn default() -> Self {
    Self::default_threshold()
  }
}

impl Metric for Precision {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    let metrics =
      BinaryClassificationMetrics::from_predictions(predictions, targets, self.threshold)?;
    Ok(metrics.precision())
  }

  fn name(&self) -> &'static str {
    "precision"
  }
}

/// Recall metric for binary classification
#[derive(Debug, Clone)]
pub struct Recall {
  threshold: f64,
}

impl Recall {
  /// Create new recall metric
  pub fn new(threshold: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&threshold),
      "Threshold must be between 0 and 1"
    );
    Self { threshold }
  }

  /// Create recall metric with default threshold of 0.5
  pub fn default_threshold() -> Self {
    Self::new(0.5)
  }
}

impl Default for Recall {
  fn default() -> Self {
    Self::default_threshold()
  }
}

impl Metric for Recall {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    let metrics =
      BinaryClassificationMetrics::from_predictions(predictions, targets, self.threshold)?;
    Ok(metrics.recall())
  }

  fn name(&self) -> &'static str {
    "recall"
  }
}

/// F1 Score metric for binary classification
#[derive(Debug, Clone)]
pub struct F1Score {
  threshold: f64,
}

impl F1Score {
  /// Create new F1 score metric
  pub fn new(threshold: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&threshold),
      "Threshold must be between 0 and 1"
    );
    Self { threshold }
  }

  /// Create F1 score metric with default threshold of 0.5
  pub fn default_threshold() -> Self {
    Self::new(0.5)
  }
}

impl Default for F1Score {
  fn default() -> Self {
    Self::default_threshold()
  }
}

impl Metric for F1Score {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    let metrics =
      BinaryClassificationMetrics::from_predictions(predictions, targets, self.threshold)?;
    Ok(metrics.f1_score())
  }

  fn name(&self) -> &'static str {
    "f1_score"
  }
}

/// Categorical accuracy metric for multi-class classification
#[derive(Debug, Clone, Default)]
pub struct CategoricalAccuracy;

impl Metric for CategoricalAccuracy {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    let rows = predictions.shape().0;
    if rows == 0 {
      return Ok(1.0);
    }

    let pred_cols = predictions.shape().1;
    let target_cols = targets.shape().1;
    assert!(
      target_cols == 1 || target_cols == pred_cols,
      "Targets must be class indices (n x 1) or one-hot vectors"
    );

    let mut correct = 0;

    for i in 0..rows {
      let pred_class = predictions
        .data
        .row(i)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

      let target_class = if target_cols == 1 {
        targets.data[[i, 0]]
          .round()
          .clamp(0.0, (pred_cols.saturating_sub(1)) as f64) as usize
      } else {
        targets
          .data
          .row(i)
          .iter()
          .enumerate()
          .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
          .map(|(idx, _)| idx)
          .unwrap_or(0)
      };

      if pred_class == target_class {
        correct += 1;
      }
    }

    Ok(correct as f64 / rows as f64)
  }

  fn name(&self) -> &'static str {
    "categorical_accuracy"
  }
}

/// Mean squared error metric for regression tasks
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredErrorMetric;

impl Metric for MeanSquaredErrorMetric {
  fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have identical shapes"
    );

    let diff = targets.sub(predictions)?;
    let squared = diff.mul(&diff)?;
    let mse = squared.mean()?;
    Ok(mse.data[[0, 0]])
  }

  fn name(&self) -> &'static str {
    "mse"
  }

  fn higher_is_better(&self) -> bool {
    false
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_binary_classification_metrics_creation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);

    assert_eq!(metrics.tp, 10);
    assert_eq!(metrics.tn, 20);
    assert_eq!(metrics.fp, 5);
    assert_eq!(metrics.fn_, 3);

    assert_eq!(metrics.total(), 38);
    assert_eq!(metrics.actual_positives(), 13);
    assert_eq!(metrics.actual_negatives(), 25);
    assert_eq!(metrics.predicted_positives(), 15);
    assert_eq!(metrics.predicted_negatives(), 23);
  }

  #[test]
  fn test_binary_classification_metrics_from_predictions() {
    let predictions = Tensor::new(vec![vec![0.8, 0.2, 0.6, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let metrics =
      BinaryClassificationMetrics::from_predictions(&predictions, &targets, 0.5).unwrap();

    // Predictions: [1, 0, 1, 0] (with threshold 0.5)
    // Targets:     [1, 0, 1, 0]
    // All correct: TP=2, TN=2, FP=0, FN=0
    assert_eq!(metrics.tp, 2);
    assert_eq!(metrics.tn, 2);
    assert_eq!(metrics.fp, 0);
    assert_eq!(metrics.fn_, 0);
  }

  #[test]
  fn test_binary_classification_metrics_from_predictions_with_errors() {
    let predictions = Tensor::new(vec![vec![0.8, 0.7, 0.3, 0.2]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let metrics =
      BinaryClassificationMetrics::from_predictions(&predictions, &targets, 0.5).unwrap();

    // Predictions: [1, 1, 0, 0] (with threshold 0.5)
    // Targets:     [1, 0, 1, 0]
    // TP=1 (pred=1, true=1), TN=1 (pred=0, true=0), FP=1 (pred=1, true=0), FN=1 (pred=0, true=1)
    assert_eq!(metrics.tp, 1);
    assert_eq!(metrics.tn, 1);
    assert_eq!(metrics.fp, 1);
    assert_eq!(metrics.fn_, 1);
  }

  #[test]
  fn test_accuracy_calculation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let accuracy = metrics.accuracy();

    // Accuracy = (TP + TN) / Total = (10 + 20) / 38 = 30/38 ≈ 0.789
    assert_abs_diff_eq!(accuracy, 30.0 / 38.0, epsilon = 1e-6);
  }

  #[test]
  fn test_precision_calculation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let precision = metrics.precision();

    // Precision = TP / (TP + FP) = 10 / (10 + 5) = 10/15 ≈ 0.667
    assert_abs_diff_eq!(precision, 10.0 / 15.0, epsilon = 1e-6);
  }

  #[test]
  fn test_recall_calculation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let recall = metrics.recall();

    // Recall = TP / (TP + FN) = 10 / (10 + 3) = 10/13 ≈ 0.769
    assert_abs_diff_eq!(recall, 10.0 / 13.0, epsilon = 1e-6);
  }

  #[test]
  fn test_specificity_calculation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let specificity = metrics.specificity();

    // Specificity = TN / (TN + FP) = 20 / (20 + 5) = 20/25 = 0.8
    assert_abs_diff_eq!(specificity, 0.8, epsilon = 1e-6);
  }

  #[test]
  fn test_f1_score_calculation() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let f1_score = metrics.f1_score();

    let precision = 10.0 / 15.0; // ≈ 0.667
    let recall = 10.0 / 13.0; // ≈ 0.769
    let expected_f1 = 2.0 * (precision * recall) / (precision + recall);

    assert_abs_diff_eq!(f1_score, expected_f1, epsilon = 1e-6);
  }

  #[test]
  fn test_matthews_correlation_coefficient() {
    let metrics = BinaryClassificationMetrics::new(10, 20, 5, 3);
    let mcc = metrics.matthews_correlation_coefficient();

    // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    let numerator = (10 * 20) - (5 * 3); // 200 - 15 = 185
    let denominator = ((10 + 5) * (10 + 3) * (20 + 5) * (20 + 3)) as f64; // 15*13*25*23 = 111975
    let expected_mcc = numerator as f64 / denominator.sqrt();

    assert_abs_diff_eq!(mcc, expected_mcc, epsilon = 1e-6);
  }

  #[test]
  fn test_perfect_classification() {
    let metrics = BinaryClassificationMetrics::new(10, 10, 0, 0);

    assert_abs_diff_eq!(metrics.accuracy(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(metrics.precision(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(metrics.recall(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(metrics.f1_score(), 1.0, epsilon = 1e-6);
  }

  #[test]
  fn test_edge_cases() {
    // No positive predictions
    let metrics_no_pred_pos = BinaryClassificationMetrics::new(0, 10, 0, 5);
    assert_abs_diff_eq!(metrics_no_pred_pos.precision(), 1.0, epsilon = 1e-6);

    // No actual positives
    let metrics_no_actual_pos = BinaryClassificationMetrics::new(0, 10, 5, 0);
    assert_abs_diff_eq!(metrics_no_actual_pos.recall(), 1.0, epsilon = 1e-6);

    // No actual negatives
    let metrics_no_actual_neg = BinaryClassificationMetrics::new(10, 0, 0, 5);
    assert_abs_diff_eq!(metrics_no_actual_neg.specificity(), 1.0, epsilon = 1e-6);
  }

  #[test]
  fn test_accuracy_metric() {
    let accuracy_metric = Accuracy::new(0.6);
    assert_eq!(accuracy_metric.name(), "accuracy");
    assert!(accuracy_metric.higher_is_better());

    let predictions = Tensor::new(vec![vec![0.8, 0.3, 0.7, 0.2]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let accuracy = accuracy_metric.compute(&predictions, &targets).unwrap();

    // With threshold 0.6: predictions become [1, 0, 1, 0], targets are [1, 0, 1, 0]
    // All correct, so accuracy = 1.0
    assert_abs_diff_eq!(accuracy, 1.0, epsilon = 1e-6);
  }

  #[test]
  fn test_precision_metric() {
    let precision_metric = Precision::default();
    assert_eq!(precision_metric.name(), "precision");

    let predictions = Tensor::new(vec![vec![0.8, 0.7, 0.3, 0.2]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let precision = precision_metric.compute(&predictions, &targets).unwrap();

    // Predictions: [1, 1, 0, 0], Targets: [1, 0, 1, 0]
    // TP=1, FP=1, so precision = 1/(1+1) = 0.5
    assert_abs_diff_eq!(precision, 0.5, epsilon = 1e-6);
  }

  #[test]
  fn test_recall_metric() {
    let recall_metric = Recall::default();
    assert_eq!(recall_metric.name(), "recall");

    let predictions = Tensor::new(vec![vec![0.8, 0.7, 0.3, 0.2]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let recall = recall_metric.compute(&predictions, &targets).unwrap();

    // Predictions: [1, 1, 0, 0], Targets: [1, 0, 1, 0]
    // TP=1, FN=1, so recall = 1/(1+1) = 0.5
    assert_abs_diff_eq!(recall, 0.5, epsilon = 1e-6);
  }

  #[test]
  fn test_f1_score_metric() {
    let f1_metric = F1Score::default();
    assert_eq!(f1_metric.name(), "f1_score");

    let predictions = Tensor::new(vec![vec![0.8, 0.7, 0.3, 0.2]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0, 1.0, 0.0]]).unwrap();

    let f1_score = f1_metric.compute(&predictions, &targets).unwrap();

    // Precision = 0.5, Recall = 0.5, F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    assert_abs_diff_eq!(f1_score, 0.5, epsilon = 1e-6);
  }

  #[test]
  #[should_panic(expected = "Threshold must be between 0 and 1")]
  fn test_invalid_threshold() {
    Accuracy::new(1.5);
  }

  #[test]
  #[should_panic(expected = "Predictions and targets must have the same shape")]
  fn test_shape_mismatch() {
    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0]]).unwrap();

    BinaryClassificationMetrics::from_predictions(&predictions, &targets, 0.5).unwrap();
  }

  #[test]
  fn test_categorical_accuracy_one_hot() {
    let predictions = Tensor::new(vec![vec![0.2, 0.8], vec![0.7, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    let metric = CategoricalAccuracy;
    let score = metric.compute(&predictions, &targets).unwrap();
    assert_abs_diff_eq!(score, 1.0, epsilon = 1e-9);
  }

  #[test]
  fn test_categorical_accuracy_class_indices() {
    let predictions = Tensor::new(vec![vec![0.6, 0.4], vec![0.1, 0.9]]).unwrap();
    let targets = Tensor::new(vec![vec![0.0], vec![1.0]]).unwrap();
    let metric = CategoricalAccuracy;
    let score = metric.compute(&predictions, &targets).unwrap();
    assert_abs_diff_eq!(score, 1.0, epsilon = 1e-9);
  }

  #[test]
  fn test_mean_squared_error_metric() {
    let predictions = Tensor::new(vec![vec![1.0], vec![2.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0], vec![3.0]]).unwrap();
    let metric = MeanSquaredErrorMetric;
    let score = metric.compute(&predictions, &targets).unwrap();
    assert_abs_diff_eq!(score, 0.5, epsilon = 1e-9);
    assert!(!metric.higher_is_better());
  }
}
