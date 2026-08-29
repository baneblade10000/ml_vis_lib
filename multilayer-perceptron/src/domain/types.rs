use crate::core::{Result, Tensor, TensorError};
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
  BinaryClassification,
  MultiClassification,
  Regression,
}

#[derive(Debug, Clone)]
pub struct DataConfig {
  /// Whether to apply standardization (z-score normalization)
  pub standardize: bool,
  /// Whether to apply min-max normalization
  pub normalize: bool,
  /// Random seed for reproducible train/test splits
  pub random_seed: Option<u64>,
}

impl Default for DataConfig {
  fn default() -> Self {
    Self {
      standardize: true,
      normalize: false,
      random_seed: Some(42),
    }
  }
}

pub type PreprocessConfig = DataConfig;

#[derive(Debug, Clone)]
pub struct FeatureStats {
  pub means: Array1<f64>,
  pub stds: Array1<f64>,
  pub mins: Array1<f64>,
  pub maxs: Array1<f64>,
}

impl FeatureStats {
  /// Compute statistics from feature matrix
  pub fn from_features(features: &Array2<f64>) -> Self {
    let means = features
      .mean_axis(Axis(0))
      .unwrap_or_else(|| Array1::zeros(features.ncols()));
    let stds = features.std_axis(Axis(0), 0.0);
    let mins = features.fold_axis(Axis(0), f64::INFINITY, |&acc, &x| acc.min(x));
    let maxs = features.fold_axis(Axis(0), f64::NEG_INFINITY, |&acc, &x| acc.max(x));

    Self {
      means,
      stds,
      mins,
      maxs,
    }
  }

  /// Apply standardization to features using computed statistics
  pub fn standardize(&self, features: &mut Array2<f64>) -> Result<()> {
    if features.ncols() != self.means.len() {
      return Err(TensorError::DimensionMismatch(
        "Feature dimension mismatch".to_string(),
      ));
    }

    for mut row in features.rows_mut() {
      for (i, value) in row.iter_mut().enumerate() {
        if self.stds[i] != 0.0 {
          *value = (*value - self.means[i]) / self.stds[i];
        }
      }
    }
    Ok(())
  }

  /// Apply min-max normalization to features using computed statistics
  pub fn normalize(&self, features: &mut Array2<f64>) -> Result<()> {
    if features.ncols() != self.mins.len() {
      return Err(TensorError::DimensionMismatch(
        "Feature dimension mismatch".to_string(),
      ));
    }

    for mut row in features.rows_mut() {
      for (i, value) in row.iter_mut().enumerate() {
        let range = self.maxs[i] - self.mins[i];
        if range != 0.0 {
          *value = (*value - self.mins[i]) / range;
        }
      }
    }
    Ok(())
  }
}

#[derive(Debug, Clone)]
pub struct Dataset {
  pub task: TaskKind,
  pub features: Array2<f64>,
  pub targets: Array2<f64>,
  pub sample_ids: Option<Vec<String>>,
  pub feature_names: Option<Vec<String>>,
  pub target_names: Option<Vec<String>>,
  pub stats: Option<FeatureStats>,
  pub config: DataConfig,
}

impl Dataset {
  pub fn new(
    task: TaskKind,
    features: Array2<f64>,
    targets: Array2<f64>,
    config: DataConfig,
  ) -> Self {
    Self {
      task,
      features,
      targets,
      sample_ids: None,
      feature_names: None,
      target_names: None,
      stats: None,
      config,
    }
  }

  pub fn with_sample_ids(mut self, sample_ids: Vec<String>) -> Self {
    self.sample_ids = Some(sample_ids);
    self
  }

  pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Self {
    self.feature_names = Some(feature_names);
    self
  }

  pub fn with_target_names(mut self, target_names: Vec<String>) -> Self {
    self.target_names = Some(target_names);
    self
  }

  pub fn len(&self) -> usize {
    self.features.nrows()
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn n_features(&self) -> usize {
    self.features.ncols()
  }

  pub fn target_dim(&self) -> usize {
    self.targets.ncols()
  }

  pub fn preprocess(&mut self) -> Result<()> {
    if self.stats.is_none() {
      self.stats = Some(FeatureStats::from_features(&self.features));
    }

    if let Some(stats) = &self.stats {
      if self.config.standardize {
        stats.standardize(&mut self.features)?;
      }

      if self.config.normalize {
        stats.normalize(&mut self.features)?;
      }
    }

    Ok(())
  }

  pub fn preprocess_with_stats(&mut self, stats: &FeatureStats) -> Result<()> {
    if self.config.standardize {
      stats.standardize(&mut self.features)?;
    }

    if self.config.normalize {
      stats.normalize(&mut self.features)?;
    }

    Ok(())
  }

  pub fn train_test_split(&self, test_size: f64) -> Result<(Dataset, Dataset)> {
    if test_size <= 0.0 || test_size >= 1.0 {
      return Err(TensorError::InvalidValue(
        "Test size must be between 0 and 1".to_string(),
      ));
    }

    let total = self.len();
    if total == 0 {
      return Err(TensorError::InvalidValue(
        "Dataset is empty; cannot split".to_string(),
      ));
    }

    let n_test = (total as f64 * test_size).round() as usize;
    let n_train = total.saturating_sub(n_test.max(1));

    let mut indices: Vec<usize> = (0..total).collect();
    if let Some(seed) = self.config.random_seed {
      let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
      indices.shuffle(&mut rng);
    } else {
      let mut rng = rand::thread_rng();
      indices.shuffle(&mut rng);
    }

    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    let train_dataset = self.subset(train_indices)?;
    let test_dataset = self.subset(test_indices)?;

    Ok((train_dataset, test_dataset))
  }

  pub fn to_tensors(&self) -> Result<(Tensor, Tensor)> {
    let features_tensor = Tensor::from_array2(self.features.clone())?;
    let targets_tensor = Tensor::from_array2(self.targets.clone())?;
    Ok((features_tensor, targets_tensor))
  }

  pub fn subset(&self, indices: &[usize]) -> Result<Dataset> {
    let (sample_features, sample_targets) = self.select_rows(indices)?;
    let sample_ids = self.sample_ids.as_ref().map(|ids| {
      indices
        .iter()
        .map(|&i| ids.get(i).cloned().unwrap_or_default())
        .collect()
    });

    Ok(Dataset {
      task: self.task,
      features: sample_features,
      targets: sample_targets,
      sample_ids,
      feature_names: self.feature_names.clone(),
      target_names: self.target_names.clone(),
      stats: self.stats.clone(),
      config: self.config.clone(),
    })
  }

  pub fn stats(&self) -> Option<&FeatureStats> {
    self.stats.as_ref()
  }

  pub fn class_distribution(&self) -> Option<HashMap<String, usize>> {
    match self.task {
      TaskKind::Regression => None,
      TaskKind::BinaryClassification | TaskKind::MultiClassification => {
        let mut distribution = HashMap::new();
        if self.target_dim() == 1 {
          for &value in self.targets.column(0).iter() {
            let class_label = format!("{}", value.round());
            *distribution.entry(class_label).or_insert(0) += 1;
          }
        } else {
          for row in self.targets.rows() {
            if let Some((class_idx, _)) = row
              .iter()
              .enumerate()
              .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
              let class_label = class_idx.to_string();
              *distribution.entry(class_label).or_insert(0) += 1;
            }
          }
        }
        Some(distribution)
      }
    }
  }

  fn select_rows(&self, indices: &[usize]) -> Result<(Array2<f64>, Array2<f64>)> {
    let n_samples = self.len();
    let feature_dim = self.n_features();
    let target_dim = self.target_dim();

    let mut features = Array2::zeros((indices.len(), feature_dim));
    let mut targets = Array2::zeros((indices.len(), target_dim));

    for (row_idx, &source_idx) in indices.iter().enumerate() {
      if source_idx >= n_samples {
        return Err(TensorError::InvalidValue("Index out of bounds".to_string()));
      }
      features
        .row_mut(row_idx)
        .assign(&self.features.row(source_idx));
      targets
        .row_mut(row_idx)
        .assign(&self.targets.row(source_idx));
    }

    Ok((features, targets))
  }
}

/// Data loader for mini-batch training
#[derive(Debug)]
pub struct DataLoader {
  dataset: Dataset,
  batch_size: usize,
  shuffle: bool,
  rng: rand::rngs::StdRng,
  indices: Vec<usize>,
  current_pos: usize,
}

impl DataLoader {
  pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool, seed: Option<u64>) -> Self {
    let n_samples = dataset.len();
    let indices: Vec<usize> = (0..n_samples).collect();

    let rng = if let Some(seed) = seed {
      rand::rngs::StdRng::seed_from_u64(seed)
    } else {
      rand::rngs::StdRng::from_entropy()
    };

    Self {
      dataset,
      batch_size: batch_size.max(1),
      shuffle,
      rng,
      indices,
      current_pos: 0,
    }
  }

  pub fn len(&self) -> usize {
    self.dataset.len()
  }

  pub fn is_empty(&self) -> bool {
    self.dataset.is_empty()
  }

  pub fn num_batches(&self) -> usize {
    self.dataset.len().div_ceil(self.batch_size.max(1)).max(1)
  }

  pub fn reset(&mut self) {
    self.current_pos = 0;
    if self.shuffle {
      self.indices.shuffle(&mut self.rng);
    }
  }

  pub fn next_batch(&mut self) -> Option<Result<(Tensor, Tensor)>> {
    if self.current_pos >= self.dataset.len() {
      return None;
    }

    let end_pos = (self.current_pos + self.batch_size).min(self.dataset.len());
    let batch_indices = &self.indices[self.current_pos..end_pos];

    let (features, targets) = match self.dataset.select_rows(batch_indices) {
      Ok(data) => data,
      Err(err) => return Some(Err(err)),
    };

    self.current_pos = end_pos;

    Some(
      Tensor::from_array2(features).and_then(|fx| Tensor::from_array2(targets).map(|ty| (fx, ty))),
    )
  }

  pub fn batches(&mut self) -> DataLoaderIterator<'_> {
    self.reset();
    DataLoaderIterator { loader: self }
  }

  pub fn dataset(&self) -> &Dataset {
    &self.dataset
  }

  pub fn batch_size(&self) -> usize {
    self.batch_size
  }
}

pub struct DataLoaderIterator<'a> {
  loader: &'a mut DataLoader,
}

impl Iterator for DataLoaderIterator<'_> {
  type Item = Result<(Tensor, Tensor)>;

  fn next(&mut self) -> Option<Self::Item> {
    self.loader.next_batch()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  fn dummy_dataset(task: TaskKind) -> Dataset {
    let features =
      Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 0.0, -1.0, 2.0, 2.0, -2.0]).unwrap();
    let targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    Dataset::new(task, features, targets, DataConfig::default())
  }

  #[test]
  fn test_dataset_preprocess() {
    let mut dataset = dummy_dataset(TaskKind::Regression);
    dataset.preprocess().unwrap();
    assert!(dataset.stats.is_some());
  }

  #[test]
  fn test_dataset_train_test_split() {
    let dataset = dummy_dataset(TaskKind::BinaryClassification);
    let (train, test) = dataset.train_test_split(0.25).unwrap();
    assert_eq!(train.len(), 3);
    assert_eq!(test.len(), 1);
    assert_eq!(train.n_features(), 2);
  }

  #[test]
  fn test_dataset_subset() {
    let dataset = dummy_dataset(TaskKind::BinaryClassification).with_sample_ids(vec![
      "a".into(),
      "b".into(),
      "c".into(),
      "d".into(),
    ]);
    let subset = dataset.subset(&[1, 3]).unwrap();
    assert_eq!(subset.len(), 2);
    assert_eq!(subset.sample_ids.unwrap(), vec!["b", "d"]);
  }

  #[test]
  fn test_class_distribution_binary() {
    let dataset = dummy_dataset(TaskKind::BinaryClassification);
    let distribution = dataset.class_distribution().unwrap();
    assert_eq!(distribution.get("0").copied().unwrap_or_default(), 2);
    assert_eq!(distribution.get("1").copied().unwrap_or_default(), 2);
  }

  #[test]
  fn test_dataloader_iteration() {
    let dataset = dummy_dataset(TaskKind::Regression);
    let mut loader = DataLoader::new(dataset, 2, false, Some(42));
    let mut batches = loader.batches();
    let (features, targets) = batches.next().unwrap().unwrap();
    assert_eq!(features.shape(), (2, 2));
    assert_eq!(targets.shape(), (2, 1));
  }

  #[test]
  fn test_feature_stats_normalize() {
    let mut features =
      Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape should match");
    let stats = FeatureStats::from_features(&features);
    stats.normalize(&mut features).unwrap();
    assert_abs_diff_eq!(features[[0, 0]], 0.0, epsilon = 1e-9);
    assert_abs_diff_eq!(features[[1, 1]], 1.0, epsilon = 1e-9);
  }
}
