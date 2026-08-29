use crate::core::Result;
use crate::domain::types::{DataConfig, Dataset, FeatureStats};

pub trait Transform {
  fn fit(&mut self, dataset: &Dataset) -> Result<()>;
  fn apply(&self, dataset: &mut Dataset) -> Result<()>;
}

pub struct TransformPipeline {
  steps: Vec<Box<dyn Transform + Send + Sync>>,
}

impl TransformPipeline {
  pub fn new() -> Self {
    Self { steps: Vec::new() }
  }

  pub fn push<T>(mut self, transform: T) -> Self
  where
    T: Transform + Send + Sync + 'static,
  {
    self.steps.push(Box::new(transform));
    self
  }

  pub fn is_empty(&self) -> bool {
    self.steps.is_empty()
  }

  pub fn fit(&mut self, dataset: &Dataset) -> Result<()> {
    for step in &mut self.steps {
      step.fit(dataset)?;
    }
    Ok(())
  }

  pub fn apply(&self, dataset: &mut Dataset) -> Result<()> {
    for step in &self.steps {
      step.apply(dataset)?;
    }
    Ok(())
  }
}

impl Default for TransformPipeline {
  fn default() -> Self {
    Self::new()
  }
}

pub fn build_pipeline(config: &DataConfig) -> TransformPipeline {
  let mut pipeline = TransformPipeline::new();

  if config.standardize || config.normalize {
    pipeline = pipeline.push(FeatureScalingTransform::new(
      config.standardize,
      config.normalize,
    ));
  }

  pipeline
}

struct FeatureScalingTransform {
  apply_standardize: bool,
  apply_normalize: bool,
  stats: Option<FeatureStats>,
}

impl FeatureScalingTransform {
  fn new(apply_standardize: bool, apply_normalize: bool) -> Self {
    Self {
      apply_standardize,
      apply_normalize,
      stats: None,
    }
  }
}

impl Transform for FeatureScalingTransform {
  fn fit(&mut self, dataset: &Dataset) -> Result<()> {
    if !(self.apply_standardize || self.apply_normalize) {
      return Ok(());
    }

    self.stats = Some(FeatureStats::from_features(&dataset.features));
    Ok(())
  }

  fn apply(&self, dataset: &mut Dataset) -> Result<()> {
    if !(self.apply_standardize || self.apply_normalize) {
      return Ok(());
    }

    if let Some(stats) = &self.stats {
      dataset.preprocess_with_stats(stats)?;
      dataset.stats = Some(stats.clone());
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::Array2;

  use crate::domain::types::TaskKind;

  fn dummy_dataset() -> Dataset {
    let features =
      Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 0.0, -1.0, 2.0, 2.0, -2.0]).unwrap();
    let targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    Dataset::new(
      TaskKind::BinaryClassification,
      features,
      targets,
      DataConfig::default(),
    )
  }

  #[test]
  fn test_pipeline_fit_apply() {
    let dataset = dummy_dataset();
    let mut pipeline = build_pipeline(&DataConfig {
      standardize: true,
      normalize: false,
      random_seed: None,
    });

    assert!(!pipeline.is_empty());
    pipeline.fit(&dataset).unwrap();

    let mut dataset_clone = dataset.clone();
    pipeline.apply(&mut dataset_clone).unwrap();

    assert!(dataset_clone.stats().is_some());
  }
}
