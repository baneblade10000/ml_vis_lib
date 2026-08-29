use crate::adapters::data::base_csv_loader::{BaseCsvRepository, CsvLoader};
use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
use std::path::PathBuf;

/// Iris dataset loader
pub struct IrisLoader;

impl CsvLoader for IrisLoader {
  fn task_kind(&self) -> TaskKind {
    TaskKind::MultiClassification
  }

  fn parse_record(&self, record: &csv::StringRecord) -> Result<(Vec<f64>, f64)> {
    let mut features = Vec::new();

    // Iris has 4 features + 1 class label (5 columns total)
    // Features are columns 0-3 (sepal length, sepal width, petal length, petal width)
    // Class is column 4
    for i in 0..4 {
      let raw_value = record
        .get(i)
        .ok_or_else(|| TensorError::InvalidValue(format!("Missing feature column {}", i + 1)))?;

      let feature_value: f64 = raw_value.trim().parse().map_err(|e| {
        TensorError::InvalidValue(format!(
          "Invalid feature value at column {}: {} (error: {})",
          i + 1,
          raw_value,
          e
        ))
      })?;
      features.push(feature_value);
    }

    // Parse class label
    let class_str = record.get(4).ok_or_else(|| {
      TensorError::InvalidValue("Missing class label column for Iris data".to_string())
    })?;
    let class_str = class_str.trim();
    let target = self.parse_iris_class(class_str)?;

    Ok((features, target))
  }

  fn validate_record(&self, record: &csv::StringRecord) -> Result<()> {
    if record.len() != 5 {
      return Err(TensorError::InvalidValue(format!(
        "Expected 5 columns for Iris data, got {}",
        record.len()
      )));
    }
    Ok(())
  }

  fn has_headers(&self) -> bool {
    false
  }
}

impl IrisLoader {
  fn parse_iris_class(&self, class_str: &str) -> Result<f64> {
    match class_str.to_lowercase().as_str() {
      "iris-setosa" | "setosa" => Ok(0.0),
      "iris-versicolor" | "versicolor" => Ok(1.0),
      "iris-virginica" | "virginica" => Ok(2.0),
      _ => Err(TensorError::InvalidValue(format!(
        "Unknown Iris class: {}. Expected 'Iris-setosa', 'Iris-versicolor', or 'Iris-virginica'",
        class_str
      ))),
    }
  }
}

pub struct IrisRepository {
  inner: BaseCsvRepository<IrisLoader>,
}

impl IrisRepository {
  pub fn new<P: Into<PathBuf>>(path: P) -> Self {
    Self {
      inner: BaseCsvRepository::new(path, IrisLoader),
    }
  }
}

impl DataRepository for IrisRepository {
  fn load_dataset(&self, config: &DataConfig) -> Result<Dataset> {
    self.inner.load_dataset(config)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::domain::ports::DataRepository;
  use crate::domain::types::PreprocessConfig;
  use std::fs::{remove_file, File};
  use std::io::Write;

  #[cfg(not(target_arch = "wasm32"))]
  use std::time::{SystemTime, UNIX_EPOCH};

  #[test]
  fn test_iris_loader() {
    #[cfg(not(target_arch = "wasm32"))]
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    #[cfg(target_arch = "wasm32")]
    let timestamp = 12345u128;
    let csv_path = std::env::temp_dir().join(format!("iris_test_{timestamp}.csv"));

    let mut file = File::create(&csv_path).unwrap();
    writeln!(file, "5.1,3.5,1.4,0.2,Iris-setosa").unwrap();
    writeln!(file, "7.0,3.2,4.7,1.4,Iris-versicolor").unwrap();
    writeln!(file, "6.3,3.3,6.0,2.5,Iris-virginica").unwrap();

    let repo = IrisRepository::new(&csv_path);
    let dataset = repo
      .load_dataset(&PreprocessConfig::default())
      .expect("Failed to load dataset");

    assert_eq!(dataset.len(), 3);
    assert_eq!(dataset.n_features(), 4);

    remove_file(csv_path).unwrap();
  }

  #[test]
  fn test_iris_class_parsing() {
    let loader = IrisLoader;

    assert_eq!(loader.parse_iris_class("Iris-setosa").unwrap(), 0.0);
    assert_eq!(loader.parse_iris_class("Iris-versicolor").unwrap(), 1.0);
    assert_eq!(loader.parse_iris_class("Iris-virginica").unwrap(), 2.0);

    // Test case variations
    assert_eq!(loader.parse_iris_class("setosa").unwrap(), 0.0);
    assert_eq!(loader.parse_iris_class("VERSICOLOR").unwrap(), 1.0);

    // Test invalid class
    assert!(loader.parse_iris_class("unknown").is_err());
  }
}
