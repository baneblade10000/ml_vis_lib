use crate::adapters::data::base_csv_loader::{BaseCsvRepository, CsvLoader};
use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
use std::path::PathBuf;

/// Wisconsin Breast Cancer dataset loader
pub struct BreastCancerLoader;

impl CsvLoader for BreastCancerLoader {
  fn task_kind(&self) -> TaskKind {
    TaskKind::BinaryClassification
  }

  fn parse_record(&self, record: &csv::StringRecord) -> Result<(Vec<f64>, f64)> {
    // Record format: ID, Diagnosis (M/B), followed by 30 feature columns
    let id_present = record.len() >= 32;
    if !id_present {
      return Err(TensorError::InvalidValue(
        "Breast cancer record requires ID, diagnosis, and 30 features".to_string(),
      ));
    }

    let diagnosis = record.get(1).ok_or_else(|| {
      TensorError::InvalidValue("Missing diagnosis column in breast cancer data".to_string())
    })?;

    let target = match diagnosis.trim() {
      "M" => 1.0,
      "B" => 0.0,
      other => {
        return Err(TensorError::InvalidValue(format!(
          "Invalid diagnosis value: {}. Expected 'M' or 'B'",
          other
        )))
      }
    };

    let mut features = Vec::with_capacity(30);
    for col_idx in 2..32 {
      let raw_value = record.get(col_idx).ok_or_else(|| {
        TensorError::InvalidValue(format!(
          "Missing feature column {} in breast cancer data",
          col_idx + 1
        ))
      })?;

      let value: f64 = raw_value.trim().parse().map_err(|e| {
        TensorError::InvalidValue(format!(
          "Invalid feature value at column {}: {} (error: {})",
          col_idx + 1,
          raw_value,
          e
        ))
      })?;
      features.push(value);
    }

    Ok((features, target))
  }

  fn validate_record(&self, record: &csv::StringRecord) -> Result<()> {
    if record.len() != 32 {
      return Err(TensorError::InvalidValue(format!(
        "Expected 32 columns for breast cancer data (ID, diagnosis, 30 features), got {}",
        record.len()
      )));
    }
    Ok(())
  }

  fn has_headers(&self) -> bool {
    false
  }
}

pub struct BreastCancerRepository {
  inner: BaseCsvRepository<BreastCancerLoader>,
}

impl BreastCancerRepository {
  pub fn new<P: Into<PathBuf>>(path: P) -> Self {
    Self {
      inner: BaseCsvRepository::new(path, BreastCancerLoader),
    }
  }
}

impl DataRepository for BreastCancerRepository {
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
  fn test_breast_cancer_loader() {
    #[cfg(not(target_arch = "wasm32"))]
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    #[cfg(target_arch = "wasm32")]
    let timestamp = 12345u128;
    let csv_path = std::env::temp_dir().join(format!("breast_cancer_test_{timestamp}.csv"));

    let mut file = File::create(&csv_path).unwrap();
    let one_features = vec!["1.0"; 30].join(",");
    let zero_features = vec!["0.5"; 30].join(",");
    writeln!(file, "1001,M,{one_features}").unwrap();
    writeln!(file, "1002,B,{zero_features}").unwrap();

    let repo = BreastCancerRepository::new(&csv_path);
    let dataset = repo
      .load_dataset(&PreprocessConfig::default())
      .expect("Failed to load dataset");

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.n_features(), 30);

    remove_file(csv_path).unwrap();
  }
}
