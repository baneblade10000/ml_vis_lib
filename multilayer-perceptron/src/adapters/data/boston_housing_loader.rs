use crate::adapters::data::base_csv_loader::{BaseCsvRepository, CsvLoader};
use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
use std::path::PathBuf;

/// Boston Housing dataset loader
pub struct BostonHousingLoader;

impl CsvLoader for BostonHousingLoader {
  fn task_kind(&self) -> TaskKind {
    TaskKind::Regression
  }

  fn parse_record(&self, record: &csv::StringRecord) -> Result<(Vec<f64>, f64)> {
    let mut features = Vec::new();

    // Boston Housing has 13 features + 1 target (14 columns total)
    // Features are columns 0-12, target is column 13 (house price)
    for i in 0..13 {
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

    // Parse target (house price in $1000s)
    let raw_target = record.get(13).ok_or_else(|| {
      TensorError::InvalidValue("Missing target column for Boston Housing data".to_string())
    })?;

    let target: f64 = raw_target.trim().parse().map_err(|e| {
      TensorError::InvalidValue(format!(
        "Invalid target value: {} (error: {})",
        raw_target, e
      ))
    })?;

    Ok((features, target))
  }

  fn validate_record(&self, record: &csv::StringRecord) -> Result<()> {
    if record.len() != 14 {
      return Err(TensorError::InvalidValue(format!(
        "Expected 14 columns for Boston Housing data, got {}",
        record.len()
      )));
    }
    Ok(())
  }

  fn has_headers(&self) -> bool {
    false
  }
}

pub struct BostonHousingRepository {
  inner: BaseCsvRepository<BostonHousingLoader>,
}

impl BostonHousingRepository {
  pub fn new<P: Into<PathBuf>>(path: P) -> Self {
    Self {
      inner: BaseCsvRepository::new(path, BostonHousingLoader),
    }
  }
}

impl DataRepository for BostonHousingRepository {
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
  fn test_boston_housing_loader() {
    #[cfg(not(target_arch = "wasm32"))]
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    #[cfg(target_arch = "wasm32")]
    let timestamp = 12345u128;
    let csv_path = std::env::temp_dir().join(format!("boston_test_{timestamp}.csv"));

    let mut file = File::create(&csv_path).unwrap();
    // Sample Boston Housing record with 13 features + 1 target
    writeln!(
      file,
      "0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98,24.00"
    )
    .unwrap();
    writeln!(
      file,
      "0.02731,0.00,7.070,0,0.4690,6.4210,78.90,4.9671,2,242.0,17.80,396.90,9.14,21.60"
    )
    .unwrap();

    let repo = BostonHousingRepository::new(&csv_path);
    let dataset = repo
      .load_dataset(&PreprocessConfig::default())
      .expect("Failed to load dataset");

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.n_features(), 13);

    remove_file(csv_path).unwrap();
  }
}
