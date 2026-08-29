use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
use ndarray::Array2;
use std::fs::File;
use std::path::PathBuf;

/// Base trait for CSV data loaders
pub trait CsvLoader {
  /// Get the task type for this dataset
  fn task_kind(&self) -> TaskKind;

  /// Parse a single CSV record into features and target
  fn parse_record(&self, record: &csv::StringRecord) -> Result<(Vec<f64>, f64)>;

  /// Validate the CSV record format
  fn validate_record(&self, record: &csv::StringRecord) -> Result<()>;

  /// Whether the CSV has headers
  fn has_headers(&self) -> bool {
    false
  }
}

/// Base CSV repository implementation
pub struct BaseCsvRepository<T: CsvLoader> {
  path: PathBuf,
  loader: T,
}

impl<T: CsvLoader> BaseCsvRepository<T> {
  pub fn new<P: Into<PathBuf>>(path: P, loader: T) -> Self {
    Self {
      path: path.into(),
      loader,
    }
  }

  fn read_records(&self) -> Result<(Array2<f64>, Array2<f64>)> {
    let file = File::open(&self.path)?;
    let mut reader = csv::ReaderBuilder::new()
      .has_headers(self.loader.has_headers())
      .from_reader(file);

    let mut features_vec = Vec::new();
    let mut targets_vec = Vec::new();
    let mut n_features = None;

    for (row_idx, result) in reader.records().enumerate() {
      let record = result?;

      // Validate record format
      self.loader.validate_record(&record)?;

      // Parse record
      let (features, target) = self.loader.parse_record(&record).map_err(|e| {
        TensorError::InvalidValue(format!("Error parsing row {}: {}", row_idx + 1, e))
      })?;

      // Check feature dimension consistency
      if let Some(expected_features) = n_features {
        if features.len() != expected_features {
          return Err(TensorError::InvalidValue(format!(
            "Inconsistent number of features at row {}: expected {}, got {}",
            row_idx + 1,
            expected_features,
            features.len()
          )));
        }
      } else {
        n_features = Some(features.len());
      }

      features_vec.push(features);
      targets_vec.push(target);
    }

    if features_vec.is_empty() {
      return Err(TensorError::InvalidValue(
        "No data found in CSV file".to_string(),
      ));
    }

    let n_samples = features_vec.len();
    let n_features = n_features.unwrap();

    // Convert to matrices
    let mut features_matrix = Array2::zeros((n_samples, n_features));
    let mut targets_matrix = Array2::zeros((n_samples, 1));

    for (i, (features, target)) in features_vec.into_iter().zip(targets_vec).enumerate() {
      for (j, feature) in features.into_iter().enumerate() {
        features_matrix[[i, j]] = feature;
      }
      targets_matrix[[i, 0]] = target;
    }

    Ok((features_matrix, targets_matrix))
  }
}

impl<T: CsvLoader> DataRepository for BaseCsvRepository<T> {
  fn load_dataset(&self, config: &DataConfig) -> Result<Dataset> {
    let (features_matrix, targets) = self.read_records()?;

    let mut dataset = Dataset::new(
      self.loader.task_kind(),
      features_matrix,
      targets,
      config.clone(),
    );

    dataset.preprocess()?;
    Ok(dataset)
  }
}
