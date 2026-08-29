use crate::core::Result;
use crate::domain::models::mlp::MLP;
use crate::domain::types::{DataConfig, Dataset};

pub trait DataRepository {
  fn load_dataset(&self, config: &DataConfig) -> Result<Dataset>;
}

pub trait ModelRepository {
  fn save_model(&self, model: &MLP, path: &str) -> Result<()>;
  fn load_model(&self, path: &str) -> Result<MLP>;
}
