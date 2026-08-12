//! Serde config mirroring `@ml-vis/core` CnnConfig / LayerSpec.

use serde::{Deserialize, Serialize};

pub const IMAGE_SIZE: usize = 16;
pub const SIGNAL_LENGTH: usize = 48;

fn default_image_size() -> usize {
    IMAGE_SIZE
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum CnnMode {
    #[serde(rename = "2d")]
    D2,
    #[serde(rename = "1d")]
    D1,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ActivationId {
    Relu,
    Tanh,
    Sigmoid,
    Linear,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PoolKind {
    Max,
    Avg,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum LayerKind {
    Conv2d,
    Pool2d,
    Conv1d,
    Pool1d,
    Gap2d,
    Gap1d,
    Flatten,
    Dense,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerSpec {
    pub kind: LayerKind,
    pub filters: Option<usize>,
    pub kernel_size: Option<usize>,
    pub pool_kind: Option<PoolKind>,
    pub units: Option<usize>,
    pub activation: Option<ActivationId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CnnConfig {
    pub mode: CnnMode,
    pub dataset: String,
    pub layers: Vec<LayerSpec>,
    pub learning_rate: f64,
    pub optimizer: String,
    pub activation: ActivationId,
    pub batch_size: usize,
    pub noise: f64,
    pub perc_train_data: f64,
    pub regularization: String,
    pub regularization_rate: f64,
    /// H=W of 2-D inputs. `three-four-loops` uses 32; everything else 16.
    #[serde(default = "default_image_size")]
    pub image_size: usize,
}

impl CnnConfig {
    pub fn image_size(&self) -> usize {
        self.image_size.max(4)
    }
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            mode: CnnMode::D2,
            dataset: "digits".into(),
            layers: vec![
                LayerSpec {
                    kind: LayerKind::Conv2d,
                    filters: Some(4),
                    kernel_size: Some(3),
                    pool_kind: None,
                    units: None,
                    activation: Some(ActivationId::Relu),
                },
                LayerSpec {
                    kind: LayerKind::Pool2d,
                    filters: None,
                    kernel_size: None,
                    pool_kind: Some(PoolKind::Max),
                    units: None,
                    activation: None,
                },
                LayerSpec {
                    kind: LayerKind::Conv2d,
                    filters: Some(8),
                    kernel_size: Some(3),
                    pool_kind: None,
                    units: None,
                    activation: Some(ActivationId::Relu),
                },
                LayerSpec {
                    kind: LayerKind::Gap2d,
                    filters: None,
                    kernel_size: None,
                    pool_kind: None,
                    units: None,
                    activation: None,
                },
                LayerSpec {
                    kind: LayerKind::Dense,
                    filters: None,
                    kernel_size: None,
                    pool_kind: None,
                    units: Some(1),
                    activation: Some(ActivationId::Linear),
                },
            ],
            learning_rate: 0.1,
            optimizer: "SGD".into(),
            activation: ActivationId::Relu,
            batch_size: 16,
            noise: 0.1,
            perc_train_data: 50.0,
            regularization: "none".into(),
            regularization_rate: 0.0,
            image_size: IMAGE_SIZE,
        }
    }
}
