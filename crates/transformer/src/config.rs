//! Transformer playground configuration.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TransformerConfig {
    /// Number of data tokens in the alphabet (specials `<s>`/`</s>` excluded).
    pub alphabet_size: usize,
    pub d_model: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub ff_dim: usize,
    pub max_len: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            alphabet_size: 20,
            d_model: 32,
            heads: 4,
            enc_layers: 2,
            dec_layers: 2,
            ff_dim: 64,
            max_len: 6,
            learning_rate: 0.003,
            seed: 42,
        }
    }
}

impl TransformerConfig {
    pub fn vocab_size(&self) -> usize {
        self.alphabet_size + 2
    }

    pub fn bos(&self) -> usize {
        self.alphabet_size
    }

    pub fn eos(&self) -> usize {
        self.alphabet_size + 1
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.heads
    }
}
