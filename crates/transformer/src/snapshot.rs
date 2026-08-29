//! Snapshot DTOs serialized for the JS side (worker → React).

use serde::Serialize;

use crate::model::AttentionRecord;
use crate::tensor::Mat;

/// `[head][row][col]` attention probabilities.
pub type HeadMatrices = Vec<Vec<Vec<f32>>>;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttentionDto {
    pub enc_self: Vec<HeadMatrices>,
    pub dec_self: Vec<HeadMatrices>,
    pub cross: Vec<HeadMatrices>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LossPoint {
    pub step: u64,
    pub loss: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotDto {
    pub step: u64,
    /// Exponential moving average of the training loss.
    pub loss: f32,
    /// Token-level accuracy of the greedy decode on the display sample.
    pub accuracy: f32,
    pub task: String,
    pub input_tokens: Vec<usize>,
    pub target_tokens: Vec<usize>,
    pub predicted_tokens: Vec<usize>,
    /// Decoder input tokens (<s> + target) aligned with dec_self/cross rows.
    pub dec_in_tokens: Vec<usize>,
    /// Encoder input tokens (input + </s>) aligned with enc_self/cross cols.
    pub enc_in_tokens: Vec<usize>,
    pub attention: AttentionDto,
    pub loss_history: Vec<LossPoint>,
    pub alphabet_size: usize,
    /// Display label per token id, including `<s>`/`</s>` at the end.
    pub labels: Vec<String>,
}

fn heads_to_dto(heads: &[Mat]) -> HeadMatrices {
    heads
        .iter()
        .map(|m| (0..m.rows).map(|i| m.row(i).to_vec()).collect())
        .collect()
}

impl From<&AttentionRecord> for AttentionDto {
    fn from(record: &AttentionRecord) -> Self {
        AttentionDto {
            enc_self: record.enc_self.iter().map(|l| heads_to_dto(l)).collect(),
            dec_self: record.dec_self.iter().map(|l| heads_to_dto(l)).collect(),
            cross: record.cross.iter().map(|l| heads_to_dto(l)).collect(),
        }
    }
}
