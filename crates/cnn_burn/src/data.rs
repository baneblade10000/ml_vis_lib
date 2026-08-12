//! Dataset helpers (no ML framework).

use crate::config::IMAGE_SIZE;

/// Accuracy helper kept for smoke tests.
pub fn scalar_first(v: &[f32]) -> f32 {
    v.first().copied().unwrap_or(0.0)
}

pub fn blank_image() -> Vec<f32> {
    vec![0.0; IMAGE_SIZE * IMAGE_SIZE]
}
