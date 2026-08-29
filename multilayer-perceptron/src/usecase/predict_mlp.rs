use crate::core::{Result, Tensor};
use crate::domain::models::Sequential;

pub fn predict(model: &mut Sequential, input: Tensor) -> Result<Tensor> {
  model.forward(input)
}
