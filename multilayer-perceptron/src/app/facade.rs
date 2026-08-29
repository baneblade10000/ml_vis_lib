use crate::domain::ports::DataRepository;
use crate::usecase::{TrainMLPUsecase, TrainRequest, TrainResponse};
use std::sync::Arc;

pub struct Application<R: DataRepository> {
  train_usecase: TrainMLPUsecase<R>,
}

impl<R: DataRepository> Application<R> {
  pub fn new(data_repo: Arc<R>) -> Self {
    Self {
      train_usecase: TrainMLPUsecase::new(data_repo),
    }
  }

  pub fn train(&self, request: TrainRequest) -> crate::core::Result<TrainResponse> {
    self.train_usecase.execute(request)
  }
}
