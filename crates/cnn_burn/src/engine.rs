//! Playground CNN engine — hand-rolled Rust (CPU). WebGPU used when available for GEMM assist.

use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

use crate::config::{CnnConfig, CnnMode, SIGNAL_LENGTH};
use crate::gpu::GpuContext;
use crate::layers::{
    add_grads, backward_from_logit, bce_logit_grad, build_pipeline, dense_matrix,
    display_kernels_1d, display_kernels_1d_in, display_kernels_2d, display_kernels_2d_in,
    forward_1d, forward_2d, param_count, read_params,
    sgd_step, weight_mag, write_grads, write_params, zero_grads, Layer, LayerMeta,
};
use crate::tensor::{zeros2d, Signal, Volume};

#[derive(Clone, Debug, Serialize)]
pub struct LossPoint {
    pub epoch: usize,
    pub train: f32,
    pub test: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainStatsDto {
    pub epoch: usize,
    #[serde(rename = "lossTrain")]
    pub loss_train: f32,
    #[serde(rename = "lossTest")]
    pub loss_test: f32,
    #[serde(rename = "accTrain")]
    pub acc_train: f32,
    #[serde(rename = "accTest")]
    pub acc_test: f32,
}

pub struct PlaygroundEngine {
    pub config: CnnConfig,
    layers: Vec<Layer>,
    metas: Vec<LayerMeta>,
    /// Flat NCHW: n * 1 * H * W
    images2d: Vec<f32>,
    /// Flat: n * 1 * L
    images1d: Vec<f32>,
    labels: Vec<i32>,
    n: usize,
    n_train: usize,
    pub epoch: usize,
    pub loss_train: f32,
    pub loss_test: f32,
    pub acc_train: f32,
    pub acc_test: f32,
    pub loss_history: Vec<LossPoint>,
    pub inspected: usize,
    rng: StdRng,
    /// "cpu" | "webgpu"
    pub backend_name: &'static str,
    #[allow(dead_code)]
    gpu: Option<GpuContext>,
}

impl PlaygroundEngine {
    pub fn new(config: CnnConfig) -> Self {
        Self::new_with_gpu(config, None)
    }

    pub fn new_with_gpu(config: CnnConfig, gpu: Option<GpuContext>) -> Self {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let (layers, metas) = build_pipeline(&config, &mut rng);
        let backend_name = match &gpu {
            Some(g) if g.adapter_name.to_lowercase().contains("webgpu") => "webgpu",
            Some(_) => "gpu",
            None => "cpu",
        };
        Self {
            config,
            layers,
            metas,
            images2d: vec![],
            images1d: vec![],
            labels: vec![0],
            n: 0,
            n_train: 0,
            epoch: 0,
            loss_train: 0.0,
            loss_test: 0.0,
            acc_train: 0.0,
            acc_test: 0.0,
            loss_history: vec![],
            inspected: 0,
            rng,
            backend_name,
            gpu,
        }
    }

    pub fn rebuild_model(&mut self) {
        let (layers, metas) = build_pipeline(&self.config, &mut self.rng);
        self.layers = layers;
        self.metas = metas;
        self.epoch = 0;
        self.loss_history.clear();
        self.loss_train = 0.0;
        self.loss_test = 0.0;
        self.acc_train = 0.0;
        self.acc_test = 0.0;
    }

    pub fn set_data_2d(&mut self, flat: &[f32], labels: &[i32], n: usize, train_n: usize) {
        let size = self.config.image_size();
        assert_eq!(flat.len(), n * size * size);
        assert_eq!(labels.len(), n);
        self.images2d = flat.to_vec();
        self.images1d.clear();
        self.labels = labels.to_vec();
        self.n = n;
        self.n_train = train_n.clamp(1, n.saturating_sub(1).max(1));
        self.inspected = 0;
        self.refresh_metrics();
        self.push_loss_history();
    }

    pub fn set_data_1d(&mut self, flat: &[f32], labels: &[i32], n: usize, train_n: usize) {
        assert_eq!(flat.len(), n * SIGNAL_LENGTH);
        assert_eq!(labels.len(), n);
        self.images1d = flat.to_vec();
        self.images2d.clear();
        self.labels = labels.to_vec();
        self.n = n;
        self.n_train = train_n.clamp(1, n.saturating_sub(1).max(1));
        self.inspected = 0;
        self.refresh_metrics();
        self.push_loss_history();
    }

    fn example_2d(&self, idx: usize) -> Volume {
        example_2d_from(&self.images2d, idx, self.config.image_size())
    }

    fn example_1d(&self, idx: usize) -> Signal {
        example_1d_from(&self.images1d, idx)
    }

    fn forward_idx(&mut self, idx: usize) -> f32 {
        match self.config.mode {
            CnnMode::D2 => {
                let ex = self.example_2d(idx);
                forward_2d(&mut self.layers, ex)
            }
            CnnMode::D1 => {
                let ex = self.example_1d(idx);
                forward_1d(&mut self.layers, ex)
            }
        }
    }

    /// One SGD epoch. `record_history` is false on the play hot path — paint pushes a point later.
    pub fn train_epoch(&mut self, record_history: bool) {
        let n = self.n_train;
        if n == 0 {
            return;
        }
        let bs = self.config.batch_size.max(1);
        let lr = self.config.learning_rate as f32;
        let mut loss_sum = 0.0f32;
        let mut seen = 0usize;
        let mut start = 0usize;
        // Reuse one index buffer — avoid alloc-per-batch on the play loop.
        let mut indices = Vec::with_capacity(bs);
        while start < n {
            let end = (start + bs).min(n);
            indices.clear();
            indices.extend((start..end).map(|i| i as u32));
            let batch_loss = self.accumulate_batch(&indices);
            let bn = (end - start) as f32;
            sgd_step(&mut self.layers, lr, bn);
            loss_sum += batch_loss;
            seen += end - start;
            start = end;
        }
        self.epoch += 1;
        self.loss_train = if seen > 0 { loss_sum / seen as f32 } else { 0.0 };
        if record_history {
            self.push_loss_history();
        }
    }

    /// Zero grads, run forward+backward for each index, accumulate parameter grads.
    /// Returns sum of per-example losses.
    pub fn accumulate_batch(&mut self, indices: &[u32]) -> f32 {
        zero_grads(&mut self.layers);
        if indices.is_empty() {
            return 0.0;
        }

        // Rayon only pays off on fat batches — cloning the net per example is costly.
        #[cfg(not(target_arch = "wasm32"))]
        if indices.len() >= 64 {
            use rayon::prelude::*;
            let mode = self.config.mode.clone();
            let labels = &self.labels;
            let images2d = &self.images2d;
            let images1d = &self.images1d;
            let n = self.n;
            let image_size = self.config.image_size();
            let template = &self.layers;

            let results: Vec<(f32, Vec<f32>)> = indices
                .par_iter()
                .filter_map(|&i| {
                    let i = i as usize;
                    if i >= n {
                        return None;
                    }
                    let mut local = template.clone();
                    zero_grads(&mut local);
                    let logit = match mode {
                        CnnMode::D2 => {
                            let ex = example_2d_from(images2d, i, image_size);
                            forward_2d(&mut local, ex)
                        }
                        CnnMode::D1 => {
                            let ex = example_1d_from(images1d, i);
                            forward_1d(&mut local, ex)
                        }
                    };
                    let (loss, dlogit) = bce_logit_grad(logit, labels[i] as f32);
                    backward_from_logit(&mut local, dlogit);
                    let mut grads = vec![0.0; param_count(&local)];
                    write_grads(&local, &mut grads);
                    Some((loss, grads))
                })
                .collect();

            let mut batch_loss = 0.0f32;
            for (loss, grads) in results {
                batch_loss += loss;
                add_grads(&mut self.layers, &grads);
            }
            return batch_loss;
        }

        {
            let mut batch_loss = 0.0f32;
            for &i in indices {
                let i = i as usize;
                if i >= self.n {
                    continue;
                }
                let logit = self.forward_idx(i);
                let y = self.labels[i] as f32;
                let (loss, dlogit) = bce_logit_grad(logit, y);
                batch_loss += loss;
                backward_from_logit(&mut self.layers, dlogit);
            }
            batch_loss
        }
    }

    pub fn param_count(&self) -> usize {
        param_count(&self.layers)
    }

    pub fn export_params(&self) -> Vec<f32> {
        let mut out = vec![0.0; param_count(&self.layers)];
        write_params(&self.layers, &mut out);
        out
    }

    pub fn load_params(&mut self, params: &[f32]) {
        read_params(&mut self.layers, params);
    }

    pub fn export_grads(&self) -> Vec<f32> {
        let mut out = vec![0.0; param_count(&self.layers)];
        write_grads(&self.layers, &mut out);
        out
    }

    pub fn add_grad_sums(&mut self, grads: &[f32]) {
        add_grads(&mut self.layers, grads);
    }

    pub fn zero_all_grads(&mut self) {
        zero_grads(&mut self.layers);
    }

    pub fn apply_sgd(&mut self, lr: f32, batch_n: f32) {
        sgd_step(&mut self.layers, lr, batch_n);
    }

    pub fn bump_epoch(&mut self, loss_train: f32) {
        self.epoch += 1;
        self.loss_train = loss_train;
        self.push_loss_history();
    }

    fn eval_loss_range(&mut self, range: std::ops::Range<usize>) -> f32 {
        if range.start >= range.end {
            return 0.0;
        }
        let mut s = 0.0;
        let mut n = 0usize;
        for i in range {
            let logit = self.forward_idx(i);
            let (loss, _) = bce_logit_grad(logit, self.labels[i] as f32);
            s += loss;
            n += 1;
        }
        s / n.max(1) as f32
    }

    fn eval_acc_range(&mut self, range: std::ops::Range<usize>) -> f32 {
        if range.start >= range.end {
            return 0.0;
        }
        let mut correct = 0usize;
        let mut n = 0usize;
        for i in range {
            let logit = self.forward_idx(i);
            let pred = if logit > 0.0 { 1 } else { 0 };
            if pred == self.labels[i] {
                correct += 1;
            }
            n += 1;
        }
        correct as f32 / n.max(1) as f32
    }

    pub fn refresh_metrics(&mut self) {
        self.loss_train = self.eval_loss_range(0..self.n_train);
        self.loss_test = self.eval_loss_range(self.n_train..self.n);
        self.acc_train = self.eval_acc_range(0..self.n_train);
        self.acc_test = self.eval_acc_range(self.n_train..self.n);
    }

    pub fn refresh_accuracy_sampled(&mut self) {
        let tr_end = self.n_train.min(48);
        let te_end = (self.n_train + 48).min(self.n);
        self.loss_test = self.eval_loss_range(self.n_train..self.n);
        self.acc_train = self.eval_acc_range(0..tr_end);
        self.acc_test = self.eval_acc_range(self.n_train..te_end);
        self.push_loss_history();
    }

    pub fn push_loss_history(&mut self) {
        if let Some(last) = self.loss_history.last_mut() {
            if last.epoch == self.epoch {
                last.train = self.loss_train;
                last.test = self.loss_test;
                return;
            }
        }
        self.loss_history.push(LossPoint {
            epoch: self.epoch,
            train: self.loss_train,
            test: self.loss_test,
        });
    }

    pub fn stats(&self) -> TrainStatsDto {
        TrainStatsDto {
            epoch: self.epoch,
            loss_train: self.loss_train,
            loss_test: self.loss_test,
            acc_train: self.acc_train,
            acc_test: self.acc_test,
        }
    }

    pub fn set_inspected(&mut self, index: usize) {
        let n = self.n.saturating_sub(self.n_train).max(1);
        self.inspected = index.min(n - 1);
    }

    pub fn gallery_base(&self) -> usize {
        self.n_train
    }

    pub fn n_total(&self) -> usize {
        self.n
    }

    pub fn metas(&self) -> &[LayerMeta] {
        &self.metas
    }

    pub fn probability_at(&mut self, absolute_index: usize) -> f32 {
        let z = self.forward_idx(absolute_index);
        1.0 / (1.0 + (-z).exp())
    }

    pub fn weight_mags(&self) -> Vec<Option<f64>> {
        self.layers.iter().map(weight_mag).collect()
    }

    pub fn inspect_feature_maps_json(&mut self, include_kernels: bool) -> serde_json::Value {
        let abs = (self.n_train + self.inspected).min(self.n.saturating_sub(1));
        let _ = self.forward_idx(abs);

        let mut arr = Vec::new();
        for (i, meta) in self.metas.iter().enumerate() {
            let mut obj = serde_json::json!({ "layerId": meta.id });
            if let Some(layer) = self.layers.get(i) {
                match layer {
                    Layer::Input2d { .. } => {
                        if matches!(self.config.mode, CnnMode::D2) {
                            let v = self.example_2d(abs);
                            obj["maps2d"] = serde_json::to_value(&v).unwrap();
                        }
                    }
                    Layer::Input1d { .. } => {
                        if matches!(self.config.mode, CnnMode::D1) {
                            let s = self.example_1d(abs);
                            obj["signals"] = serde_json::to_value(&s).unwrap();
                        }
                    }
                    Layer::Conv2d { last_out, .. } | Layer::Pool2d { last_out, .. } => {
                        obj["maps2d"] = serde_json::to_value(last_out).unwrap();
                    }
                    Layer::Gap2d { last_out, .. } | Layer::Gap1d { last_out, .. } => {
                        // Flatten to [[v0..vC]] — same shape FlattenVector/Dense expect.
                        let flat: Vec<f32> = last_out
                            .iter()
                            .map(|row| row.first().copied().unwrap_or(0.0))
                            .collect();
                        obj["signals"] = serde_json::to_value(vec![flat]).unwrap();
                    }
                    Layer::Conv1d { last_out, .. } | Layer::Pool1d { last_out, .. } => {
                        obj["signals"] = serde_json::to_value(last_out).unwrap();
                    }
                    Layer::Dense { last_out, .. } => {
                        obj["signals"] = serde_json::to_value(vec![last_out.clone()]).unwrap();
                    }
                    Layer::Flatten { last_flat, .. } => {
                        obj["signals"] = serde_json::to_value(vec![last_flat.clone()]).unwrap();
                    }
                    Layer::Output { last_logit, .. } => {
                        obj["signals"] = serde_json::to_value(vec![vec![*last_logit]]).unwrap();
                    }
                }
                if include_kernels {
                    if let Some((k, b)) = display_kernels_2d(layer) {
                        obj["kernels2d"] = serde_json::to_value(k).unwrap();
                        obj["biases"] = serde_json::to_value(b).unwrap();
                    }
                    if let Some(kin) = display_kernels_2d_in(layer) {
                        obj["kernels2dIn"] = serde_json::to_value(kin).unwrap();
                    }
                    if let Some((k, b)) = display_kernels_1d(layer) {
                        obj["kernels1d"] = serde_json::to_value(k).unwrap();
                        obj["biases"] = serde_json::to_value(b).unwrap();
                    }
                    if let Some(kin) = display_kernels_1d_in(layer) {
                        obj["kernels1dIn"] = serde_json::to_value(kin).unwrap();
                    }
                } else {
                    // Biases are tiny — always send so UI chips don't flash on/off.
                    if let Some((_, b)) = display_kernels_2d(layer) {
                        obj["biases"] = serde_json::to_value(b).unwrap();
                    }
                    if let Some((_, b)) = display_kernels_1d(layer) {
                        obj["biases"] = serde_json::to_value(b).unwrap();
                    }
                }
                // Dense/output matrix is tiny — always send so weight edges don't drop.
                if let Some(m) = dense_matrix(layer) {
                    obj["matrix"] = serde_json::to_value(m).unwrap();
                }
            }
            arr.push(obj);
        }
        serde_json::Value::Array(arr)
    }
}

fn example_2d_from(images2d: &[f32], idx: usize, size: usize) -> Volume {
    let base = idx * size * size;
    let mut map = zeros2d(size, size);
    for r in 0..size {
        for c in 0..size {
            map[r][c] = images2d[base + r * size + c];
        }
    }
    vec![map]
}

fn example_1d_from(images1d: &[f32], idx: usize) -> Signal {
    let base = idx * SIGNAL_LENGTH;
    let row: Vec<f32> = images1d[base..base + SIGNAL_LENGTH].to_vec();
    vec![row]
}
