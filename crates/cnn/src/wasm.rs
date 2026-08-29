//! WASM bindings — async WebGPU init with CPU fallback.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::config::CnnConfig;
use crate::engine::PlaygroundEngine;
use crate::gpu::GpuContext;
use crate::layers::LayerMeta;

fn to_js_value<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LayerViewJs<'a> {
    id: &'a str,
    kind: &'a str,
    label: &'a str,
    params: usize,
    weight_mag: Option<f64>,
    shape: ShapeJs,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
enum ShapeJs {
    #[serde(rename = "2d")]
    TwoD {
        channels: usize,
        rows: usize,
        cols: usize,
    },
    #[serde(rename = "1d")]
    OneD { channels: usize, length: usize },
}

fn layer_views<'a>(metas: &'a [LayerMeta], mags: &[Option<f64>]) -> Vec<LayerViewJs<'a>> {
    metas
        .iter()
        .zip(mags.iter().copied())
        .map(|(m, weight_mag)| LayerViewJs {
            id: &m.id,
            kind: &m.kind,
            label: &m.label,
            params: m.params,
            weight_mag,
            shape: if m.space == "2d" {
                ShapeJs::TwoD {
                    channels: m.channels,
                    rows: m.rows,
                    cols: m.cols,
                }
            } else {
                ShapeJs::OneD {
                    channels: m.channels.max(1),
                    length: m.length,
                }
            },
        })
        .collect()
}

#[wasm_bindgen]
pub struct WasmCnnEngine {
    inner: PlaygroundEngine,
}

#[wasm_bindgen]
impl WasmCnnEngine {
    /// Prefer WebGPU; fall back to CPU if adapter/device unavailable.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(config_json: String) -> Result<WasmCnnEngine, JsValue> {
        console_error_panic_hook::set_once();
        let config: CnnConfig = serde_json::from_str(&config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse: {e}")))?;
        let gpu = GpuContext::try_init().await;
        Ok(Self {
            inner: PlaygroundEngine::new_with_gpu(config, gpu),
        })
    }

    /// Sync CPU-only constructor (no WebGPU request).
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmCnnEngine, JsValue> {
        console_error_panic_hook::set_once();
        let config: CnnConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse: {e}")))?;
        Ok(Self {
            inner: PlaygroundEngine::new(config),
        })
    }

    pub fn backend(&self) -> String {
        self.inner.backend_name.into()
    }

    #[wasm_bindgen(js_name = setData2d)]
    pub fn set_data_2d(
        &mut self,
        images: &[f32],
        labels: &[i32],
        n: usize,
        train_n: usize,
    ) -> Result<(), JsValue> {
        self.inner.set_data_2d(images, labels, n, train_n);
        Ok(())
    }

    #[wasm_bindgen(js_name = setData1d)]
    pub fn set_data_1d(
        &mut self,
        signals: &[f32],
        labels: &[i32],
        n: usize,
        train_n: usize,
    ) -> Result<(), JsValue> {
        self.inner.set_data_1d(signals, labels, n, train_n);
        Ok(())
    }

    /// Play hot path: train only — no history push, no JSValue marshalling.
    #[wasm_bindgen(js_name = trainEpoch)]
    pub fn train_epoch(&mut self) {
        self.inner.train_epoch(false);
    }

    /// Step / pause: train + append loss curve point.
    #[wasm_bindgen(js_name = trainEpochRecord)]
    pub fn train_epoch_record(&mut self) -> Result<JsValue, JsValue> {
        self.inner.train_epoch(true);
        to_js_value(&self.inner.stats())
    }

    #[wasm_bindgen(js_name = refreshMetrics)]
    pub fn refresh_metrics(&mut self) {
        self.inner.refresh_metrics();
    }

    #[wasm_bindgen(js_name = refreshAccuracy)]
    pub fn refresh_accuracy(&mut self) {
        self.inner.refresh_accuracy_sampled();
    }

    #[wasm_bindgen(js_name = pushLossHistory)]
    pub fn push_loss_history(&mut self) {
        self.inner.push_loss_history();
    }

    pub fn stats(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.inner.stats())
    }

    #[wasm_bindgen(js_name = lossHistory)]
    pub fn loss_history(&self) -> Result<JsValue, JsValue> {
        to_js_value(&self.inner.loss_history)
    }

    pub fn metas(&self) -> Result<JsValue, JsValue> {
        let mags = self.inner.weight_mags();
        let rows = layer_views(self.inner.metas(), &mags);
        to_js_value(&rows)
    }

    #[wasm_bindgen(js_name = featureMaps)]
    pub fn feature_maps(&mut self, include_kernels: Option<bool>) -> Result<JsValue, JsValue> {
        let v = self
            .inner
            .inspect_feature_maps_json(include_kernels.unwrap_or(true));
        let s = serde_json::to_string(&v).map_err(|e| JsValue::from_str(&e.to_string()))?;
        js_sys::JSON::parse(&s).map_err(|e| e)
    }

    #[wasm_bindgen(js_name = setInspected)]
    pub fn set_inspected(&mut self, index: usize) {
        self.inner.set_inspected(index);
    }

    pub fn inspected(&self) -> usize {
        self.inner.inspected
    }

    pub fn probability(&mut self) -> f32 {
        let abs = self.inner.gallery_base() + self.inner.inspected;
        self.inner.probability_at(abs)
    }

    #[wasm_bindgen(js_name = predictGallery)]
    pub fn predict_gallery(&mut self, max: usize) -> Result<JsValue, JsValue> {
        let base = self.inner.gallery_base();
        let n = self.inner.n_total().saturating_sub(base);
        let take = n.min(max);
        let mut out = Vec::with_capacity(take);
        for i in 0..take {
            out.push(self.inner.probability_at(base + i));
        }
        to_js_value(&out)
    }

    #[wasm_bindgen(js_name = configJson)]
    pub fn config_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.config).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = applyConfigJson)]
    pub fn apply_config_json(&mut self, config_json: &str) -> Result<(), JsValue> {
        let config: CnnConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse: {e}")))?;
        self.inner.config = config;
        self.inner.rebuild_model();
        Ok(())
    }

    #[wasm_bindgen(js_name = rebuildModel)]
    pub fn rebuild_model(&mut self) {
        self.inner.rebuild_model();
    }

    pub fn epoch(&self) -> usize {
        self.inner.epoch
    }

    #[wasm_bindgen(js_name = paramCount)]
    pub fn param_count(&self) -> usize {
        self.inner.param_count()
    }

    #[wasm_bindgen(js_name = exportParams)]
    pub fn export_params(&self) -> Vec<f32> {
        self.inner.export_params()
    }

    #[wasm_bindgen(js_name = loadParams)]
    pub fn load_params(&mut self, params: &[f32]) {
        self.inner.load_params(params);
    }

    #[wasm_bindgen(js_name = exportGrads)]
    pub fn export_grads(&self) -> Vec<f32> {
        self.inner.export_grads()
    }

    #[wasm_bindgen(js_name = addGradSums)]
    pub fn add_grad_sums(&mut self, grads: &[f32]) {
        self.inner.add_grad_sums(grads);
    }

    #[wasm_bindgen(js_name = zeroAllGrads)]
    pub fn zero_all_grads(&mut self) {
        self.inner.zero_all_grads();
    }

    #[wasm_bindgen(js_name = accumulateBatch)]
    pub fn accumulate_batch(&mut self, indices: &[u32]) -> f32 {
        self.inner.accumulate_batch(indices)
    }

    #[wasm_bindgen(js_name = applySgd)]
    pub fn apply_sgd(&mut self, lr: f32, batch_n: f32) {
        self.inner.apply_sgd(lr, batch_n);
    }

    #[wasm_bindgen(js_name = bumpEpoch)]
    pub fn bump_epoch(&mut self, loss_train: f32) {
        self.inner.bump_epoch(loss_train);
    }

    #[wasm_bindgen(js_name = nTrain)]
    pub fn n_train(&self) -> usize {
        self.inner.gallery_base()
    }
}
