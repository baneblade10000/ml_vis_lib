//! WASM bindings for the transformer playground.

use wasm_bindgen::prelude::*;

use crate::config::TransformerConfig;
use crate::engine::PlaygroundEngine;
use crate::tasks::TransformerTask;

fn to_js_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value
        .serialize(&serializer)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub struct WasmTransformerEngine {
    inner: PlaygroundEngine,
}

#[wasm_bindgen]
impl WasmTransformerEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, task: &str) -> Result<WasmTransformerEngine, JsValue> {
        console_error_panic_hook::set_once();
        let config: TransformerConfig = if config_json.trim().is_empty() {
            TransformerConfig::default()
        } else {
            serde_json::from_str(config_json)
                .map_err(|e| JsValue::from_str(&format!("config parse: {e}")))?
        };
        let task = TransformerTask::from_id(task).ok_or_else(|| JsValue::from_str(&format!("unknown task: {task}")))?;
        Ok(Self { inner: PlaygroundEngine::new(config, task) })
    }

    /// Runs `n` Adam steps; returns the EMA loss. No snapshot allocation —
    /// this is the Play hot path.
    pub fn train_steps(&mut self, n: usize) -> f64 {
        self.inner.train_steps(n) as f64
    }

    pub fn get_step(&self) -> u64 {
        self.inner.get_step()
    }

    pub fn get_loss(&self) -> f64 {
        self.inner.get_loss() as f64
    }

    /// Full snapshot for the UI (greedy decode + attention + loss history).
    pub fn snapshot(&mut self) -> Result<JsValue, JsValue> {
        to_js_value(self.inner.snapshot())
    }

    /// Runs `n` steps, then returns a fresh snapshot.
    pub fn train_steps_snapshot(&mut self, n: usize) -> Result<JsValue, JsValue> {
        self.inner.train_steps(n);
        self.inner.refresh_snapshot();
        to_js_value(self.inner.snapshot())
    }

    pub fn reset(&mut self, task: &str) -> Result<(), JsValue> {
        let task = TransformerTask::from_id(task).ok_or_else(|| JsValue::from_str(&format!("unknown task: {task}")))?;
        self.inner.reset(task);
        Ok(())
    }

    pub fn reroll_sample(&mut self) -> Result<JsValue, JsValue> {
        self.inner.reroll_sample();
        to_js_value(self.inner.snapshot())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr as f32);
    }
}
