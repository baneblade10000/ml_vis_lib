# Hand-rolled educational transformer

Small encoder-decoder transformer for the seq2seq playground (reverse / sort).
Rust CPU with manual forward/backward (no autograd framework), Adam, and
attention-probability capture for visualisation.

- `model.rs` — params, forward caches, manual backward (softmax, layernorm,
  multi-head self-/cross-attention, FFN), Adam, greedy decode.
- `engine.rs` — sampling, train loop, display snapshot (greedy decode +
  attention matrices + loss history).
- `wasm.rs` — `WasmTransformerEngine` bindings for the browser worker.
- `tests/gradcheck.rs` — numeric gradient check + task convergence test.

Build WASM: `pnpm wasm:build:transformer`
