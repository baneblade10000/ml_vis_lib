# Hand-rolled educational CNN

Rust CPU train/forward/backward + optional **wgpu WebGPU** device init.

- Browser: `WasmCnnEngine.create(configJson)` tries WebGPU, falls back to CPU.
- `engine.backend()` → `"webgpu"` | `"gpu"` | `"cpu"`.
- Playground net is tiny (16×16); math runs on CPU (fast). GPU context is live for
  future GEMM/conv shaders.

Build: `pnpm wasm:build:cnn`
