/// <reference lib="webworker" />
/**
 * Grad shard — own Rust WasmCnnEngine (CPU). Parent fans out batch indices.
 */
import init, { WasmCnnEngine } from "./pkg/cnn.js";
import wasmUrl from "./pkg/cnn_bg.wasm?url";

declare const self: DedicatedWorkerGlobalScope;

type InMsg =
  | {
      type: "init";
      configJson: string;
      mode: "2d" | "1d";
      flat: Float32Array;
      labels: Int32Array;
      n: number;
      trainN: number;
    }
  | { type: "setParams"; params: Float32Array }
  | { type: "setData"; mode: "2d" | "1d"; flat: Float32Array; labels: Int32Array; n: number; trainN: number }
  | { type: "compute"; indices: Uint32Array }
  | { type: "dispose" };

type OutMsg =
  | { type: "ready" }
  | { type: "grads"; loss: number; grads: Float32Array }
  | { type: "error"; message: string };

let engine: WasmCnnEngine | null = null;
let wasmReady: Promise<void> | null = null;

async function ensureWasm(): Promise<void> {
  if (!wasmReady) wasmReady = init({ module_or_path: wasmUrl }).then(() => undefined);
  await wasmReady;
}

function post(msg: OutMsg, transfer?: Transferable[]): void {
  if (transfer?.length) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

self.onmessage = (ev: MessageEvent<InMsg>) => {
  void (async () => {
    const msg = ev.data;
    try {
      await ensureWasm();
      switch (msg.type) {
        case "init": {
          engine?.free();
          // Sync CPU ctor — shards don't need their own WebGPU device.
          engine = new WasmCnnEngine(msg.configJson);
          if (msg.mode === "2d") engine.setData2d(msg.flat, msg.labels, msg.n, msg.trainN);
          else engine.setData1d(msg.flat, msg.labels, msg.n, msg.trainN);
          post({ type: "ready" });
          break;
        }
        case "setData": {
          if (!engine) return;
          if (msg.mode === "2d") engine.setData2d(msg.flat, msg.labels, msg.n, msg.trainN);
          else engine.setData1d(msg.flat, msg.labels, msg.n, msg.trainN);
          break;
        }
        case "setParams": {
          engine?.loadParams(msg.params);
          break;
        }
        case "compute": {
          if (!engine) {
            post({ type: "error", message: "shard not ready" });
            return;
          }
          const loss = engine.accumulateBatch(msg.indices);
          const grads = engine.exportGrads();
          post({ type: "grads", loss, grads }, [grads.buffer]);
          break;
        }
        case "dispose": {
          engine?.free();
          engine = null;
          break;
        }
      }
    } catch (err) {
      post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    }
  })();
};
