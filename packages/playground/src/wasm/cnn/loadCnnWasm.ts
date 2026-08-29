/** Lazy-load CNN WASM with download progress (demo dock / tooling). */

import type { WasmCnnEngine as WasmCnnEngineType } from "./pkg/cnn";

export type WasmLoadPhase = "idle" | "fetch" | "compile" | "ready" | "error";

export interface WasmLoadProgress {
  phase: WasmLoadPhase;
  loaded: number;
  total: number;
  ratio: number | null;
  error?: string;
}

export interface CnnWasmModule {
  WasmCnnEngine: new (config_json: string) => WasmCnnEngineType;
  default: (moduleOrBytes?: Response | BufferSource | WebAssembly.Module) => Promise<unknown>;
}

let cached: Promise<CnnWasmModule> | null = null;

function concatChunks(chunks: Uint8Array[], totalLen: number): Uint8Array {
  const out = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function wasmAssetUrl(): string {
  return new URL("./pkg/cnn_bg.wasm", import.meta.url).href;
}

export function loadCnnWasm(
  onProgress?: (p: WasmLoadProgress) => void,
): Promise<CnnWasmModule> {
  if (cached) return cached;

  cached = (async () => {
    onProgress?.({ phase: "fetch", loaded: 0, total: 0, ratio: null });
    const res = await fetch(wasmAssetUrl());
    if (!res.ok) {
      const err = `Failed to fetch CNN WASM (${res.status})`;
      onProgress?.({ phase: "error", loaded: 0, total: 0, ratio: null, error: err });
      throw new Error(err);
    }
    const total = Number(res.headers.get("Content-Length") ?? 0);
    const reader = res.body?.getReader();
    let bytes: Uint8Array;
    if (!reader) {
      bytes = new Uint8Array(await res.arrayBuffer());
    } else {
      const chunks: Uint8Array[] = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress?.({
          phase: "fetch",
          loaded,
          total,
          ratio: total > 0 ? Math.min(1, loaded / total) : null,
        });
      }
      bytes = concatChunks(chunks, loaded);
    }
    onProgress?.({
      phase: "compile",
      loaded: bytes.byteLength,
      total: bytes.byteLength,
      ratio: 1,
    });
    const mod = (await import("./pkg/cnn.js")) as unknown as CnnWasmModule;
    await mod.default(bytes);
    onProgress?.({
      phase: "ready",
      loaded: bytes.byteLength,
      total: bytes.byteLength,
      ratio: 1,
    });
    return mod;
  })().catch((e) => {
    cached = null;
    throw e;
  });

  return cached;
}
