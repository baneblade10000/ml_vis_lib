/** Factory for the transformer WASM train worker (Vite-bundled). */
export function createTransformerTrainWorker(): Worker {
  return new Worker(new URL("./transformerWorker.ts", import.meta.url), { type: "module" });
}
