/** Factory for the CNN WASM train worker (Vite-bundled). */
export function createCnnTrainWorker(): Worker {
  return new Worker(new URL("./cnnTrainWorker.ts", import.meta.url), { type: "module" });
}
