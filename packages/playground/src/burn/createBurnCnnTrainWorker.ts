/** Factory for the Burn WASM CNN train worker (Vite-bundled). */
export function createBurnCnnTrainWorker(): Worker {
  return new Worker(new URL("./cnnTrainWorker.ts", import.meta.url), { type: "module" });
}
