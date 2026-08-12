/// <reference lib="webworker" />
/**
 * Legacy JS CNN worker — disabled.
 * Playground uses Burn WASM via `packages/playground/src/burn/cnnTrainWorker.ts`.
 */
import type { FromTrainWorker, ToTrainWorker } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (ev: MessageEvent<ToTrainWorker>) => {
  const msg = ev.data;
  const out: FromTrainWorker = {
    type: "error",
    message:
      "JS CNN train worker removed. Use createBurnCnnTrainWorker() from the playground (Burn WASM).",
  };
  if (msg.type !== "dispose") self.postMessage(out);
};
