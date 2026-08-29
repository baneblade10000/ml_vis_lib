/// <reference lib="webworker" />
/**
 * Legacy JS CNN worker — disabled.
 * Playground uses CNN WASM via `packages/playground/src/wasm/cnn/cnnTrainWorker.ts`.
 */
import type { FromTrainWorker, ToTrainWorker } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (ev: MessageEvent<ToTrainWorker>) => {
  const msg = ev.data;
  const out: FromTrainWorker = {
    type: "error",
    message:
      "JS CNN train worker removed. Use createCnnTrainWorker() from the playground app (CNN WASM).",
  };
  if (msg.type !== "dispose") self.postMessage(out);
};
