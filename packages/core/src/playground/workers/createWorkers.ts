/**
 * Factories that spawn playground train workers.
 * `import.meta.url` resolves relative to this file in `dist/workers/`.
 */

export function createCnnTrainWorker(): Worker {
  return new Worker(new URL("./cnn-train.js", import.meta.url), { type: "module" });
}

export function createNetworkTrainWorker(): Worker {
  return new Worker(new URL("./network-train.js", import.meta.url), { type: "module" });
}

export function createMlpTrainWorker(): Worker {
  return new Worker(new URL("./mlp-train.js", import.meta.url), { type: "module" });
}

/** Grad-shard workers are normally spawned by coordinators; exposed for tests. */
export function createGradShardWorker(): Worker {
  return new Worker(new URL("./grad-shard.js", import.meta.url), { type: "module" });
}
