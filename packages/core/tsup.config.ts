import { defineConfig } from "tsup";

// Two configs run in parallel; only the first may `clean`, otherwise workers
// (or index) briefly vanish and Vite HMR dies mid-reload.
export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    sourcemap: true,
    clean: true,
    treeshake: true,
  },
  {
    entry: {
      "workers/cnn-train": "src/playground/workers/cnn-train.ts",
      "workers/network-train": "src/playground/workers/network-train.ts",
      "workers/mlp-train": "src/playground/workers/mlp-train.ts",
      "workers/grad-shard": "src/playground/workers/grad-shard.ts",
      "workers/createWorkers": "src/playground/workers/createWorkers.ts",
    },
    format: ["esm"],
    // Only emit declarations for the factory entry (worker scripts are runtime-only).
    dts: {
      entry: {
        "workers/createWorkers": "src/playground/workers/createWorkers.ts",
      },
    },
    sourcemap: true,
    clean: false,
    treeshake: true,
    // Keep worker files as separate modules (no code-splitting into shared chunks
    // that the Worker constructor can't resolve).
    splitting: false,
  },
]);

