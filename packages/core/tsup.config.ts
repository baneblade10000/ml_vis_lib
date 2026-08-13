import { defineConfig } from "tsup";

// Both configs run in parallel — never `clean` here. package.json runs
// `rm -rf dist` first so one config can't wipe the other's output mid-build
// (that race dropped createWorkers.d.ts in CI and broke @ml-vis/react dts).
export default defineConfig([
  {
    // Root barrel + per-domain subpath barrels. Each entry is also a public
    // subpath in package.json `exports` (e.g. @ml-vis/core/network).
    entry: {
      index: "src/index.ts",
      section: "src/section/index.ts",
      signal: "src/signal/index.ts",
      charts: "src/charts/index.ts",
      i18n: "src/i18n/index.ts",
      network: "src/playground/network/index.ts",
      cnn: "src/playground/cnn/index.ts",
      autograd: "src/playground/autograd/index.ts",
      mlp: "src/playground/index.ts",
      workers: "src/playground/workers/index.ts",
    },
    format: ["esm", "cjs"],
    dts: true,
    sourcemap: true,
    clean: false,
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

