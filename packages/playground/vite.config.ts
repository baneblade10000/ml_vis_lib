import { copyFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const root = path.dirname(fileURLToPath(import.meta.url));
const reactSrc = path.resolve(root, "../react/src");
const coreSrc = path.resolve(root, "../core/src");

export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [
    react(),
    {
      name: "spa-github-pages-fallback",
      closeBundle() {
        if (!existsSync("dist/index.html")) return;
        copyFileSync("dist/index.html", "dist/404.html");
      },
    },
  ],
  resolve: {
    // Dev: workspace source. Subpath exports must be mapped before the bare
    // package, and the deeper workers/* regex before the bare workers barrel.
    alias: [
      {
        find: /^@ml-vis\/core\/workers\/(.+)$/,
        replacement: path.join(coreSrc, "playground/workers/$1.ts"),
      },
      { find: "@ml-vis/core/section", replacement: path.join(coreSrc, "section/index.ts") },
      { find: "@ml-vis/core/signal", replacement: path.join(coreSrc, "signal/index.ts") },
      { find: "@ml-vis/core/charts", replacement: path.join(coreSrc, "charts/index.ts") },
      { find: "@ml-vis/core/i18n", replacement: path.join(coreSrc, "i18n/index.ts") },
      {
        find: "@ml-vis/core/network",
        replacement: path.join(coreSrc, "playground/network/index.ts"),
      },
      { find: "@ml-vis/core/cnn", replacement: path.join(coreSrc, "playground/cnn/index.ts") },
      {
        find: "@ml-vis/core/transformer",
        replacement: path.join(coreSrc, "playground/transformer/index.ts"),
      },
      {
        find: "@ml-vis/core/autograd",
        replacement: path.join(coreSrc, "playground/autograd/index.ts"),
      },
      { find: "@ml-vis/core/mlp", replacement: path.join(coreSrc, "playground/index.ts") },
      {
        find: "@ml-vis/core/workers",
        replacement: path.join(coreSrc, "playground/workers/index.ts"),
      },
      { find: "@ml-vis/react", replacement: path.join(reactSrc, "index.ts") },
      { find: "@ml-vis/core", replacement: path.join(coreSrc, "index.ts") },
    ],
  },
  assetsInclude: ["**/*.wasm"],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
});
