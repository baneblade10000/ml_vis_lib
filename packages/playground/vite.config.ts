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
    // Dev: workspace source. Subpath exports must be mapped before the bare package.
    alias: [
      {
        find: /^@ml-vis\/core\/workers\/(.+)$/,
        replacement: path.join(coreSrc, "playground/workers/$1.ts"),
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
