import { copyFileSync, existsSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

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
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
});
