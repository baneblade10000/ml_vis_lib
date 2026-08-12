# Burn CNN WASM used to live here.

Artifacts are now under `packages/playground/src/burn/pkg/` so Vite can
`import()` the glue JS (files in `/public` cannot be imported from source).

Rebuild with:

```bash
pnpm burn:build
```
