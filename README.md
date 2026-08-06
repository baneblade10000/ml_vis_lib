# ML Visualizations

Interactive demos for exploring machine learning concepts. Canvas 2D renderer with automatic downsampling for large datasets.

## Stack

- **pnpm** + **Turborepo** monorepo
- **TypeScript** strict mode
- **@ml-vis/core** — framework-agnostic Canvas renderer
- **@ml-vis/react** — thin React wrappers + Storybook
- **@ml-vis/playground** — demo app
- **Docker** + **nginx** + **cloudflared** for deployment
- **GitHub Pages** for the playground demo

## Quick start (local)

```bash
pnpm install
pnpm build
pnpm dev
```

- Playground: http://localhost:5173
- Storybook: `pnpm storybook` → http://localhost:6006

## Docker (production)

```bash
# Build and run playground (local access on :8080)
cp .env.example .env
docker compose up -d --build web

# With Cloudflare Tunnel
# 1. Create tunnel in Cloudflare Zero Trust
# 2. Set public hostname → http://web:80
# 3. Copy token to .env as TUNNEL_TOKEN
docker compose --profile tunnel up -d --build
```

Health check: `curl http://localhost:8080/health`

## GitHub Pages

Push to `master` (or run **Deploy GitHub Pages** manually) publishes the playground to:

https://baneblade10000.github.io/ml_vis_lib/

Setup once: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

Local build with the same base path:

```bash
BASE_PATH=/ml_vis_lib/ pnpm build
pnpm --filter @ml-vis/playground preview
```

## Packages

| Package | Description |
|---------|-------------|
| `@ml-vis/core` | ScatterPlot, section layout/registry/store, data utils |
| `@ml-vis/react` | `<ScatterPlotChart />`, `<Section>`, `<SectionLayout>`, `<SectionNav>` |
| `@ml-vis/playground` | Interactive demo |

## Development

```bash
pnpm test          # vitest in core
pnpm lint          # typecheck all packages
pnpm build         # build all packages
```

## License

MIT
