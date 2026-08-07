# ML Visualizations

Interactive demos for exploring machine learning concepts. Canvas 2D renderer with automatic downsampling for large datasets.

## Stack

- **pnpm** + **Turborepo** monorepo
- **TypeScript** strict mode
- **@ml-vis/core** — framework-agnostic Canvas renderer
- **@ml-vis/react** — thin React wrappers + Storybook
- **@ml-vis/playground** — demo app
- **Docker** + **nginx** + **cloudflared** for deployment
- Static playground publish via **`gh-pages` + raw.githack** (GitHub Pages queue is unreliable for this repo)

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

## Public demo

Push to `master` publishes `packages/playground/dist` to the `gh-pages` branch.

raw.githack caches the **branch** URL aggressively, so after a deploy open the
**commit-pinned** URL (tip of `gh-pages`, from the Actions “deploy: &lt;sha&gt;” commit):

https://raw.githack.com/baneblade10000/ml_vis_lib/8e9d9fc/index.html

(Branch URL may stay stale for hours: `…/gh-pages/index.html`.)

Workflow: https://github.com/baneblade10000/ml_vis_lib/actions/workflows/pages.yml

```bash
BASE_PATH=./ pnpm build
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
