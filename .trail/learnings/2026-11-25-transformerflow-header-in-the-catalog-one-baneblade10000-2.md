---
title: "TransformerFlow: header in the catalog, one token file for every page"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-718a62c7
scope: repo
status: active
replaces: l-bd538c0f
src: agent
touches:
  - eslint.config.mjs
  - packages/core/src/playground/network/engine.test.ts
  - packages/playground/index.html
  - packages/playground/public/constructor/app.js
  - packages/playground/public/constructor/constructor.css
  - packages/playground/public/ensemble.html
  - packages/playground/public/ensemble/main.js
  - packages/playground/public/ensemble/palette.js
  - packages/playground/public/ensemble/style.css
  - packages/playground/public/tflow/style.css
---

## Decided
- Framed by the catalog (`#embedded` in the frame's URL), TransformerFlow shows no header of its own: its tabs, model badge and «Открыть»/«Сохранить» sit in the catalog's header. Opened on its own, it keeps its header.
- The page reports its header with a `ml-vis:header` message (the tab it switched to on its own, title, badge). Tabs and actions are declared in the catalog, localized like the rest of its header, and sent back as `ml-vis:tab` / `ml-vis:action`. A click there carries user activation into the same-origin frame, so «Открыть» opens the page's own file chooser.
- Design tokens are one file, `public/tokens.css`, linked by index.html and by the pages in public/. TransformerFlow has no color of its own: styles on `--nn-*`, canvases through `tflow/palette.js`. A core test fails on a written-out color, a token nobody defines, or `--scale-*` drifting from the core constants.
- An embedded page fills the viewport under its header (it stopped about 4rem short); the controls of that header are all 2rem tall.

## Tried and ruled out
- A copy of the `--nn-*` block per page: it drifts. Ensemble still has one; pointing it at `tokens.css` is two lines.
- Swapping the catalog's remaining literals (white, indigo glows): not token values, they need semantic tokens of their own. Only literals equal to a token were swapped; the computed colors of the five catalog pages are unchanged.

## Watch out
- Another session edited the same tree at the same time and shared `VizPage.tsx`, `visualizations/index.ts`, `i18n.ts`, `network-playground.css`: re-read before each edit, add beside its rules, check that both sets survive.
- The in-app browser's tabs are shared between sessions by hostname, and clicks there may never reach the page. Verify on `127.0.0.1` or in a headless Chromium (`playwright-core` under `~/.npm/_npx`, browsers in `~/.cache/ms-playwright`): real clicks, file chooser, downloads.
- In index.html keep the slash in `href="/tokens.css"`: that is how Vite knows a public file and rebases it with the base.
- `public/tflow/app.js` is an orphan; the live script is `constructor/app.js`.
- `trail check` flags `.nn-page--compact:has(.public-page-frame)` as the compact catalog page. A false alarm: the catalog has no such frame.

In the code: packages/playground/public/tokens.css, packages/playground/index.html, packages/playground/src/network-playground.css, packages/playground/src/pages/VizPage.tsx, packages/playground/src/visualizations/index.ts, packages/playground/src/i18n.ts, packages/playground/public/transformer-constructor.html, packages/playground/public/tflow/style.css, packages/playground/public/tflow/palette.js, packages/playground/public/constructor/constructor.css, packages/playground/public/constructor/app.js, packages/core/src/playground/network/engine.test.ts, eslint.config.mjs
