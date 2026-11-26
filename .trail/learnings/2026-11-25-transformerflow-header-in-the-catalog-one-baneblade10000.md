---
title: "TransformerFlow: header in the catalog, one token file for every page"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-bd538c0f
scope: repo
status: active
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
- When the catalog frames TransformerFlow (`#embedded` in the frame's URL), the page shows no header of its own: its tabs, the model badge and the «Открыть»/«Сохранить» actions sit in the catalog's header. Opened on its own, the page keeps its header. A header under a header was called out as wrong.
- The page reports its header with a `ml-vis:header` message: the tab it switched to on its own, its title, the model badge. Tabs and actions are declared in the catalog and localized like the rest of its header; a pick is sent back as `ml-vis:tab` / `ml-vis:action`.
- A click in the catalog's header carries user activation into the same-origin frame, so «Открыть» opens the page's own file chooser. Nothing moved out of the page.
- The design tokens are one file, `public/tokens.css`, linked by the catalog's index.html and by the pages in public/ (an iframe inherits no custom properties). The catalog's stylesheet has no `:root` block any more. TransformerFlow carries no color of its own: styles on the `--nn-*` tokens, canvases reading them through `tflow/palette.js`. Four tokens were added for what pages lacked: `--nn-on-accent`, `--nn-crimson-soft`, `--nn-surface-glass`, `--nn-scrim`.
- A core test keeps it that way: it fails on a color written out in the page's files or on a token neither `tokens.css` nor the page defines, and keeps `--scale-*` equal to the core constants.
- The frame of an embedded page fills the viewport under the header; it used to stop about 4rem short. Every control of that header row is 2rem tall, as in the catalog's own header.

## Tried and ruled out
- A copy of the `--nn-*` block in each page of public/: it drifts as soon as a token changes. One linked file is the same idea without the copying. Ensemble still carries its copy; pointing it at `tokens.css` is a two-line change.
- Tab and action labels sent by the page: the catalog localizes its header itself.
- Replacing every literal left in the catalog's stylesheet (white, indigo glows, teal and green states): they are not token values and need semantic tokens of their own. Only the literals equal to a token were swapped, which changes nothing on screen (checked: same computed colors on every element of the five catalog pages).
- Moving the file-chooser and save code into the catalog to dodge the user-activation question: unnecessary, activation does reach the frame.

## Watch out
- Two sessions edited this working tree at the same time (the other one: the Ensemble page's header) and shared `VizPage.tsx`, `visualizations/index.ts`, `i18n.ts` and `network-playground.css`. Read a shared file right before each edit, add beside the other's rules instead of rewriting them, and check at the end that both sets of changes are still there.
- The in-app browser's tabs are shared between sessions by hostname: the other session navigated a `localhost` tab away in the middle of a test, and in the shared pane clicks may never reach the page. Verify on `127.0.0.1`, or in a headless Chromium: `playwright-core` from `~/.npm/_npx/*/node_modules` with the browsers in `~/.cache/ms-playwright`. It also gives real clicks, a file chooser and downloads.
- In index.html the tokens link keeps its leading slash (`href="/tokens.css"`): Vite recognizes a public file by it and rebases it with the base (`./` on Pages). A page in public/ links the file as `./tokens.css` or `../tokens.css`.
- `public/tflow/app.js` is an orphan, no page loads it. The live script is `constructor/app.js`.
- The title and the badge in the catalog's header are in the page's own language; only tabs and actions follow the catalog's locale.
- `trail check` flags `.nn-page--compact:has(.public-page-frame)` as bringing back the compact page for the catalog. It does not: the catalog has no such frame, and the rule mirrors the one the immersive playgrounds already use.
- Nothing is committed: the tree also holds the other session's uncommitted work.

In the code: packages/playground/public/tokens.css, packages/playground/index.html, packages/playground/src/network-playground.css, packages/playground/src/pages/VizPage.tsx, packages/playground/src/visualizations/index.ts, packages/playground/src/i18n.ts, packages/playground/public/transformer-constructor.html, packages/playground/public/tflow/style.css, packages/playground/public/tflow/palette.js, packages/playground/public/constructor/constructor.css, packages/playground/public/constructor/app.js, packages/core/src/playground/network/engine.test.ts, eslint.config.mjs
