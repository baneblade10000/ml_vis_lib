---
title: "Pages embedded from public/: one shared iframe class and a token mirror for canvas"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-34205aee
scope: repo
status: active
src: agent
touches:
  - eslint.config.mjs
  - packages/playground/public/constructor/app.js
  - packages/playground/public/tflow/app.js
  - packages/playground/src/catalog/tree.ts
---

## Decided
- TransformerFlow and Ensemble Playground are embedded through one shared `.public-page-frame` iframe. An iframe does not inherit the catalog's custom properties, so an embedded page redeclares the `--nn-*` tokens on its own `:root` (a mirror of network-playground.css) and its canvases read them through a small palette module, so DOM and canvas stay on one palette.
- Class colors of the embedded ensemble page sit on the core numeric scale (class 0 indigo → white → class 1 blue) to match the decision plane of the network playground; the boosting accent is the catalog accent of «Classical ML».

## Tried and ruled out
- Hex colors hard-coded in canvas code: they drift from the stylesheet as soon as a token changes.
- Importing the playground stylesheet into a public page: it is a separate document with its own class names.

## Watch out
- Canvas text drawn before the web font loads keeps the fallback: redraw once `document.fonts.ready` resolves.
- A grid inside a centered flex column shrinks to one column; give it `width: 100%` when it is meant to fill the card.

In the code: packages/playground/src/visualizations/index.ts, packages/playground/src/network-playground.css, packages/playground/public/ensemble/style.css, packages/playground/public/ensemble/palette.js
