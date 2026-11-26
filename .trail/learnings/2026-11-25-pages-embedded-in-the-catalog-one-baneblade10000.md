---
title: "Pages embedded in the catalog: one header, full width, tabs and switches where the eye lands"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-e709219e
scope: repo
status: active
replaces: l-34205aee
src: agent
touches:
  - packages/playground/public/ensemble.html
  - packages/playground/public/ensemble/main.js
  - packages/playground/public/ensemble/style.css
  - packages/playground/src/i18n.ts
  - packages/playground/src/network-playground.css
  - packages/playground/src/pages/VizPage.tsx
  - packages/playground/src/visualizations/index.ts
  - packages/playground/public/ensemble/palette.js
---

## Decided
- A page embedded from public/ shows no header of its own when the catalog opens it: the catalog header carries the page's tabs (bilingual, like the rest of the catalog) and tells the page which one is active. A header inside a header was called out as wrong.
- An embedded page spans the full width: no max-width, no centring, a small side padding. The empty bands around a centred column are unwanted here as on the catalog main page.
- On the ensemble page the switch between the two ensembles sits first in the left control card, and the right card shows the parameters of the selected one only. A count of trees next to the switch was removed as redundant: that number is a setting on the right.
- Canvases read the design tokens through a small palette module. An iframe inherits no custom properties, so the page redeclares the house set on its own root.

## Tried and ruled out
- The page's own title bar and tab row inside the frame: it reads as a second header under the catalog's.
- Steering the tabs through the frame's URL: navigating the frame would drop the trained models, a message keeps them.
- A switch floating above the model card, with a "N trees" badge beside it.

## Watch out
- Without the marker in the frame's URL the page still shows its own header, so opening it directly stays usable.
- The page hears a tab only once it has loaded: send it again on the frame's load event.
- Author styles on a hidden element beat the browser's default, so keep the global hidden override.
- A container-query rule written before the base rule of the same selector is overridden by it.

In the code: packages/playground/src/visualizations/index.ts, packages/playground/src/pages/VizPage.tsx, packages/playground/src/network-playground.css, packages/playground/public/ensemble.html, packages/playground/public/ensemble/style.css, packages/playground/public/ensemble/main.js, packages/playground/public/ensemble/palette.js
