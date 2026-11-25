---
title: "Catalog main page: full-width, search and view toggle in one header row"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-7f846a41
scope: repo
status: active
src: agent
touches:
  - packages/playground/src/network-playground.css
  - packages/playground/src/pages/CatalogPage.tsx
  - packages/playground/src/catalog/search.ts
  - packages/playground/src/catalog/CatalogTopicGraph.tsx
  - packages/playground/src/i18n.ts
---

## Decided

- `.nn-page` (catalog main page) has no `max-width` and no auto-centering on purpose: the user wanted the blank side bands of the old centered 1440px column gone on wide monitors (2026-11-25). The 2rem side padding stays; don't bring a max-width back without asking.
- The header row `.catalog-intro-head` holds title, search box, Graph/List switch and locale switcher, all 2rem tall. The user asked to lift the switch and then the search out of their own rows, and to drop the old toolbar row with its "Fit" button (React Flow's own Controls still fit the view).
- One search for both views: `searchCatalog` in `catalog/search.ts`, query state in `CatalogPage`. A hit lights itself, its path to the root and the whole branch under a hit node; the graph dims the rest, the list drops it. No match shows "Nothing found" in both views.
- `.catalog-graph-canvas` is `100dvh - 190px`: retuned each time a header row was removed so the canvas bottom edge stays put.

## Tried and ruled out

- `nn-page--compact` for the catalog instead of editing `.nn-page`: it also swaps in the viz pages' 0.5rem/1rem paddings.
- The toggle at its old 38px height in the header: taller than the 2rem locale pill beside it. Buttons went from `padding: 0.35rem 1.1rem` to `0 1.1rem`, wrapper got `height: 2rem`.
- Keeping the old "N found" counter: it lived in the removed toolbar and would shift the header as it appears. The empty-state message covers its "0" case.

## Watch out

- "Боковые колонки" on the main page meant the empty margins around the centered content, not a layout column. Look at a screenshot before hunting for a sidebar.
- `viewportForBounds` (`CatalogTopicGraph`) falls back to `window.innerWidth - 64` = 2 × the catalog's 2rem side padding. Keep them in sync.
- Verify the header at 768px: `useIsMobile` swaps in `MobileStub` below that, so the `@media (max-width: 640px)` header rules are out of the catalog's range. 768px only fits thanks to the `@media (max-width: 800px)` gap override.
- Vite dev server twice served stale modules after several edits within milliseconds: once a CSS file with only the first edit, once a TSX half old, half new. Disk was right; `curl http://localhost:5173/src/<file>` showed it; `touch` fixed it. Check the served module before blaming the browser or the edit.
