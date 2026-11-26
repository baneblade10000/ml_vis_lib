---
title: "Graph catalog: fitting all 30 leaves makes it unreadable; section badges read 0"
author: Baneblade10000
date: 2026-11-25
branch: master
cost: { minutes: 40 }
id: l-5a9d3bbf
scope: repo
status: active
src: agent
touches:
  - packages/playground/src/catalog/layout.ts
  - packages/playground/src/catalog/CatalogTopicGraph.tsx
  - packages/playground/src/catalog/tree.ts
  - packages/playground/src/network-playground.css
---

## Decided
- The graph catalog is unreadable because of what the default view shows, not because of a rendering fault. It expands all 30 subtopics; that tree is 1858×1902 world units, almost square, while the canvas is wide and low (2222×1034 on a wide monitor, `100dvh - 190px`). `viewportForBounds` fits by height: zoom 0.44, labels 6 px, the tree on 37% of the canvas width. The same formula gives 0.30 on a 1440×900 laptop. Measured on the live page on 2026-11-25.
- So a redesign must not count on fitting the whole tree. Ready visualizations go on screen at 100%, planned topics stay collapsed until asked for. Three layouts were proposed for that (compact tree with a details panel, section map with previews, overview of 8 sections with drill-down); the person had not chosen one when this was written.
- The page stays full width with the one header row.

## Tried and ruled out
- A radial layout with readable boxes: root, 8 section boxes and the cards on three levels around a centre need about 900 px of canvas height. A 1440×900 window gives the canvas 710 px.

## Watch out
- The count badge on every section hub reads 0. `vizCountOf` in `catalog/layout.ts` returns the node's own `vizIds`, and a section keeps theirs on its subtopics, so the badge of a section is always 0 while its branch has visualizations. Sum the subtree.
- The fit in `fitCanvas` (`CatalogTopicGraph.tsx`) lands only once the page paints a frame. In a webview that was not painting (0 animation frames in half a second) the transform stayed identity for seconds and the tree overflowed the canvas. A foreground tab never shows it; an embedded preview or a shared in-app browser tab does.
- Box sizes are written in three places that must agree: `COLUMN_X`, `SLOT_HEIGHT`, `SUBTOPIC_HEIGHT` and `CARD_HEIGHT` in `layout.ts`, `NODE_SIZE` in `CatalogTopicGraph.tsx`, and the rules in `network-playground.css`. Bring them into one place before changing a size.
- New catalog styles take `--nn-*` tokens from `public/tokens.css` only: a core test fails on a written-out color.
