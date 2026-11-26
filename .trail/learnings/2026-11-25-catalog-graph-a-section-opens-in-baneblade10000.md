---
title: "Catalog graph: a section opens in place, the map never leaves the screen"
author: Baneblade10000
date: 2026-11-25
branch: master
cost: { minutes: 30 }
id: l-0ed9bd07
scope: repo
status: active
replaces: l-5a9d3bbf
src: agent
touches:
  - packages/playground/src/catalog/layout.ts
  - packages/playground/src/catalog/CatalogTopicGraph.tsx
  - packages/playground/src/catalog/tree.ts
  - packages/playground/src/network-playground.css
---

## Decided
- The direction of the graph catalog is «фокус на разделе»: an overview of the eight sections with one of them open at a time. The person chose it on 2026-11-25 from three layouts.
- The open section stays on the same screen. Its visualizations unfold next to its tile, the other tiles only dim, the root and the edges stay where they were. Where the canvas is too narrow for room beside the tiles, the same map stays on top and the cards come out in a strip under it.
- Why the first screen is unreadable stays true whatever the layout: the default view expands all 30 subtopics, that tree is 1858×1902 world units, the canvas is wide and low, and fitting it by height gives zoom 0.44 (2222×1034 canvas, labels 6 px, tree on 37% of the width). Ready visualizations go on screen at 100%, planned topics stay collapsed until a section is opened.
- The page stays full width with the one header row.

## Tried and ruled out
- A detail view that replaced the canvas and offered «‹ Все разделы» to get back. The person: it reads as going to another page. It was a mockup, never code.
- A radial layout with readable boxes: root, 8 section boxes and the cards on three levels around a centre need about 900 px of canvas height. A 1440×900 window gives the canvas 710 px.

## Watch out
- The count badge on every section hub reads 0. `vizCountOf` in `catalog/layout.ts` returns the node's own `vizIds`, and a section keeps theirs on its subtopics. Sum the subtree.
- The fit in `fitCanvas` (`CatalogTopicGraph.tsx`) lands only once the page paints a frame. In a webview that was not painting the transform stayed identity for seconds. A foreground tab never shows it; an embedded preview or a shared in-app browser tab does.
- Box sizes are written in three places that must agree: `COLUMN_X`, `SLOT_HEIGHT`, `SUBTOPIC_HEIGHT` and `CARD_HEIGHT` in `layout.ts`, `NODE_SIZE` in `CatalogTopicGraph.tsx`, and the rules in `network-playground.css`. Bring them into one place before changing a size.
- New catalog styles take `--nn-*` tokens from `public/tokens.css` only: a core test fails on a written-out color.

In the code: packages/playground/src/catalog/layout.ts, packages/playground/src/catalog/CatalogTopicGraph.tsx, packages/playground/src/catalog/tree.ts, packages/playground/src/network-playground.css
