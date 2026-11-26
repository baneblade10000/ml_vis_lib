---
title: "Catalog graph: the map of sections stays, the chosen one opens as a block beside its tile"
author: Baneblade10000
date: 2026-11-25
branch: master
cost: { minutes: 25 }
id: l-c5bd8f82
scope: repo
status: active
replaces: l-0ed9bd07
src: agent
touches:
  - packages/playground/src/catalog/layout.ts
  - packages/playground/src/catalog/CatalogTopicGraph.tsx
  - packages/playground/src/catalog/tree.ts
  - packages/playground/src/network-playground.css
---

## Decided
- The graph catalog keeps the map the person liked: the root in the middle, eight section tiles around it (three a side, one on top, one below), a tile with mini previews or with its planned topics.
- One section is open at a time. It opens as a block in the free band beside its tile, joined to the tile by an edge. Inside is a row per topic: a pill, then its visualization cards, or «появится позже» when the topic is only planned. The other tiles and the root stay where they are and only dim. A section without visualizations opens the same way.
- Nothing navigates away: no separate layout, no «Все разделы» link, no mini-map. Where the canvas is too narrow for the bands, the same map stays on top and the block opens under it.
- The live graph is unreadable because the default view expands all 30 subtopics: the tree is 1858×1902 world units and fitting it by height gives zoom 0.44 with 6 px labels. Ready visualizations go on screen at 100%, planned topics stay collapsed.

## Tried and ruled out
- A detail view that replaced the canvas and offered «‹ Все разделы»: the person reads it as going to another page (2026-11-25).
- Visualization cards unfolding beside the tile without topic nodes: the person wants the topics inside the block.
- A radial layout with readable boxes: it needs about 900 px of canvas height, a 1440×900 window gives the canvas 710.

## Watch out
- The count badge on a section hub reads 0: `vizCountOf` in `catalog/layout.ts` returns the node's own `vizIds`, and a section keeps theirs on its subtopics. Sum the subtree.
- The fit in `fitCanvas` (`CatalogTopicGraph.tsx`) lands only once the page paints a frame.
- Box sizes are written in `layout.ts`, in `NODE_SIZE` of `CatalogTopicGraph.tsx` and in `network-playground.css`, and must agree.
- New catalog styles take `--nn-*` tokens only: a core test fails on a written-out color.

In the code: packages/playground/src/catalog/layout.ts, packages/playground/src/catalog/CatalogTopicGraph.tsx, packages/playground/src/catalog/tree.ts, packages/playground/src/network-playground.css
