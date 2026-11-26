---
title: "Deploy lands on gh-pages, read via raw.githack.com"
author: Baneblade10000
date: 2026-11-26
branch: master
id: l-32a4815a
scope: repo
status: active
src: agent
touches:
  - ".github/workflows/pages.yml"
---

## Decided

The playground deploy is read from raw.githack.com, not jsDelivr or github.io: after the pages workflow pushes dist to the gh-pages branch, the address to open and share is
https://raw.githack.com/baneblade10000/ml_vis_lib/gh-pages/index.html#/

raw.githack serves the gh-pages branch directly with correct MIME types, so every deploy lands there the moment the workflow finishes — no extra publish step exists or is needed.

## Watch out

A jsDelivr URL (cdn.jsdelivr.net/gh/...) resolves too and is easy to reach for, but it is the wrong pointer for this repo: stale CDN caching and not the address the person shares.

## In the code

- `.github/workflows/pages.yml`: on push to master it builds (BASE_PATH=./) and force-publishes `packages/playground/dist` to gh-pages.
