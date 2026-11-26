---
title: "raw.githack nodes lag a deploy by minutes; retry, don't redeploy"
author: Baneblade10000
date: 2026-11-26
branch: master
id: l-b2444131
scope: repo
status: active
replaces: l-32a4815a
src: agent
touches:
  - ".github/workflows/pages.yml"
---

## Decided

The deploy address stays https://raw.githack.com/baneblade10000/ml_vis_lib/gh-pages/index.html#/ — the pages workflow publishes dist to gh-pages, raw.githubusercontent (the origin githack proxies) serves the new content immediately. But githack itself runs several proxy nodes whose caches refresh independently: for a few minutes (seen up to ~20) after a deploy, some requests return pre-deploy copies while others return fresh ones — one file was fresh while its sibling stylesheet stayed old, and the same URL flip-flopped between tries.

## Watch out

«Страницы отличаются» right after a deploy is almost always this: a stale githack node, not a bad deploy. Verify by curl-comparing the file against the local copy via raw.githack, then via raw.githubusercontent: if origin is fresh and githack is stale, just retry — repeated requests land on a fresh node (took 5 tries once). A hard refresh (Ctrl+Shift+R) does the same in the browser. Query strings do NOT bypass: githack keys the cache by path.

## In the code

- `.github/workflows/pages.yml`: push to master → build (BASE_PATH=./) → force-publish `packages/playground/dist` to gh-pages.
