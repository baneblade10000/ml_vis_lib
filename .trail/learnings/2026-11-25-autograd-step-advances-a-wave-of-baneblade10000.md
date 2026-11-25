---
title: "Autograd step advances a wave of ready nodes, not one node"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-8869fed7
scope: repo
status: active
src: agent
touches:
  - "packages/core/src/playground/autograd/graph.ts#AutogradGraph.step"
---

## Decided
- One press of "Step" in the computational graph takes every operation node that is ready at the same time, in both passes, because the user asked for operations that parallelize to light up together instead of one after another. `AutogradGraph.step()` returns `AutogradStep { ids, pass }`; `nextStep` and `lastStep` are waves too, and the React adapter marks every node of a wave as active or next.
- A node is forward-ready once all of its inputs have a value, and backward-ready once all of its consumers are done. A node still waits for its slowest input (or consumer), so chains stay one node per step.
- The node-order tricks in the `mlp` and `attention` presets (x2 added before x1, k2 before k1, the reversed fan-out loop) were removed: they only shaped the old one-node-per-step walk.

## Tried and ruled out
- Keeping `AutogradStep.id` and adding wave arrays beside it: the adapter, the `stepInput` edges and the tests all read the step, so one shape is simpler.

## Watch out
- Do not turn the step back into a single node, and do not bring node-order tricks back into presets: inside a wave the order is not visible.
- The Vite dev server on :5173 served stale transforms of files that were edited twice within seconds (an old `step()` next to a new `nextStep`), so "Step" threw `Cannot read properties of undefined (reading 'id')` in the browser while vitest and tsc were green, because they read the disk. After an edit, check what the server returns: `curl http://localhost:5173/@fs/<absolute path>`; `touch <file>` forces a re-transform.
- The playground uses `HashRouter`: a visualization lives at `http://localhost:5173/#/viz/<id>`. The same path without the `#` shows the catalog with no error.
