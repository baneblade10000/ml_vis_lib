---
title: "Autograd nodes: one input port per operand, declared as data"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-a3b67e79
scope: repo
status: active
src: agent
touches:
  - "packages/core/src/playground/autograd/graph.ts#AutogradGraph.connect"
  - "packages/react/src/playground/autograd/adapter.ts#autogradToFlow"
---

## Decided
- Every edge carries the input port of its target it feeds (`AutogradEdge.slot`), and an operation shows exactly one port per operand: unary 1, binary 2, n-ary `add`/`mul` at least 2 and one more than its highest fed port. The user asked for two inputs on binary operations and for one argument to feed both (x·x), which the gradient collects over both edges.
- A wire dropped on a free port fills it. On a fed port, an operation of fixed arity hands the port to the new source, while an n-ary operation gains an operand instead, so `+` and `×` keep the old "drop adds an input" behaviour.
- A removed edge leaves its port empty instead of shifting the others: for `sub` and `div` the port is the operand's position, and an unfed port reads as a missing operand.
- React Flow ports are declared on the node (`node.handles`, @xyflow/system 0.0.78) rather than left to DOM measurement, so an edge and its new port appear in one frame.

## Tried and ruled out
- A spare "add an operand" port on n-ary nodes: it puts a third input on `a+b`, which the user asked to have two.
- `useUpdateNodeInternals` once a port appears: a frame late, with React Flow's 008 warning in between.

## Watch out
- The declared port box (9px) must match `.ag-handle`, which is `box-sizing: border-box`: a mismatch makes edges jump when React Flow re-measures.
- `connect` without a port rejects a fixed-ary operation that is fully fed; the UI always names the port.
