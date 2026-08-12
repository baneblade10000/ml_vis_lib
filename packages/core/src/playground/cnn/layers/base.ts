/**
 * Layer taxonomy + shape types for CNN UI / worker snapshots.
 * No forward/backward — compute is Burn WASM only.
 */

export type LayerKind =
  | "input"
  | "conv2d"
  | "pool2d"
  | "conv1d"
  | "pool1d"
  | "gap2d"
  | "gap1d"
  | "flatten"
  | "dense"
  | "output";

export type DataSpace = "2d" | "1d";

export type LayerShape =
  | { kind: "2d"; channels: number; rows: number; cols: number }
  | { kind: "1d"; channels: number; length: number };

export function shapeSpace(shape: LayerShape): DataSpace {
  return shape.kind;
}
