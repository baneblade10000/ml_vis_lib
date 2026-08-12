import type { LayerKind } from "@ml-vis/core";

/** Spatial layer row used to back-project a feature-map pixel to the input. */
export type RfLayerMeta = {
  id: string;
  kind: LayerKind;
  /** Conv kernel / pool window; ignored for input. */
  kernelSize: number;
  rows: number;
  cols: number;
};

/** Inclusive axis-aligned rect on a feature map / input grid. */
export type RfRect = { y0: number; x0: number; y1: number; x1: number };

/** RF highlight on one preceding layer. */
export type RfLayerHighlight = RfRect & {
  /**
   * When set, only this channel is highlighted (pool is channel-preserving).
   * When omitted, every channel of the layer gets the spatial rect (conv mixes).
   */
  channel?: number;
};

export type RfSelection = {
  sourceLayerId: string;
  channel: number;
  outY: number;
  outX: number;
  /**
   * Receptive-field rect on every preceding spatial layer (including input),
   * keyed by layer id. Source layer itself is omitted — the picked pixel is
   * drawn separately.
   */
  byLayer: Record<string, RfLayerHighlight>;
};

function clampRect(rect: RfRect, rows: number, cols: number): RfRect {
  const y0 = Math.max(0, Math.min(rows - 1, rect.y0));
  const y1 = Math.max(0, Math.min(rows - 1, rect.y1));
  const x0 = Math.max(0, Math.min(cols - 1, rect.x0));
  const x1 = Math.max(0, Math.min(cols - 1, rect.x1));
  return {
    y0: Math.min(y0, y1),
    y1: Math.max(y0, y1),
    x0: Math.min(x0, x1),
    x1: Math.max(x0, x1),
  };
}

/**
 * Expand one output cell through `layer` onto the previous activation grid.
 * Matches the WASM engine: same-pad stride-1 conv, 2×2 stride-2 pool.
 */
function expandThroughLayer(layer: RfLayerMeta, rect: RfRect): RfRect | null {
  if (layer.kind === "conv2d") {
    const k = Math.max(1, layer.kernelSize);
    const pad = Math.floor(k / 2);
    return {
      y0: rect.y0 - pad,
      y1: rect.y1 - pad + k - 1,
      x0: rect.x0 - pad,
      x1: rect.x1 - pad + k - 1,
    };
  }
  if (layer.kind === "pool2d") {
    return {
      y0: rect.y0 * 2,
      y1: rect.y1 * 2 + 1,
      x0: rect.x0 * 2,
      x1: rect.x1 * 2 + 1,
    };
  }
  return null;
}

/**
 * Cascade a pixel on `sourceLayerId` / `sourceChannel` back through every
 * previous spatial layer.
 *
 * Pool preserves channels → highlight only that channel on the previous maps.
 * Conv mixes channels → spatial rect on every previous channel.
 */
export function receptiveFieldCascade(
  layers: RfLayerMeta[],
  sourceLayerId: string,
  outY: number,
  outX: number,
  sourceChannel = 0,
): Pick<RfSelection, "byLayer"> | null {
  const fromIdx = layers.findIndex((l) => l.id === sourceLayerId);
  if (fromIdx <= 0) return null;

  const byLayer: Record<string, RfLayerHighlight> = {};
  let rect: RfRect = { y0: outY, y1: outY, x0: outX, x1: outX };
  /** Channel identity while walking only through pools; cleared by conv. */
  let channel: number | undefined = sourceChannel;

  for (let i = fromIdx; i >= 1; i--) {
    const layer = layers[i]!;
    const prev = layers[i - 1]!;
    const next = expandThroughLayer(layer, rect);
    if (!next) return null;
    rect = clampRect(next, prev.rows, prev.cols);

    if (layer.kind === "pool2d" || layer.kind === "pool1d") {
      byLayer[prev.id] =
        channel != null ? { ...rect, channel } : { ...rect };
    } else if (layer.kind === "conv2d" || layer.kind === "conv1d") {
      // Output pixel depends on every input channel → paint all.
      byLayer[prev.id] = { ...rect };
      channel = undefined;
    } else {
      byLayer[prev.id] =
        channel != null ? { ...rect, channel } : { ...rect };
    }
  }

  return { byLayer };
}

/** @deprecated use {@link receptiveFieldCascade} */
export const receptiveFieldOnInput = receptiveFieldCascade;

/** Build RF metas from the live pipeline + optional kernel sizes from dumps. */
export function buildRfLayerMetas(
  layers: Array<{
    id: string;
    kind: LayerKind;
    shape: { kind: string; rows?: number; cols?: number; length?: number; channels: number };
  }>,
  kernelSizeByLayerId: Record<string, number>,
): RfLayerMeta[] {
  return layers.map((layer) => {
    const shape = layer.shape;
    const rows = shape.kind === "2d" ? (shape.rows ?? 1) : 1;
    const cols =
      shape.kind === "2d" ? (shape.cols ?? 1) : shape.kind === "1d" ? (shape.length ?? 1) : 1;
    const kernelSize =
      kernelSizeByLayerId[layer.id] ??
      (layer.kind === "pool2d" || layer.kind === "pool1d" ? 2 : 3);
    return {
      id: layer.id,
      kind: layer.kind,
      kernelSize,
      rows: Math.max(1, rows),
      cols: Math.max(1, cols),
    };
  });
}
