import { useCallback, useContext, useLayoutEffect, useRef, type ReactNode } from "react";
import {
  BaseEdge,
  getBezierPath,
  getStraightPath,
  Handle,
  Position,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import {
  CLASS_0_HEX,
  CLASS_1_HEX,
  renderValueMatrix,
  reduceMatrix,
  weightColor,
  weightValueNormalized,
  type FeatureMapSnapshot,
} from "@ml-vis/core";
import {
  downsample1D,
  unitStackSize,
  type CnnNodeData,
  type CnnEdgeData,
} from "./cnnAdapter";
import {
  CnnPlayVizRefContext,
  FeatureMapRefContext,
  PaintGenerationContext,
  TrainingLiveRefContext,
} from "./featureMapContext";
import { registerBoundaryPainter } from "./featureMapPaint";
import { useCnnMessages } from "./messages";

/** Pixel size of one feature-map tile. */
const MAP_PX = 44;
/** Pixel size of the kernel thumb beside each conv channel output. */
const KERNEL_PX = 36;
/** Cap on tiles shown per node (extra channels are dropped to fit). */
const MAX_TILES = 16;

/** Read the per-node feature-map snapshot from the ref context. */
function useLayerSnapshot(layerId: string): FeatureMapSnapshot | undefined {
  const ref = useContext(FeatureMapRefContext);
  return ref?.current?.[layerId];
}

/** Render a single 2-D map to a small canvas via the shared value→rgba rasterizer. */
function Map2DCanvas({
  map,
  px,
  label,
}: {
  map: number[][];
  px: number;
  label?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !map.length) return;
    // Downsample large maps to ~px for display.
    const target = px;
    let grid = map;
    if (map.length > target) {
      let factor = Math.max(1, Math.round(map.length / target));
      while (factor > 1 && map.length % factor !== 0) factor -= 1;
      if (factor > 1) grid = reduceMatrix(map, factor);
    }
    renderValueMatrix(heat, grid, { layout: "row-major", palette: "gray" });
    // Integer-scale nearest-neighbor into a crisp bitmap, then CSS-scale to `px`
    // with `image-rendering: pixelated` — avoids bilinear smear on tiny maps.
    const rows = grid.length;
    const cols = grid[0]?.length ?? rows;
    const cell = Math.max(1, Math.floor(px / Math.max(rows, cols)));
    const bw = cols * cell;
    const bh = rows * cell;
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
      canvas.style.width = `${px}px`;
      canvas.style.height = `${px}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(heat, 0, 0, bw, bh);
  }, [map, px]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  return (
    <div className="cnn-feature-cell" title={label} style={{ width: px, height: px }}>
      <canvas ref={heatRef} width={1} height={1} hidden aria-hidden />
      <canvas ref={canvasRef} className="cnn-feature-canvas" />
    </div>
  );
}

/** Map kernel cells onto the same tanh-normalized scale as NN edge weights. */
function normalizeWeightMap(map: number[][]): number[][] {
  return map.map((row) => row.map((w) => weightValueNormalized(w)));
}

/** Per-filter bias chip — same diverging palette as NN neuron biases. */
function ConvBiasIndicator({ bias }: { bias: number }) {
  return (
    <span
      className="cnn-filter-bias"
      data-sign={bias >= 0 ? "pos" : "neg"}
      aria-hidden
      title={`bias ${bias.toFixed(3)}`}
      style={{ background: weightColor(weightValueNormalized(bias)) }}
    />
  );
}

/**
 * Mini pixelated kernel preview (nearest-neighbor), sized for pairing with a
 * feature-map tile. Accepts a 2-D map or a 1-row wrapper around a 1-D kernel.
 * Colored with the NN weight palette (violet → orchid → magenta).
 */
function KernelMini({
  map,
  size,
  label,
  bias,
}: {
  map: number[][];
  size: number;
  label?: string;
  bias?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const rows = map.length;
  const cols = map[0]?.length ?? 0;

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !rows || !cols) return;
    renderValueMatrix(heat, normalizeWeightMap(map), {
      layout: "row-major",
      palette: "diverging",
    });
    const cell = Math.max(1, Math.floor(size / Math.max(rows, cols)));
    const bw = cols * cell;
    const bh = rows * cell;
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(heat, 0, 0, bw, bh);
  }, [map, size, rows, cols]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  if (!rows || !cols) {
    return (
      <div
        className="cnn-kernel-mini cnn-kernel-mini--empty"
        style={{ width: size, height: size }}
        title={label}
      />
    );
  }

  const title =
    typeof bias === "number" ? `${label ?? "kernel"} · bias ${bias.toFixed(3)}` : label;

  return (
    <div className="cnn-kernel-with-bias" title={title}>
      {typeof bias === "number" && <ConvBiasIndicator bias={bias} />}
      <div className="cnn-kernel-mini" style={{ width: size, height: size }}>
        <canvas ref={heatRef} width={cols} height={rows} hidden aria-hidden />
        <canvas ref={canvasRef} className="cnn-feature-canvas" />
      </div>
    </div>
  );
}

/** Render a 1-D signal as a thin horizontal bar of pixels. */
function Signal1DCanvas({ values, px }: { values: number[]; px: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !values.length) return;
    // Wrap as a 1-row matrix and rasterize, then draw stretched to height.
    const mat = [values.slice()];
    renderValueMatrix(heat, mat, { layout: "row-major", palette: "gray" });
    const h = 8;
    const cell = Math.max(1, Math.floor(px / values.length));
    const bw = values.length * cell;
    if (canvas.width !== bw || canvas.height !== h) {
      canvas.width = bw;
      canvas.height = h;
      canvas.style.width = `${px}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, h);
    ctx.drawImage(heat, 0, 0, bw, h);
  }, [values, px]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  return (
    <div className="cnn-feature-cell cnn-feature-cell--signal" style={{ width: px, height: 8 }}>
      <canvas ref={heatRef} width={1} height={1} hidden aria-hidden />
      <canvas ref={canvasRef} className="cnn-feature-canvas" />
    </div>
  );
}

/** Activation → grayscale fill for flatten unit squares. */
function activationGray(values: number[]): string[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || max - min < 1e-9) {
    min -= 0.5;
    max += 0.5;
  }
  const span = max - min;
  return values.map((v) => {
    const t = Math.min(1, Math.max(0, (v - min) / span));
    const g8 = Math.round(Math.pow(t, 0.85) * 255);
    return `rgb(${g8},${g8},${g8})`;
  });
}

/** Grid of feature-map tiles for one layer node. */
function FeatureGrid({
  layerId,
  channels,
  mode,
  showKernels = false,
  vertical = false,
}: {
  layerId: string;
  channels: number;
  mode: "2d" | "1d";
  /** When true (conv layers), draw each filter kernel beside its channel output. */
  showKernels?: boolean;
  /** Stack channel tiles in a column (pool layers). */
  vertical?: boolean;
}) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  const trainingLiveRef = useContext(TrainingLiveRefContext);
  const groupRef = useRef<HTMLDivElement>(null);

  // Register a no-op painter keyed by the node so paintAllFeatureMaps() keeps us
  // in sync during the Play loop (the canvases repaint from the live snapshot).
  useLayoutEffect(() => {
    return registerBoundaryPainter(layerId, () => {
      // Re-render is data-driven via paintGeneration; the registration alone
      // ensures paintAllFeatureMapsAfterCommit covers this node.
    });
  }, [layerId]);

  void trainingLiveRef;
  void paintGeneration;

  const maps2d = snapshot?.maps2d ?? [];
  const signals = snapshot?.signals ?? [];
  const kernels2d = snapshot?.kernels2d ?? [];
  const kernels1d = snapshot?.kernels1d ?? [];
  const biases = snapshot?.biases ?? [];
  const showCount = Math.min(channels, MAX_TILES);
  const tiles: ReactNode[] = [];

  if (mode === "2d") {
    for (let i = 0; i < showCount; i++) {
      const map = maps2d[i] ?? [];
      if (showKernels) {
        const kernel = kernels2d[i] ?? [];
        tiles.push(
          <div key={i} className="cnn-channel-pair" title={`kernel → channel ${i + 1}`}>
            <KernelMini
              map={kernel}
              size={KERNEL_PX}
              label={`kernel ${i + 1}`}
              bias={biases[i]}
            />
            <Map2DCanvas map={map} px={MAP_PX} label={`filter ${i + 1}`} />
          </div>,
        );
      } else {
        tiles.push(<Map2DCanvas key={i} map={map} px={MAP_PX} label={`filter ${i + 1}`} />);
      }
    }
  } else {
    for (let i = 0; i < showCount; i++) {
      const values = signals[i] ?? [];
      if (showKernels) {
        const kernel = kernels1d[i] ?? [];
        tiles.push(
          <div key={i} className="cnn-channel-pair cnn-channel-pair--1d" title={`kernel → channel ${i + 1}`}>
            <KernelMini
              map={kernel.length ? [kernel] : []}
              size={KERNEL_PX}
              label={`kernel ${i + 1}`}
              bias={biases[i]}
            />
            <Signal1DCanvas values={values} px={MAP_PX} />
          </div>,
        );
      } else {
        tiles.push(<Signal1DCanvas key={i} values={values} px={MAP_PX} />);
      }
    }
  }

  if (tiles.length === 0) {
    return <div className="cnn-feature-empty" ref={groupRef} />;
  }
  return (
    <div
      className={[
        "cnn-feature-grid",
        showKernels ? "cnn-feature-grid--with-kernels" : "",
        vertical ? "cnn-feature-grid--vertical" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      ref={groupRef}
    >
      {tiles}
      {channels > MAX_TILES && <span className="cnn-feature-more">+{channels - MAX_TILES}</span>}
    </div>
  );
}

function BaseCnnNode({
  data,
  children,
  className,
  hideSource,
  hideTarget,
}: {
  data: CnnNodeData;
  children?: ReactNode;
  className: string;
  hideSource?: boolean;
  hideTarget?: boolean;
}) {
  return (
    <div className={`cnn-node ${className}${data.selected ? " selected" : ""}`}>
      {!hideTarget && <Handle type="target" position={Position.Left} className="cnn-handle" />}
      <div className="cnn-node-head">
        <span className="cnn-node-label">{data.label}</span>
      </div>
      {children}
      {!hideSource && <Handle type="source" position={Position.Right} className="cnn-handle" />}
    </div>
  );
}

export function CnnInputNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--input" hideTarget>
      <FeatureGrid layerId={data.layerId} channels={data.channels} mode={data.mode} />
    </BaseCnnNode>
  );
}

export function CnnConvNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--conv">
      <FeatureGrid
        layerId={data.layerId}
        channels={data.channels}
        mode={data.mode}
        showKernels
      />
    </BaseCnnNode>
  );
}

export function CnnPoolNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--pool">
      <FeatureGrid
        layerId={data.layerId}
        channels={data.channels}
        mode={data.mode}
        vertical
      />
    </BaseCnnNode>
  );
}

/** Flatten column: grayscale unit squares + per-unit source handles for dense edges. */
function FlattenVector({ layerId, length }: { layerId: string; length?: number }) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  void paintGeneration;
  const values = snapshot?.signals?.[0] ?? [];
  const n = values.length || length || 32;
  const raw = values.length ? values : new Array(n).fill(0);
  const stack = unitStackSize(raw.length, 1);
  const shown = downsample1D(raw, stack.visCount);
  const fills = activationGray(shown);
  const { d, gap, width, height } = stack;

  return (
    <div
      className="cnn-unit-stack cnn-unit-stack--flatten"
      style={{ width, height }}
      title={`${raw.length} units`}
    >
      {shown.map((_, i) => (
        <div
          key={i}
          className="cnn-unit-row"
          style={{
            width: d,
            height: d,
            marginBottom: i < shown.length - 1 ? gap : 0,
          }}
        >
          <span
            className="cnn-unit-sq"
            style={{ width: d, height: d, background: fills[i] }}
          />
          <Handle type="source" position={Position.Right} id={`s-${i}`} className="cnn-handle" />
        </div>
      ))}
    </div>
  );
}

export function CnnFlattenNode({ data }: NodeProps<Node<CnnNodeData>>) {
  const isGap = data.kind === "gap2d" || data.kind === "gap1d";
  return (
    <BaseCnnNode
      data={data}
      className={`cnn-node--flatten${isGap ? " cnn-node--gap" : ""}`}
      hideSource
    >
      <FlattenVector layerId={data.layerId} length={data.length} />
    </BaseCnnNode>
  );
}

/**
 * Dense layer as a column of neurons (NN-style). Weights are drawn as Bezier
 * edges into/out of these units rather than a weight-circle matrix.
 */
function DenseNeuronColumn({ layerId, units }: { layerId: string; units: number }) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  void paintGeneration;
  const t = useCnnMessages();

  const acts = snapshot?.signals?.[0] ?? [];
  const biases = snapshot?.biases ?? [];
  const n = Math.max(1, acts.length || units);
  const stack = unitStackSize(n, 1);
  const shownActs = acts.length
    ? downsample1D(acts, stack.visCount)
    : new Array(stack.visCount).fill(0);
  const shownBias = biases.length
    ? downsample1D(biases, stack.visCount)
    : new Array(stack.visCount).fill(0);
  const { d, gap, width, height } = stack;

  if (!snapshot?.matrix?.length && !acts.length) {
    return <div className="cnn-feature-empty" title={t.denseWeightsEmpty} />;
  }

  const cell = Math.max(d, 28);
  const stackH = shownActs.length * cell + gap * Math.max(0, shownActs.length - 1);

  return (
    <div
      className="cnn-unit-stack cnn-unit-stack--dense"
      style={{ width: cell, height: stackH }}
      title={`${n} units`}
    >
      {shownActs.map((a, i) => {
        const bias = shownBias[i] ?? 0;
        const fill = weightColor(weightValueNormalized(a));
        return (
          <div
            key={i}
            className="cnn-unit-row cnn-unit-row--neuron"
            style={{
              width: cell,
              height: cell,
              marginBottom: i < shownActs.length - 1 ? gap : 0,
            }}
          >
            <Handle type="target" position={Position.Left} id={`t-${i}`} className="cnn-handle" />
            <span
              className="cnn-unit-neuron"
              style={{ width: cell, height: cell, background: fill }}
              title={`unit ${i + 1}${biases.length ? ` · bias ${bias.toFixed(3)}` : ""}`}
            >
              {biases.length > 0 && (
                <span
                  className="cnn-filter-bias"
                  data-sign={bias >= 0 ? "pos" : "neg"}
                  aria-hidden
                  style={{ background: weightColor(weightValueNormalized(bias)) }}
                />
              )}
            </span>
            <Handle type="source" position={Position.Right} id={`s-${i}`} className="cnn-handle" />
          </div>
        );
      })}
    </div>
  );
}

export function CnnDenseNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--dense" hideSource hideTarget>
      <DenseNeuronColumn layerId={data.layerId} units={data.length || 1} />
    </BaseCnnNode>
  );
}

export function CnnReadoutNode({ id, data }: NodeProps<Node<CnnNodeData>>) {
  const t = useCnnMessages();
  const paintGeneration = useContext(PaintGenerationContext);
  const playVizRef = useContext(CnnPlayVizRefContext);
  const p1FillRef = useRef<HTMLDivElement>(null);
  const p0FillRef = useRef<HTMLDivElement>(null);
  const p1ValRef = useRef<HTMLSpanElement>(null);
  const p0ValRef = useRef<HTMLSpanElement>(null);

  const paintProbs = useCallback(() => {
    const live = playVizRef?.current?.probability;
    const p1 = Math.min(1, Math.max(0, live ?? data.probability ?? 0.5));
    const p0 = 1 - p1;
    if (p1FillRef.current) p1FillRef.current.style.height = `${Math.max(2, p1 * 100)}%`;
    if (p0FillRef.current) p0FillRef.current.style.height = `${Math.max(2, p0 * 100)}%`;
    if (p1ValRef.current) p1ValRef.current.textContent = p1.toFixed(2);
    if (p0ValRef.current) p0ValRef.current.textContent = p0.toFixed(2);
  }, [data.probability, playVizRef]);

  useLayoutEffect(() => {
    paintProbs();
  }, [paintProbs, paintGeneration]);

  const p1 = Math.min(1, Math.max(0, data.probability ?? 0.5));
  const p0 = 1 - p1;

  return (
    <BaseCnnNode data={data} className="cnn-node--readout" hideSource>
      <div className="cnn-readout-body" title={t.readoutProb}>
        <div className="cnn-readout-cols">
          <div className="cnn-readout-col">
            <div className="cnn-readout-col__track">
              <div
                ref={p0FillRef}
                className="cnn-readout-col__fill"
                style={{ height: `${Math.max(2, p0 * 100)}%`, background: CLASS_0_HEX }}
              />
            </div>
            <span ref={p0ValRef} className="cnn-readout-col__val">
              {p0.toFixed(2)}
            </span>
            <span className="cnn-readout-col__label">{t.class0}</span>
          </div>
          <div className="cnn-readout-col">
            <div className="cnn-readout-col__track">
              <div
                ref={p1FillRef}
                className="cnn-readout-col__fill"
                style={{ height: `${Math.max(2, p1 * 100)}%`, background: CLASS_1_HEX }}
              />
            </div>
            <span ref={p1ValRef} className="cnn-readout-col__val">
              {p1.toFixed(2)}
            </span>
            <span className="cnn-readout-col__label">{t.class1}</span>
          </div>
        </div>
      </div>
      <span className="cnn-node-label cnn-node-label--hidden" aria-hidden>
        {id}
      </span>
    </BaseCnnNode>
  );
}

export const cnnNodeTypes = {
  cnnInput: CnnInputNode,
  cnnConv: CnnConvNode,
  cnnPool: CnnPoolNode,
  cnnFlatten: CnnFlattenNode,
  cnnDense: CnnDenseNode,
  cnnReadout: CnnReadoutNode,
};

export const CnnWeightEdge = function CnnWeightEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
  style,
}: EdgeProps<Edge<CnnEdgeData>>) {
  void id;
  // Per-weight dense/output edges use Bezier (NN-style); layer hops stay straight.
  const curved = typeof data?.weight === "number";
  const [path] = curved
    ? getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition })
    : getStraightPath({ sourceX, sourceY, targetX, targetY });
  return (
    <BaseEdge
      path={path}
      style={{
        ...style,
        strokeWidth: selected ? Math.max(3.5, (style?.strokeWidth as number) ?? 3) : style?.strokeWidth,
        strokeOpacity: selected ? 1 : style?.strokeOpacity,
      }}
    />
  );
};

export const cnnEdgeTypes = {
  cnnWeight: CnnWeightEdge,
};
