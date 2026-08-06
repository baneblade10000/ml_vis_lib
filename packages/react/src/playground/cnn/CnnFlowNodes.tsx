import { useCallback, useContext, useLayoutEffect, useRef, type ReactNode } from "react";
import {
  BaseEdge,
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
  valueToRgb,
  type FeatureMapSnapshot,
} from "@ml-vis/core";
import type { CnnNodeData, CnnEdgeData } from "./cnnAdapter";
import {
  FeatureMapRefContext,
  PaintGenerationContext,
  TrainingStatsRefContext,
  TrainingLiveRefContext,
} from "./featureMapContext";
import { registerBoundaryPainter } from "./featureMapPaint";

/** Pixel size of one feature-map tile. */
const MAP_PX = 44;
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
    const dpr = window.devicePixelRatio || 1;
    const w = px;
    if (canvas.width !== w * dpr || canvas.height !== w * dpr) {
      canvas.width = w * dpr;
      canvas.height = w * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${w}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, w, w);
    ctx.drawImage(heat, 0, 0, w, w);
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
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== px * dpr || canvas.height !== h * dpr) {
      canvas.width = px * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${px}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, px, h);
    ctx.drawImage(heat, 0, 0, px, h);
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

/** Unit diameter / gap for Flatten + Dense circle stacks. */
function unitCircleMetrics(count: number, columns = 1) {
  const n = Math.max(1, count);
  const gap = 1.5;
  const gapX = 3;
  const d = Math.max(5.25, Math.min(10.5, (520 / n) * 1.5));
  return {
    d,
    r: d / 2,
    gap,
    gapX,
    width: columns * d + gapX * Math.max(0, columns - 1),
    height: n * d + gap * Math.max(0, n - 1),
  };
}

/** Flattened vector as a vertical stack of squares. */
function VectorColumnCanvas({ values }: { values: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !values.length) return;

    const n = values.length;
    const { d, gap, width, height } = unitCircleMetrics(n);
    const radius = Math.min(1.25, d * 0.25);

    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.ceil(width * dpr) || canvas.height !== Math.ceil(height * dpr)) {
      canvas.width = Math.ceil(width * dpr);
      canvas.height = Math.ceil(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

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

    for (let i = 0; i < n; i++) {
      const t = Math.min(1, Math.max(0, (values[i]! - min) / span));
      const g8 = Math.round(Math.pow(t, 0.85) * 255);
      const y = i * (d + gap);
      ctx.fillStyle = `rgb(${g8},${g8},${g8})`;
      ctx.beginPath();
      ctx.roundRect(0, y, d, d, radius);
      ctx.fill();
    }
  }, [values]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  return (
    <div className="cnn-feature-cell cnn-feature-cell--vector" title={`${values.length} units`}>
      <canvas ref={canvasRef} className="cnn-feature-canvas" />
    </div>
  );
}

/** Grid of feature-map tiles for one layer node. */
function FeatureGrid({ layerId, channels, mode }: { layerId: string; channels: number; mode: "2d" | "1d" }) {
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
  const showCount = Math.min(channels, MAX_TILES);
  const tiles: ReactNode[] = [];

  if (mode === "2d") {
    for (let i = 0; i < showCount; i++) {
      const map = maps2d[i] ?? [];
      tiles.push(<Map2DCanvas key={i} map={map} px={MAP_PX} label={`filter ${i + 1}`} />);
    }
  } else {
    for (let i = 0; i < showCount; i++) {
      const values = signals[i] ?? [];
      tiles.push(<Signal1DCanvas key={i} values={values} px={MAP_PX} />);
    }
  }

  if (tiles.length === 0) {
    return <div className="cnn-feature-empty" ref={groupRef} />;
  }
  return (
    <div className="cnn-feature-grid" ref={groupRef}>
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
      <FeatureGrid layerId={data.layerId} channels={data.channels} mode={data.mode} />
    </BaseCnnNode>
  );
}

export function CnnPoolNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--pool">
      <FeatureGrid layerId={data.layerId} channels={data.channels} mode={data.mode} />
    </BaseCnnNode>
  );
}

function FlattenVector({ layerId, length }: { layerId: string; length?: number }) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  void paintGeneration;
  const values = snapshot?.signals?.[0] ?? [];
  const n = values.length || length || 32;
  return <VectorColumnCanvas values={values.length ? values : new Array(n).fill(0)} />;
}

export function CnnFlattenNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--flatten">
      <FlattenVector layerId={data.layerId} length={data.length} />
    </BaseCnnNode>
  );
}

/**
 * All dense weights W[out][in] as circles, colored like NN edges
 * (violet = negative, magenta = positive).
 */
function DenseWeightMatrix({ layerId, units }: { layerId: string; units: number }) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  void paintGeneration;

  const matrix = snapshot?.matrix;
  const rows = matrix?.length || Math.max(1, units);
  const cols = matrix?.[0]?.length || 0;

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix || rows === 0 || cols === 0) return;

    // One vertical stack of circles per output unit (inputs top→bottom).
    const { d, r, gap, gapX, width, height } = unitCircleMetrics(cols, rows);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.ceil(width * dpr) || canvas.height !== Math.ceil(height * dpr)) {
      canvas.width = Math.ceil(width * dpr);
      canvas.height = Math.ceil(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    for (let out = 0; out < rows; out++) {
      for (let inn = 0; inn < cols; inn++) {
        const w = matrix[out]![inn]!;
        const { r: cr, g: cg, b: cb } = valueToRgb(Math.tanh(w * 2));
        const cx = out * (d + gapX) + r;
        const cy = inn * (d + gap) + r;
        ctx.fillStyle = `rgb(${cr}, ${cg}, ${cb})`;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [matrix, rows, cols]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  if (!matrix || cols === 0) {
    return <div className="cnn-feature-empty" title="Weights initialize on first forward" />;
  }

  return (
    <div className="cnn-dense-weights" title={`W: ${rows}×${cols}`}>
      <canvas ref={canvasRef} className="cnn-feature-canvas" />
      <span className="cnn-dense-weights__meta">{rows}×{cols}</span>
    </div>
  );
}

export function CnnDenseNode({ data }: NodeProps<Node<CnnNodeData>>) {
  return (
    <BaseCnnNode data={data} className="cnn-node--dense">
      <DenseWeightMatrix layerId={data.layerId} units={data.length || 1} />
    </BaseCnnNode>
  );
}

export function CnnReadoutNode({ id, data }: NodeProps<Node<CnnNodeData>>) {
  const statsRef = useContext(TrainingStatsRefContext);
  const lossLabelRef = useRef<HTMLSpanElement>(null);
  const paintGeneration = useContext(PaintGenerationContext);

  const paintMeta = useCallback(() => {
    const stats = statsRef?.current;
    if (lossLabelRef.current && stats) {
      lossLabelRef.current.textContent = `${stats.lossTest.toFixed(3)} / ${stats.lossTrain.toFixed(3)}`;
    }
  }, [statsRef]);

  useLayoutEffect(() => {
    paintMeta();
  }, [paintMeta, paintGeneration, data.loss]);

  const prob = data.probability ?? 0.5;
  const barColor = prob >= 0.5 ? CLASS_1_HEX : CLASS_0_HEX;

  return (
    <BaseCnnNode data={data} className="cnn-node--readout" hideSource>
      <div className="cnn-readout-body">
        <div className="cnn-readout-prob" title="Predicted probability of class 1">
          <div
            className="cnn-readout-bar"
            style={{ width: `${Math.max(2, prob * 100)}%`, background: barColor }}
          />
        </div>
        <div className="cnn-readout-loss">
          <span ref={lossLabelRef} className="cnn-readout-loss-value">
            test / train
          </span>
        </div>
      </div>
      <span className="cnn-node-label cnn-node-label--hidden" aria-hidden>{id}</span>
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
  void data;
  void sourcePosition;
  void targetPosition;
  const [path] = getStraightPath({ sourceX, sourceY, targetX, targetY });
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
