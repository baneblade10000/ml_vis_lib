import { useCallback, useContext, useLayoutEffect, useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
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
import { renderValueMatrix, reduceMatrix } from "@ml-vis/core/charts";
import { CLASS_0_HEX, CLASS_1_HEX, weightColor, weightColorZeroWhite, weightValueNormalized } from "@ml-vis/core/network";
import { type FeatureMapSnapshot } from "@ml-vis/core/cnn";
import {
  CNN_CELL_PX,
  cnnGridPx,
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
  useKernelExpand,
  useReceptiveField,
} from "./featureMapContext";
import { registerBoundaryPainter } from "./featureMapPaint";
import { useCnnMessages } from "./messages";
import { receptiveFieldCascade, type RfRect } from "./receptiveField";
import { drawRfOverlay, KernelMini, Signal1DCanvas, activationGray, flatSignalValues } from "./cnnCanvasWidgets";

/** Cap on tiles shown per node (extra channels are dropped to fit). */
const MAX_TILES = 16;

/** Read the per-node feature-map snapshot from the ref context. */
function useLayerSnapshot(layerId: string): FeatureMapSnapshot | undefined {
  const ref = useContext(FeatureMapRefContext);
  return ref?.current?.[layerId];
}


/** Render a single 2-D map — each cell is {@link CNN_CELL_PX} (same as kernels). */
function Map2DCanvas({
  map,
  label,
  layerId,
  channel = 0,
  interactive = false,
  showRf = false,
}: {
  map: number[][];
  label?: string;
  layerId: string;
  channel?: number;
  /** Click/hover picks a pixel and projects RF onto previous layers. */
  interactive?: boolean;
  /** Draw the cascaded RF rect for this layer when present in the selection. */
  showRf?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const rf = useReceptiveField();

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !map.length) return;
    let grid = map;
    let factor = 1;
    if (map.length > 64) {
      factor = Math.max(1, Math.round(map.length / 64));
      while (factor > 1 && map.length % factor !== 0) factor -= 1;
      if (factor > 1) grid = reduceMatrix(map, factor);
    }
    renderValueMatrix(heat, grid, { layout: "row-major", palette: "gray" });
    const rows = grid.length;
    const cols = grid[0]?.length ?? rows;
    const { w: bw, h: bh } = cnnGridPx(rows, cols);
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${bw}px`;
    canvas.style.height = `${bh}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(heat, 0, 0, bw, bh);

    const overlay = overlayRef.current;
    if (overlay) {
      if (overlay.width !== bw || overlay.height !== bh) {
        overlay.width = bw;
        overlay.height = bh;
      }
      const octx = overlay.getContext("2d");
      if (octx) {
        octx.imageSmoothingEnabled = false;
        let rect: RfRect | null = null;
        let kind: "field" | "pixel" = "field";
        const sel = rf?.selection;
        const layerRect = showRf ? sel?.byLayer[layerId] : undefined;
        const channelOk =
          layerRect != null &&
          (layerRect.channel == null || layerRect.channel === channel);
        if (layerRect && channelOk) {
          rect = {
            y0: Math.floor(layerRect.y0 / factor),
            y1: Math.floor(layerRect.y1 / factor),
            x0: Math.floor(layerRect.x0 / factor),
            x1: Math.floor(layerRect.x1 / factor),
          };
          kind = "field";
        } else if (
          sel &&
          sel.sourceLayerId === layerId &&
          sel.channel === channel
        ) {
          rect = {
            y0: Math.floor(sel.outY / factor),
            y1: Math.floor(sel.outY / factor),
            x0: Math.floor(sel.outX / factor),
            x1: Math.floor(sel.outX / factor),
          };
          kind = "pixel";
        }
        drawRfOverlay(octx, bw, bh, CNN_CELL_PX, rect, kind);
      }
    }
  }, [map, rf?.selection, showRf, layerId, channel]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  const rows = map.length || 1;
  const cols = map[0]?.length ?? rows;
  const { w, h } = cnnGridPx(rows, cols);

  const cellFromEvent = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !map.length) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    let factor = 1;
    if (map.length > 64) {
      factor = Math.max(1, Math.round(map.length / 64));
      while (factor > 1 && map.length % factor !== 0) factor -= 1;
    }
    const dispRows = Math.max(1, Math.floor(map.length / factor));
    const dispCols = Math.max(1, Math.floor((map[0]?.length ?? map.length) / factor));
    const cx = Math.min(dispCols - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * dispCols)));
    const cy = Math.min(dispRows - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * dispRows)));
    const outX = Math.min(cols - 1, Math.floor((cx + 0.5) * factor));
    const outY = Math.min(rows - 1, Math.floor((cy + 0.5) * factor));
    return { outY, outX };
  };

  const applyPick = (e: ReactPointerEvent<HTMLCanvasElement>, pin: boolean) => {
    if (!interactive || !rf) return;
    const cell = cellFromEvent(e);
    if (!cell) return;
    const projected = receptiveFieldCascade(
      rf.layers,
      layerId,
      cell.outY,
      cell.outX,
      channel,
    );
    if (!projected) return;
    const next = {
      sourceLayerId: layerId,
      channel,
      outY: cell.outY,
      outX: cell.outX,
      ...projected,
    };
    if (pin) rf.pin(next);
    else if (!rf.pinned) rf.setHover(next);
  };

  return (
    <div
      className={`cnn-feature-cell${interactive ? " cnn-feature-cell--pick nodrag nopan" : ""}`}
      title={label}
      style={{ width: w, height: h }}
    >
      <canvas ref={heatRef} width={1} height={1} hidden aria-hidden />
      <canvas
        ref={canvasRef}
        className="cnn-feature-canvas"
        width={w}
        height={h}
        onPointerMove={interactive ? (e) => applyPick(e, false) : undefined}
        onPointerLeave={
          interactive
            ? () => {
                if (rf && !rf.pinned) rf.setHover(null);
              }
            : undefined
        }
        onPointerDown={
          interactive
            ? (e) => {
                e.stopPropagation();
                applyPick(e, true);
              }
            : undefined
        }
      />
      <canvas
        ref={overlayRef}
        className="cnn-feature-canvas cnn-feature-canvas--overlay"
        width={w}
        height={h}
        aria-hidden
      />
    </div>
  );
}






/** Grid of feature-map tiles for one layer node. */
function FeatureGrid({
  layerId,
  channels,
  mode,
  showKernels = false,
  vertical = false,
  interactiveMaps = false,
  showRf = false,
}: {
  layerId: string;
  channels: number;
  mode: "2d" | "1d";
  /** When true (conv layers), draw each filter kernel beside its channel output. */
  showKernels?: boolean;
  /** Stack channel tiles in a column (pool layers). */
  vertical?: boolean;
  /** Feature-map pixels project a receptive field onto previous layers. */
  interactiveMaps?: boolean;
  /** Draw cascaded RF overlay on this layer's maps. */
  showRf?: boolean;
}) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  const trainingLiveRef = useContext(TrainingLiveRefContext);
  const expand = useKernelExpand();
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
  const kernels2dIn = snapshot?.kernels2dIn ?? [];
  const kernels1dIn = snapshot?.kernels1dIn ?? [];
  const biases = snapshot?.biases ?? [];
  const showCount = Math.min(channels, MAX_TILES);
  const tiles: ReactNode[] = [];

  const side2d =
    expand?.selection?.prevLayerId === layerId ? expand.selection.perIn2d : undefined;
  const side1d =
    expand?.selection?.prevLayerId === layerId ? expand.selection.perIn1d : undefined;
  const selectedFilter =
    expand?.selection?.layerId === layerId ? expand.selection.filter : -1;

  if (mode === "2d") {
    for (let i = 0; i < showCount; i++) {
      const map = maps2d[i] ?? [];
      const mapTile = (
        <Map2DCanvas
          map={map}
          label={`filter ${i + 1}`}
          layerId={layerId}
          channel={i}
          interactive={interactiveMaps}
          showRf={showRf}
        />
      );
      const side = side2d?.[i];
      const sideTile = side?.length ? (
        <KernelMini map={side} label={`→ in ${i + 1}`} />
      ) : null;

      if (showKernels) {
        const kernel = kernels2d[i] ?? [];
        tiles.push(
          <div key={i} className="cnn-channel-pair" title={`kernel → channel ${i + 1}`}>
            <KernelMini
              map={kernel}
              label={`kernel ${i + 1}`}
              bias={biases[i]}
              selected={selectedFilter === i}
              interactive
              onSelect={() => {
                expand?.selectLayer(layerId);
                if (!expand) return;
                const perIn = kernels2dIn[i];
                if ((perIn?.length ?? 0) <= 1) return;
                const prevLayerId = expand.prevLayerId(layerId);
                if (!prevLayerId) return;
                expand.toggle({
                  layerId,
                  filter: i,
                  prevLayerId,
                  perIn2d: perIn,
                });
              }}
            />
            {mapTile}
            {sideTile}
          </div>,
        );
      } else if (sideTile) {
        tiles.push(
          <div key={i} className="cnn-channel-pair" title={`in-channel kernel ${i + 1}`}>
            {mapTile}
            {sideTile}
          </div>,
        );
      } else {
        tiles.push(<div key={i}>{mapTile}</div>);
      }
    }
  } else {
    for (let i = 0; i < showCount; i++) {
      const values = signals[i] ?? [];
      const signalTile = <Signal1DCanvas values={values} />;
      const sideRow = side1d?.[i];
      const sideTile = sideRow?.length ? (
        <KernelMini map={[sideRow]} label={`→ in ${i + 1}`} />
      ) : null;

      if (showKernels) {
        const kernel = kernels1d[i] ?? [];
        tiles.push(
          <div key={i} className="cnn-channel-pair cnn-channel-pair--1d" title={`kernel → channel ${i + 1}`}>
            <KernelMini
              map={kernel.length ? [kernel] : []}
              label={`kernel ${i + 1}`}
              bias={biases[i]}
              selected={selectedFilter === i}
              interactive
              onSelect={() => {
                expand?.selectLayer(layerId);
                if (!expand) return;
                const perIn = kernels1dIn[i];
                if ((perIn?.length ?? 0) <= 1) return;
                const prevLayerId = expand.prevLayerId(layerId);
                if (!prevLayerId) return;
                expand.toggle({
                  layerId,
                  filter: i,
                  prevLayerId,
                  perIn1d: perIn,
                });
              }}
            />
            {signalTile}
            {sideTile}
          </div>,
        );
      } else if (sideTile) {
        tiles.push(
          <div key={i} className="cnn-channel-pair cnn-channel-pair--1d">
            {signalTile}
            {sideTile}
          </div>,
        );
      } else {
        tiles.push(<div key={i}>{signalTile}</div>);
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
      <FeatureGrid
        layerId={data.layerId}
        channels={data.channels}
        mode={data.mode}
        showRf
      />
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
        interactiveMaps={data.mode === "2d"}
        showRf
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
        interactiveMaps={data.mode === "2d"}
        showRf
      />
    </BaseCnnNode>
  );
}


/** Flatten column: grayscale unit squares + per-unit source handles for dense edges. */
function FlattenVector({ layerId, length }: { layerId: string; length?: number }) {
  const snapshot = useLayerSnapshot(layerId);
  const paintGeneration = useContext(PaintGenerationContext);
  void paintGeneration;
  const raw = flatSignalValues(snapshot?.signals, length);
  const n = raw.length || length || 32;
  const values = raw.length ? raw : new Array(n).fill(0);
  const stack = unitStackSize(values.length, 1);
  const shown = downsample1D(values, stack.visCount);
  const fills = activationGray(shown);
  const { d, gap, width, height } = stack;

  return (
    <div
      className="cnn-unit-stack cnn-unit-stack--flatten"
      style={{ width, height }}
      title={`${values.length} units`}
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
  const { d, gap } = stack;

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
                  style={{ background: weightColorZeroWhite(weightValueNormalized(bias)) }}
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

  // Height is set only via DOM — never put it in React `style`, or every
  // paintGeneration re-render resets bars to the JSX default (visible flip).
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

  return (
    <BaseCnnNode data={data} className="cnn-node--readout" hideSource>
      <div className="cnn-readout-body" title={t.readoutProb}>
        <div className="cnn-readout-cols">
          <div className="cnn-readout-col">
            <div className="cnn-readout-col__track">
              <div
                ref={p0FillRef}
                className="cnn-readout-col__fill"
                style={{ background: CLASS_0_HEX }}
              />
            </div>
            <span ref={p0ValRef} className="cnn-readout-col__val">
              —
            </span>
            <span className="cnn-readout-col__label">{t.class0}</span>
          </div>
          <div className="cnn-readout-col">
            <div className="cnn-readout-col__track">
              <div
                ref={p1FillRef}
                className="cnn-readout-col__fill"
                style={{ background: CLASS_1_HEX }}
              />
            </div>
            <span ref={p1ValRef} className="cnn-readout-col__val">
              —
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
