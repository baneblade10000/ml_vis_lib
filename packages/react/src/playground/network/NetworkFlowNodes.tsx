import { useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
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
  NODE_BOUNDARY_DENSITY,
  PLAY_DISPLAY_DENSITY,
  reduceMatrix,
  renderValueMatrix,
  weightColor,
  weightValueNormalized,
} from "@ml-vis/core";
import type { DataPoint, NetworkNodeData, WeightEdgeData } from "./graphAdapter";
import { NODE_WIDTH, OUTPUT_NODE_WIDTH } from "./graphAdapter";
import {
  BoundaryPaintGenerationContext,
  NetworkBoundaryRefContext,
  TrainingLiveRefContext,
  TrainingStatsRefContext,
} from "./NetworkBoundaryContext";
import { registerBoundaryPainter } from "./boundaryPaint";

function matrixForDisplay(
  matrix: number[][],
  displayPx: number,
  coarseTo?: number,
  live = false,
): number[][] {
  if (live) {
    let factor = Math.max(1, Math.floor(matrix.length / PLAY_DISPLAY_DENSITY));
    while (factor > 1 && matrix.length % factor !== 0) factor -= 1;
    return factor > 1 ? reduceMatrix(matrix, factor) : matrix;
  }
  if (coarseTo !== undefined && matrix.length <= coarseTo) {
    return matrix;
  }
  let grid = matrix;
  if (coarseTo !== undefined && grid.length > coarseTo) {
    let factor = Math.max(1, Math.round(grid.length / coarseTo));
    while (factor > 1 && grid.length % factor !== 0) factor -= 1;
    if (factor > 1) grid = reduceMatrix(grid, factor);
  }
  if (coarseTo !== undefined) {
    return grid;
  }
  const dpr = window.devicePixelRatio || 1;
  const target = Math.ceil(displayPx * dpr);
  let factor = Math.max(1, Math.floor(grid.length / target));
  while (factor > 1 && grid.length % factor !== 0) factor -= 1;
  return factor > 1 ? reduceMatrix(grid, factor) : grid;
}

const OUTPUT_X_DOMAIN: [number, number] = [-6, 6];

function ensureCanvasSize(
  canvas: HTMLCanvasElement,
  px: number,
  dpr: number,
): CanvasRenderingContext2D | null {
  const w = Math.round(px * dpr);
  const h = w;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${px}px`;
    canvas.style.height = `${px}px`;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function paintTrainOverlay(
  canvas: HTMLCanvasElement,
  px: number,
  trainData: DataPoint[],
): void {
  const dpr = window.devicePixelRatio || 1;
  const ctx = ensureCanvasSize(canvas, px, dpr);
  if (!ctx) return;
  ctx.clearRect(0, 0, px, px);
  const [minX, maxX] = OUTPUT_X_DOMAIN;
  const mapX = (x: number) => ((x - minX) / (maxX - minX)) * px;
  const mapY = (y: number) => (1 - (y - minX) / (maxX - minX)) * px;
  for (const point of trainData) {
    ctx.beginPath();
    ctx.arc(mapX(point.x), mapY(point.y), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = point.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 0.75;
    ctx.stroke();
  }
}

function paintHeatmapCanvas(
  canvas: HTMLCanvasElement,
  heatmap: HTMLCanvasElement,
  matrix: number[][],
  size: number,
  discretize: boolean,
  smooth: boolean,
  coarseTo?: number,
  live = false,
): void {
  const px = size - 6;
  const mini = coarseTo !== undefined;
  const reduced = matrixForDisplay(matrix, px, coarseTo, live);
  renderValueMatrix(heatmap, reduced, discretize);

  const dpr = mini ? 1 : window.devicePixelRatio || 1;
  const ctx = ensureCanvasSize(canvas, px, dpr);
  if (!ctx) return;
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? "high" : "low";
  ctx.drawImage(heatmap, 0, 0, px, px);
}

function NodeHeatmap({
  nodeId,
  discretize,
  size,
  dimmed,
  trainData,
  smooth = true,
  coarseTo,
}: {
  nodeId: string;
  discretize: boolean;
  size: number;
  dimmed?: boolean;
  trainData?: DataPoint[];
  smooth?: boolean;
  coarseTo?: number;
  paintGeneration?: number;
}) {
  const boundaryRef = useContext(NetworkBoundaryRefContext);
  const paintGeneration = useContext(BoundaryPaintGenerationContext);
  const trainingLiveRef = useContext(TrainingLiveRefContext);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const px = size - 6;

  const paintOverlay = useCallback(() => {
    if (!trainData?.length || !overlayRef.current) return;
    paintTrainOverlay(overlayRef.current, px, trainData);
  }, [trainData, px]);

  useEffect(() => {
    paintOverlay();
  }, [paintOverlay]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heatmap = heatmapRef.current;
    const matrix = boundaryRef?.current?.[nodeId];
    if (!canvas || !heatmap || !matrix) return;
    const live = trainingLiveRef?.current ?? false;
    paintHeatmapCanvas(
      canvas,
      heatmap,
      matrix,
      size,
      discretize,
      live ? false : smooth,
      coarseTo,
      live,
    );
  }, [boundaryRef, trainingLiveRef, nodeId, size, discretize, smooth, coarseTo]);

  const paintRef = useRef(paint);
  paintRef.current = paint;

  useLayoutEffect(
    () => registerBoundaryPainter(nodeId, () => paintRef.current()),
    [nodeId],
  );
  useLayoutEffect(() => {
    paintRef.current();
  }, [paint, paintGeneration]);

  return (
    <div className="tf-flow-node-canvas-wrap" style={{ width: px, height: px }}>
      <canvas ref={heatmapRef} width={1} height={1} hidden aria-hidden />
      <canvas
        ref={canvasRef}
        className={dimmed ? "tf-flow-node-canvas dimmed" : "tf-flow-node-canvas"}
      />
      {trainData && (
        <canvas
          ref={overlayRef}
          className="tf-flow-node-canvas tf-flow-node-canvas--overlay"
          aria-hidden
        />
      )}
    </div>
  );
}

function BiasIndicator({ bias }: { bias: number }) {
  // The bias square sits in the neuron's bottom-right corner. Its fill color
  // encodes the bias via the diverging palette: violet for negative, magenta
  // for positive, with the hue saturating toward the palette extremes as |bias|
  // grows (tanh-normalized, so small biases stay readable).
  return (
    <span
      className="tf-flow-bias"
      data-sign={bias >= 0 ? "pos" : "neg"}
      aria-hidden
      style={{ background: weightColor(weightValueNormalized(bias)) }}
    />
  );
}

function BaseNetworkNode({
  data,
  children,
  className,
  hideSource,
  hideTarget,
  size = NODE_WIDTH,
}: {
  data: NetworkNodeData;
  children?: React.ReactNode;
  className: string;
  hideSource?: boolean;
  hideTarget?: boolean;
  size?: number;
}) {
  return (
    <div
      className={`tf-flow-node ${className}${data.selected ? " selected" : ""}`}
      style={{ width: size, height: size }}
    >
      {!hideTarget && <Handle type="target" position={Position.Left} className="tf-flow-handle" />}
      {children}
      {typeof data.bias === "number" && <BiasIndicator bias={data.bias} />}
      {!hideSource && <Handle type="source" position={Position.Right} className="tf-flow-handle" />}
    </div>
  );
}

export function FeatureFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  return (
    <BaseNetworkNode
      data={data}
      className={`tf-flow-node--feature${data.active === false ? " inactive" : " active"}`}
      hideTarget
    >
      <NodeHeatmap
        nodeId={id}
        discretize={data.discretize}
        size={NODE_WIDTH}
        dimmed={!data.active}
        smooth={false}
        coarseTo={NODE_BOUNDARY_DENSITY}
        paintGeneration={data.paintGeneration}
      />
      <span className="tf-flow-node-label tf-flow-node-label--left">{data.label}</span>
    </BaseNetworkNode>
  );
}

export function DenseFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  return (
    <BaseNetworkNode data={data} className="tf-flow-node--dense">
      <NodeHeatmap
        nodeId={id}
        discretize={data.discretize}
        size={NODE_WIDTH}
        smooth={false}
        coarseTo={NODE_BOUNDARY_DENSITY}
        paintGeneration={data.paintGeneration}
      />
    </BaseNetworkNode>
  );
}

export function SumFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  return (
    <BaseNetworkNode data={data} className="tf-flow-node--sum">
      <div className="tf-flow-sum-icon">+</div>
      <NodeHeatmap
        nodeId={id}
        discretize={data.discretize}
        size={NODE_WIDTH}
        smooth={false}
        coarseTo={NODE_BOUNDARY_DENSITY}
        paintGeneration={data.paintGeneration}
      />
    </BaseNetworkNode>
  );
}

export function OutputFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  const statsRef = useContext(TrainingStatsRefContext);
  const paintGeneration = useContext(BoundaryPaintGenerationContext);
  const lossLabelRef = useRef<HTMLSpanElement>(null);

  const paintMeta = useCallback(() => {
    const stats = statsRef?.current;
    if (lossLabelRef.current && stats) {
      lossLabelRef.current.textContent = `${stats.lossTest.toFixed(3)} / ${stats.lossTrain.toFixed(3)}`;
    }
  }, [statsRef]);

  useLayoutEffect(() => {
    paintMeta();
  }, [paintMeta, paintGeneration, data.lossTest, data.lossTrain]);

  const trainingLiveRef = useContext(TrainingLiveRefContext);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      if (trainingLiveRef?.current) paintMeta();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [paintMeta, trainingLiveRef]);

  return (
    <BaseNetworkNode
      data={data}
      className="tf-flow-node--output"
      hideSource
      size={OUTPUT_NODE_WIDTH}
    >
      <NodeHeatmap
        nodeId={id}
        discretize={data.discretize}
        size={OUTPUT_NODE_WIDTH}
        trainData={data.trainData}
        paintGeneration={data.paintGeneration}
      />
      <div className="tf-flow-output-meta">
        <span ref={lossLabelRef} className="tf-flow-output-loss">
          {data.lossTest !== undefined && data.lossTrain !== undefined
            ? `${data.lossTest.toFixed(3)} / ${data.lossTrain.toFixed(3)}`
            : null}
        </span>
      </div>
    </BaseNetworkNode>
  );
}

export const networkNodeTypes = {
  feature: FeatureFlowNode,
  dense: DenseFlowNode,
  sum: SumFlowNode,
  readout: OutputFlowNode,
};

export const WeightFlowEdge = function WeightFlowEdge({
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
}: EdgeProps<Edge<WeightEdgeData>>) {
  const [hovered, setHovered] = useState(false);
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const weight = data?.weight;
  const showLabel = (hovered || selected) && typeof weight === "number";

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          ...style,
          strokeWidth: selected ? Math.max(3.5, (style?.strokeWidth as number) ?? 3) : style?.strokeWidth,
          strokeOpacity: selected ? 1 : style?.strokeOpacity,
        }}
      />
      {/* Wide invisible hit area so the edge is easy to hover for the label.
          A raw <path> forwards mouse events (BaseEdge does not). */}
      <path
        d={path}
        fill="none"
        stroke="transparent"
        strokeWidth={14}
        style={{ pointerEvents: "stroke" }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      />
      {showLabel && (
        <EdgeLabelRenderer>
          <div
            className="tf-edge-label"
            data-selected={selected ? "" : undefined}
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              borderColor: style?.stroke as string | undefined,
              color: style?.stroke as string | undefined,
            }}
          >
            {weight! >= 0 ? "+" : ""}
            {weight!.toFixed(2)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export const networkEdgeTypes = {
  weight: WeightFlowEdge,
};
