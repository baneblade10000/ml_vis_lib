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
import { curveStrokeFromValues, inferYDomain, renderCurve, renderCurvePoints, renderTargetCurve } from "@ml-vis/core/charts";
import { NODE_BOUNDARY_DENSITY, X_DOMAIN } from "@ml-vis/core/network";
import type { DataPoint, NetworkNodeData, WeightEdgeData } from "./graphAdapter";
import { paintHeatmapCanvas, paintTrainOverlay, BiasIndicator } from "./networkCanvasPaint";
import { NODE_WIDTH, OUTPUT_NODE_WIDTH } from "./graphAdapter";
import {
  BoundaryPaintGenerationContext,
  NetworkBoundaryRefContext,
  NetworkCurveRefContext,
  NetworkTargetCurveRefContext,
  NetworkVizModeContext,
  TrainingLiveRefContext,
} from "./NetworkBoundaryContext";
import { registerBoundaryPainter } from "./boundaryPaint";
import { WeightMatrixFlowNode } from "./NetworkWeightMatrixNode";






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
  const vizMode = useContext(NetworkVizModeContext);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const px = size - 6;
  const regression = vizMode.problemType === "regression";

  const paintOverlay = useCallback(() => {
    if (!trainData?.length || !overlayRef.current) return;
    paintTrainOverlay(overlayRef.current, px, trainData, regression);
  }, [trainData, px, regression]);

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
    <div className="nn-flow-node-canvas-wrap" style={{ width: px, height: px }}>
      <canvas ref={heatmapRef} width={1} height={1} hidden aria-hidden />
      <canvas
        ref={canvasRef}
        className={dimmed ? "nn-flow-node-canvas dimmed" : "nn-flow-node-canvas"}
      />
      {trainData && (
        <canvas
          ref={overlayRef}
          className="nn-flow-node-canvas nn-flow-node-canvas--overlay"
          aria-hidden
        />
      )}
    </div>
  );
}

function NodeCurve({
  nodeId,
  size,
  dimmed,
  trainData,
  showTarget,
}: {
  nodeId: string;
  size: number;
  dimmed?: boolean;
  trainData?: DataPoint[];
  showTarget?: boolean;
  paintGeneration?: number;
}) {
  const curvesRef = useContext(NetworkCurveRefContext);
  const targetRef = useContext(NetworkTargetCurveRefContext);
  const vizMode = useContext(NetworkVizModeContext);
  const paintGeneration = useContext(BoundaryPaintGenerationContext);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const px = size - 6;

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const values = curvesRef?.current?.[nodeId];
    if (!canvas || !values?.length) return;

    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(px * dpr);
    const h = w;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${px}px`;
      canvas.style.height = `${px}px`;
    }

    const yDomainParts = [inferYDomain(values)];
    if (showTarget && targetRef?.current?.length) {
      yDomainParts.push(inferYDomain(targetRef.current));
    }
    if (trainData?.length) {
      yDomainParts.push(inferYDomain(trainData.map((p) => p.y)));
    }
    const yMin = Math.min(...yDomainParts.map((d) => d[0]));
    const yMax = Math.max(...yDomainParts.map((d) => d[1]));
    const yDomain: [number, number] = [yMin, yMax];

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    // Baseline first, then points (under), then prediction curve on top.
    const zeroY = ((1 - (0 - yMin) / (yMax - yMin)) * h);
    ctx.strokeStyle = "rgba(0, 0, 0, 0.12)";
    ctx.lineWidth = Math.max(1, dpr * 0.75);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();

    if (trainData?.length) {
      renderCurvePoints(canvas, trainData, {
        xDomain: X_DOMAIN,
        yDomain,
        colorByLabel: vizMode.problemType === "classification",
      });
    }

    if (showTarget && targetRef?.current?.length) {
      renderTargetCurve(canvas, targetRef.current, { yDomain });
    }

    renderCurve(canvas, values, {
      yDomain,
      stroke: curveStrokeFromValues(values),
      fill: !dimmed,
      clear: false,
      baseline: false,
    });
  }, [curvesRef, targetRef, nodeId, px, dimmed, trainData, showTarget, vizMode.problemType]);

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
    <div className="nn-flow-node-canvas-wrap" style={{ width: px, height: px }}>
      <canvas
        ref={canvasRef}
        className={dimmed ? "nn-flow-node-canvas dimmed" : "nn-flow-node-canvas"}
      />
    </div>
  );
}

function NodeViz({
  nodeId,
  discretize,
  size,
  dimmed,
  trainData,
  smooth = true,
  coarseTo,
  showTarget,
}: {
  nodeId: string;
  discretize: boolean;
  size: number;
  dimmed?: boolean;
  trainData?: DataPoint[];
  smooth?: boolean;
  coarseTo?: number;
  showTarget?: boolean;
  paintGeneration?: number;
}) {
  const { dataMode } = useContext(NetworkVizModeContext);
  if (dataMode === "1d") {
    return (
      <NodeCurve
        nodeId={nodeId}
        size={size}
        dimmed={dimmed}
        trainData={trainData}
        showTarget={showTarget}
      />
    );
  }
  return (
    <NodeHeatmap
      nodeId={nodeId}
      discretize={discretize}
      size={size}
      dimmed={dimmed}
      trainData={trainData}
      smooth={smooth}
      coarseTo={coarseTo}
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
      className={`nn-flow-node ${className}${data.selected ? " selected" : ""}`}
      style={{ width: size, height: size }}
    >
      {!hideTarget && <Handle type="target" position={Position.Left} className="nn-flow-handle" />}
      {children}
      {typeof data.bias === "number" && <BiasIndicator bias={data.bias} />}
      {!hideSource && <Handle type="source" position={Position.Right} className="nn-flow-handle" />}
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
      className={`nn-flow-node--feature${data.active === false ? " inactive" : " active"}`}
      hideTarget
    >
      <NodeViz
        nodeId={id}
        discretize={data.discretize}
        size={NODE_WIDTH}
        dimmed={!data.active}
        smooth={false}
        coarseTo={NODE_BOUNDARY_DENSITY}
        paintGeneration={data.paintGeneration}
      />
      <span className="nn-flow-node-label nn-flow-node-label--left">{data.label}</span>
    </BaseNetworkNode>
  );
}

export function DenseFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  return (
    <BaseNetworkNode data={data} className="nn-flow-node--dense">
      <NodeViz
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
    <BaseNetworkNode data={data} className="nn-flow-node--sum">
      <div className="nn-flow-sum-icon">+</div>
      <NodeViz
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

const OUTPUT_AXIS_TICKS = [-6, 0, 6] as const;

function OutputAxisTicks({
  axis,
  ticks = OUTPUT_AXIS_TICKS,
}: {
  axis: "x" | "y";
  ticks?: readonly number[];
}) {
  // Y is screen-flipped (top = +6), so reverse the label order for the column.
  const labels = axis === "y" ? [...ticks].reverse() : ticks;
  return (
    <div className={`nn-flow-axis nn-flow-axis--${axis}`} aria-hidden>
      {labels.map((t) => (
        <span key={`${axis}-${t}`}>{t > 0 ? t : t === 0 ? "0" : `−${Math.abs(t)}`}</span>
      ))}
    </div>
  );
}

export function OutputFlowNode({
  id,
  data,
}: NodeProps<Node<NetworkNodeData>>) {
  const { dataMode } = useContext(NetworkVizModeContext);
  const showAxes = dataMode === "2d";

  return (
    <BaseNetworkNode
      data={data}
      className="nn-flow-node--output"
      hideSource
      size={OUTPUT_NODE_WIDTH}
    >
      {showAxes && <OutputAxisTicks axis="y" />}
      <NodeViz
        nodeId={id}
        discretize={data.discretize}
        size={OUTPUT_NODE_WIDTH}
        trainData={data.trainData}
        showTarget
        paintGeneration={data.paintGeneration}
      />
      {showAxes && <OutputAxisTicks axis="x" />}
    </BaseNetworkNode>
  );
}

export const networkNodeTypes = {
  feature: FeatureFlowNode,
  dense: DenseFlowNode,
  sum: SumFlowNode,
  readout: OutputFlowNode,
  weightMatrix: WeightMatrixFlowNode,
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

  const vizMode = data?.vizMode ?? "weight";
  const value = vizMode === "gradient" ? data?.gradient : data?.weight;
  const showLabel = (hovered || selected) && typeof value === "number";
  const digits = vizMode === "gradient" ? 3 : 2;
  const signed = (n: number, places: number) => `${n >= 0 ? "+" : ""}${n.toFixed(places)}`;
  // SGD step implied by the shown batch-mean gradient (reg term omitted).
  const deltaW =
    vizMode === "gradient" && typeof data?.gradient === "number"
      ? -(data.learningRate ?? 0) * data.gradient
      : null;

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
            className="nn-edge-label"
            data-selected={selected ? "" : undefined}
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              borderColor: style?.stroke as string | undefined,
              color: style?.stroke as string | undefined,
            }}
          >
            {vizMode === "gradient" ? (
              <>
                ∂ {signed(value!, digits)}
                {deltaW !== null && (
                  <span className="nn-edge-label__delta"> → Δw {signed(deltaW, 3)}</span>
                )}
              </>
            ) : (
              signed(value!, digits)
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export const networkEdgeTypes = {
  weight: WeightFlowEdge,
};
