import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type OnConnect,
  type OnEdgesChange,
  type OnNodesChange,
  type OnNodeDrag,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ComputationalGraph, GraphNodeKind, TfDataMode, TfProblemType } from "@ml-vis/core";
import {
  flowKindFromDrag,
  graphToFlow,
  PALETTE_DRAG_TYPE,
  type EdgeVizMode,
  type NetworkNodeData,
  type WeightEdgeData,
} from "./graphAdapter";
import { networkEdgeTypes, networkNodeTypes } from "./NetworkFlowNodes";
import {
  BoundaryPaintGenerationContext,
  NetworkBoundaryRefContext,
  NetworkCurveRefContext,
  NetworkTargetCurveRefContext,
  NetworkVizModeContext,
  TrainingLiveRefContext,
  TrainingStatsRefContext,
  type CurveStore,
} from "./NetworkBoundaryContext";

export interface ReactFlowNetworkGraphProps {
  graph: ComputationalGraph;
  enabledFeatures: Record<string, boolean>;
  discretize?: boolean;
  dataMode?: TfDataMode;
  problemType?: TfProblemType;
  trainData?: { x: number; y: number; label: number }[];
  lossTest?: number;
  lossTrain?: number;
  height?: number;
  fillHeight?: boolean;
  children?: ReactNode;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onSelectEdge: (edgeId: string | null) => void;
  onToggleFeature: (featureId: string) => void;
  onConnect: (sourceId: string, targetId: string) => void;
  onDropNode?: (kind: GraphNodeKind, position: { x: number; y: number }) => void;
  onMoveNode: (nodeId: string, position: { x: number; y: number }) => void;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (sourceId: string, targetId: string) => void;
  fitViewKey?: string;
  refitViewKey?: string | number;
  layoutKey?: number;
  trainingLive?: boolean;
  trainingLiveRef?: RefObject<boolean>;
  paintGeneration?: number;
  edgeVizMode?: EdgeVizMode;
  learningRate?: number;
  boundaryRef: RefObject<Record<string, number[][]>>;
  curvesRef?: RefObject<CurveStore>;
  targetCurveRef?: RefObject<number[] | null>;
  statsRef: RefObject<{ epoch: number; lossTrain: number; lossTest: number }>;
}

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 1.75;
const FIT_MAX_ZOOM = 1.1;
/** Extra world-space room below nodes for their labels / loss captions. */
const LABEL_MARGIN = 36;

/**
 * Compute the viewport that fits the given nodes, keeping clear of the side
 * docks that overlay the canvas in immersive mode. Computed from node geometry
 * we already know instead of React Flow's fitView, which reads measured DOM
 * sizes and silently mis-fits when called right after nodes were replaced.
 */
function viewportForNodes(
  nodes: Node<NetworkNodeData>[],
  container: { width: number; height: number },
  fillHeight: boolean,
): { x: number; y: number; zoom: number } | null {
  if (!nodes.length || container.width <= 0 || container.height <= 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + (node.width ?? 0));
    maxY = Math.max(maxY, node.position.y + (node.height ?? 0) + LABEL_MARGIN);
  }

  const pad = fillHeight
    ? { top: 24, bottom: 24, left: 290, right: 290 }
    : { top: 32, bottom: 32, left: 48, right: 48 };
  const availWidth = Math.max(container.width - pad.left - pad.right, 120);
  const availHeight = Math.max(container.height - pad.top - pad.bottom, 120);
  const boundsWidth = Math.max(maxX - minX, 1);
  const boundsHeight = Math.max(maxY - minY, 1);

  const zoom = Math.min(
    Math.max(Math.min(availWidth / boundsWidth, availHeight / boundsHeight), MIN_ZOOM),
    FIT_MAX_ZOOM,
  );
  return {
    x: pad.left + (availWidth - boundsWidth * zoom) / 2 - minX * zoom,
    y: pad.top + (availHeight - boundsHeight * zoom) / 2 - minY * zoom,
    zoom,
  };
}

/** Color of an edge marker, or "" when it has none (string markers carry no color). */
function edgeMarkerColor(marker: Edge["markerEnd"] | Edge["markerStart"]): string {
  if (!marker || typeof marker === "string") return "";
  return marker.color ?? "";
}

function ReactFlowNetworkGraphInner({
  graph,
  enabledFeatures,
  discretize = false,
  trainData,
  lossTest,
  lossTrain,
  height = 420,
  fillHeight = false,
  children,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onToggleFeature,
  onConnect,
  onDropNode,
  onMoveNode,
  onRemoveNode,
  onRemoveEdge,
  fitViewKey,
  refitViewKey,
  layoutKey,
  trainingLive = false,
  paintGeneration = 0,
  edgeVizMode = "weight",
  learningRate = 0,
}: ReactFlowNetworkGraphProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();
  const topologyKeyRef = useRef("");
  const fitViewKeyRef = useRef("");
  const refitViewKeyRef = useRef<string | number | undefined>(undefined);
  const fittedSizeRef = useRef({ width: 0, height: 0 });
  const draggingNodesRef = useRef(new Set<string>());
  const { setViewport } = useReactFlow();
  const [animateLayout, setAnimateLayout] = useState(false);
  // Size must be state (not only a ref) so the fit effect re-runs once the
  // canvas has a real layout — first paint often measures 0×0.
  const [measured, setMeasured] = useState({ width: 0, height: 0 });

  // Briefly enable a CSS transition on node positions whenever the canonical
  // grid is re-applied ("Arrange layout"), so nodes glide into place instead of
  // jumping. The transition is disabled during normal drag/updates.
  useEffect(() => {
    if (refitViewKey === undefined) return;
    setAnimateLayout(true);
    const t = window.setTimeout(() => setAnimateLayout(false), 360);
    return () => window.clearTimeout(t);
  }, [refitViewKey]);

  useEffect(() => {
    if (!reactFlowWrapper.current) return;
    const el = reactFlowWrapper.current;
    const sync = () => {
      const next = { width: el.clientWidth, height: el.clientHeight };
      setMeasured((prev) =>
        Math.abs(prev.width - next.width) < 0.5 && Math.abs(prev.height - next.height) < 0.5
          ? prev
          : next,
      );
    };
    sync();
    const ro = new ResizeObserver(() => sync());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const topologyKey = `${graph.inputIds.join(",")};${[...graph.nodes.keys()].sort().join(",")}`;

  const flowOptions = useMemo(
    () => ({
      enabledFeatures,
      discretize,
      selectedNodeId,
      selectedEdgeId,
      trainData,
      lossTest: trainingLive ? undefined : lossTest,
      lossTrain: trainingLive ? undefined : lossTrain,
      paintGeneration,
      edgeVizMode,
      learningRate,
    }),
    trainingLive
      ? [enabledFeatures, discretize, selectedNodeId, selectedEdgeId, trainData, trainingLive, paintGeneration, edgeVizMode, learningRate]
      : [enabledFeatures, discretize, selectedNodeId, selectedEdgeId, trainData, lossTest, lossTrain, trainingLive, paintGeneration, edgeVizMode, learningRate],
  );

  const mapped = useMemo(
    () => graphToFlow(graph, flowOptions),
    [graph, topologyKey, flowOptions, layoutKey],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<NetworkNodeData>>(mapped.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge<WeightEdgeData>>(mapped.edges);

  useEffect(() => {
    const topologyChanged = topologyKeyRef.current !== topologyKey;
    if (topologyChanged) {
      topologyKeyRef.current = topologyKey;
      draggingNodesRef.current.clear();
    }

    setNodes((current) => {
      const mappedIds = new Set(mapped.nodes.map((n) => n.id));
      const currentIds = new Set(current.map((n) => n.id));
      const structuralChange =
        topologyChanged ||
        mapped.nodes.length !== current.length ||
        mapped.nodes.some((n) => !currentIds.has(n.id)) ||
        current.some((n) => !mappedIds.has(n.id));

      if (structuralChange) {
        // Node ids restart from 1 for every architecture, so positions of a
        // previous graph must never be reused — the engine's graph.positions
        // (already baked into `mapped`) are the source of truth.
        return mapped.nodes.map((next) => {
          if (draggingNodesRef.current.has(next.id)) {
            const existing = current.find((n) => n.id === next.id);
            if (existing) return existing;
          }
          return { ...next, selected: next.id === selectedNodeId };
        });
      }

      let changed = false;
      const nextNodes = current.map((node) => {
        const next = mapped.nodes.find((n) => n.id === node.id);
        if (!next) return node;
        const selected = next.id === selectedNodeId;
        if (draggingNodesRef.current.has(node.id)) return node;
        const samePos =
          node.position.x === next.position.x && node.position.y === next.position.y;
        const sameData =
          node.data.discretize === next.data.discretize &&
          node.data.active === next.data.active &&
          node.data.bias === next.data.bias &&
          node.data.lossTest === next.data.lossTest &&
          node.data.lossTrain === next.data.lossTrain &&
          node.data.trainData === next.data.trainData &&
          node.data.paintGeneration === next.data.paintGeneration;
        if (samePos && sameData && node.selected === selected) return node;
        changed = true;
        return { ...node, position: next.position, data: next.data, selected };
      });
      return changed ? nextNodes : current;
    });

    setEdges((current) => {
      if (topologyChanged || current.length !== mapped.edges.length) {
        return mapped.edges;
      }
      let changed = false;
      const nextEdges = current.map((edge) => {
        const next = mapped.edges.find((e) => e.id === edge.id);
        if (!next) return edge;
        const selected = next.id === selectedEdgeId;
        const sameStyle =
          edge.style?.stroke === next.style?.stroke &&
          edge.style?.strokeWidth === next.style?.strokeWidth &&
          edge.style?.strokeOpacity === next.style?.strokeOpacity &&
          edgeMarkerColor(edge.markerEnd) === edgeMarkerColor(next.markerEnd);
        const sameData =
          edge.data?.weight === next.data?.weight &&
          edge.data?.gradient === next.data?.gradient &&
          edge.data?.learningRate === next.data?.learningRate &&
          edge.data?.active === next.data?.active &&
          edge.data?.vizMode === next.data?.vizMode;
        if (sameStyle && sameData && edge.selected === selected) return edge;
        changed = true;
        return { ...edge, data: next.data, style: next.style, markerEnd: next.markerEnd, selected };
      });
      return changed ? nextEdges : current;
    });
  }, [mapped, topologyKey, selectedNodeId, selectedEdgeId, setNodes, setEdges]);

  // Fit/center only after the container has a real size. Consume fit keys only on
  // success so a 0×0 first measure does not permanently skip centering. Also
  // re-fit when the canvas settles to a new size (flex layout often grows after
  // the first non-zero measure).
  useEffect(() => {
    if (measured.width <= 0 || measured.height <= 0 || !mapped.nodes.length) return;

    const fitPending = fitViewKey !== undefined && fitViewKeyRef.current !== fitViewKey;
    const refitPending = refitViewKey !== undefined && refitViewKeyRef.current !== refitViewKey;
    const sizeDrift =
      Math.abs(measured.width - fittedSizeRef.current.width) > 24 ||
      Math.abs(measured.height - fittedSizeRef.current.height) > 24;
    const resizeRefit =
      fittedSizeRef.current.width > 0 &&
      sizeDrift &&
      (fitViewKeyRef.current !== "" || refitViewKeyRef.current !== undefined);

    if (!fitPending && !refitPending && !resizeRefit) return;

    const viewport = viewportForNodes(mapped.nodes, measured, fillHeight);
    if (!viewport) return;

    const hadPriorFit = fitViewKeyRef.current !== "" || refitViewKeyRef.current !== undefined;
    void setViewport(viewport, {
      duration: (refitPending || resizeRefit) && hadPriorFit ? 200 : 0,
    });
    fittedSizeRef.current = measured;
    if (fitPending) fitViewKeyRef.current = fitViewKey!;
    if (refitPending) refitViewKeyRef.current = refitViewKey;
  }, [measured, mapped.nodes, fitViewKey, refitViewKey, fillHeight, setViewport]);

  const handleNodesChange: OnNodesChange<Node<NetworkNodeData>> = useCallback(
    (changes) => {
      onNodesChange(changes);
    },
    [onNodesChange],
  );

  const onNodeDragStart = useCallback<OnNodeDrag<Node<NetworkNodeData>>>((_, node) => {
    draggingNodesRef.current.add(node.id);
  }, []);

  const onNodeDragStop = useCallback<OnNodeDrag<Node<NetworkNodeData>>>(
    (_, node) => {
      draggingNodesRef.current.delete(node.id);
      onMoveNode(node.id, node.position);
    },
    [onMoveNode],
  );

  const handleEdgesChange: OnEdgesChange<Edge<WeightEdgeData>> = useCallback(
    (changes) => {
      onEdgesChange(changes);
    },
    [onEdgesChange],
  );

  const handleConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (connection.source && connection.target) {
        onConnect(connection.source, connection.target);
      }
    },
    [onConnect],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    if (!onDropNode) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, [onDropNode]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      if (!onDropNode) return;
      event.preventDefault();
      const raw =
        event.dataTransfer.getData(PALETTE_DRAG_TYPE) || event.dataTransfer.getData("text/plain");
      const kind = flowKindFromDrag(raw);
      if (!kind) return;
      const position = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      onDropNode(kind, position);
    },
    [onDropNode, rf],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<NetworkNodeData>) => {
      onSelectEdge(null);
      onSelectNode(node.id);
    },
    [onSelectNode, onSelectEdge],
  );

  const onNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node<NetworkNodeData>) => {
      if (node.data.kind === "input") {
        onToggleFeature(node.id);
      }
    },
    [onToggleFeature],
  );

  const onEdgeClick = useCallback(
    (_: React.MouseEvent, edge: Edge<WeightEdgeData>) => {
      onSelectNode(null);
      onSelectEdge(edge.id);
    },
    [onSelectNode, onSelectEdge],
  );

  const onPaneClick = useCallback(() => {
    onSelectNode(null);
    onSelectEdge(null);
  }, [onSelectNode, onSelectEdge]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA") return;

      if (selectedEdgeId) {
        const link = graph.getAllLinks().find((l) => l.id === selectedEdgeId);
        if (link) onRemoveEdge(link.source.id, link.dest.id);
        return;
      }
      if (selectedNodeId) {
        onRemoveNode(selectedNodeId);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedNodeId, selectedEdgeId, graph, onRemoveNode, onRemoveEdge]);

  return (
    <div
      className={`tf-flow-wrap${fillHeight ? " tf-flow-wrap--fill" : ""}`}
      ref={reactFlowWrapper}
      style={fillHeight ? undefined : { height }}
    >
      <div
        className={`tf-flow-canvas${animateLayout ? " tf-flow-canvas--animate" : ""}`}
        style={fillHeight ? undefined : { width: "100%", height }}
      >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={networkNodeTypes}
        edgeTypes={networkEdgeTypes}
        onNodesChange={handleNodesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        onEdgesChange={handleEdgesChange}
        onConnect={handleConnect}
        onDrop={onDropNode ? onDrop : undefined}
        onDragOver={onDropNode ? onDragOver : undefined}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        minZoom={MIN_ZOOM}
        maxZoom={MAX_ZOOM}
        proOptions={{ hideAttribution: true }}
        nodesFocusable={false}
        edgesFocusable={false}
        autoPanOnNodeDrag={false}
        nodesDraggable
        nodesConnectable
        elementsSelectable
      >
        <Background gap={20} size={1} color="var(--tf-border)" />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
      </div>

      {children && <div className="tf-flow-overlays">{children}</div>}
    </div>
  );
}

export function ReactFlowNetworkGraph(props: ReactFlowNetworkGraphProps) {
  const vizMode = {
    dataMode: props.dataMode ?? "2d",
    problemType: props.problemType ?? "classification",
  } as const;
  const emptyCurves = useMemo(() => ({ current: {} as CurveStore }), []);
  const emptyTarget = useMemo(() => ({ current: null as number[] | null }), []);

  return (
    <NetworkVizModeContext.Provider value={vizMode}>
      <NetworkBoundaryRefContext.Provider value={props.boundaryRef}>
        <NetworkCurveRefContext.Provider value={props.curvesRef ?? emptyCurves}>
          <NetworkTargetCurveRefContext.Provider value={props.targetCurveRef ?? emptyTarget}>
            <TrainingStatsRefContext.Provider value={props.statsRef}>
              <TrainingLiveRefContext.Provider value={props.trainingLiveRef ?? null}>
                <BoundaryPaintGenerationContext.Provider value={props.paintGeneration ?? 0}>
                  <ReactFlowProvider>
                    <ReactFlowNetworkGraphInner {...props} />
                  </ReactFlowProvider>
                </BoundaryPaintGenerationContext.Provider>
              </TrainingLiveRefContext.Provider>
            </TrainingStatsRefContext.Provider>
          </NetworkTargetCurveRefContext.Provider>
        </NetworkCurveRefContext.Provider>
      </NetworkBoundaryRefContext.Provider>
    </NetworkVizModeContext.Provider>
  );
}

export { NODE_HEIGHT } from "./graphAdapter";
export { NODE_WIDTH } from "./graphAdapter";
