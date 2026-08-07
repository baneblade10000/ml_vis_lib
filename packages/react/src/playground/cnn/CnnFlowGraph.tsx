import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { FeatureMapSnapshot } from "@ml-vis/core";
import {
  cnnPipelineToFlow,
  formatCnnNodeLabel,
  type CnnNodeData,
  type CnnPipelineView,
} from "./cnnAdapter";
import { cnnEdgeTypes, cnnNodeTypes } from "./CnnFlowNodes";
import {
  FeatureMapRefContext,
  PaintGenerationContext,
  TrainingLiveRefContext,
  TrainingStatsRefContext,
  type CnnTrainingStats,
  type FeatureMapStore,
} from "./featureMapContext";
import { useCnnMessages } from "./messages";

export interface CnnFlowGraphProps {
  pipeline: CnnPipelineView;
  selectedNodeId: string | null;
  paintGeneration: number;
  featureMaps: FeatureMapSnapshot[];
  onSelectNode: (nodeId: string | null) => void;
  featureMapRef: RefObject<FeatureMapStore>;
  statsRef: RefObject<CnnTrainingStats>;
  trainingLiveRef: RefObject<boolean>;
  loss?: number;
  probability?: number;
  height?: number;
  fillHeight?: boolean;
  children?: ReactNode;
  refitViewKey?: number;
}

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 1.5;
const FIT_MAX_ZOOM = 1.05;
const LABEL_MARGIN = 20;

/**
 * Viewport that fits the pipeline between immersive side docks.
 * Uses known node geometry (same approach as the MLP graph).
 */
function viewportForNodes(
  nodes: Node<CnnNodeData>[],
  container: { width: number; height: number },
  fillHeight: boolean,
): { x: number; y: number; zoom: number } | null {
  if (!nodes.length || container.width <= 0 || container.height <= 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    const w = node.width ?? node.measured?.width ?? 0;
    const h = node.height ?? node.measured?.height ?? 0;
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + w);
    maxY = Math.max(maxY, node.position.y + h + LABEL_MARGIN);
  }

  const pad = fillHeight
    ? { top: 28, bottom: 36, left: 352, right: 280 }
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

/** Real content box height in flow coordinates (ignores zoom). */
function contentHeight(nodeId: string): number | null {
  const el = document.querySelector(
    `.react-flow__node[data-id="${CSS.escape(nodeId)}"] .cnn-node`,
  );
  if (!(el instanceof HTMLElement)) return null;
  const h = el.offsetHeight;
  return h > 0 ? h : null;
}

/**
 * Shift every node so the vertical center of its .cnn-node sits on one midline.
 * Uses DOM heights — RF `measured` / estimates are often too short for conv stacks.
 */
function alignNodesToMidline(nodes: Node<CnnNodeData>[]): Node<CnnNodeData>[] {
  if (!nodes.length) return nodes;
  const heights = nodes.map(
    (n) => contentHeight(n.id) ?? n.measured?.height ?? n.height ?? 0,
  );
  if (heights.every((h) => h < 1)) return nodes;
  const maxH = Math.max(...heights);
  let changed = false;
  const next = nodes.map((n, i) => {
    const h = heights[i]!;
    const y = (maxH - h) / 2;
    const heightChanged = n.height !== h;
    const yChanged = Math.abs(n.position.y - y) >= 0.5;
    if (!heightChanged && !yChanged) return n;
    changed = true;
    return {
      ...n,
      height: h,
      position: { x: n.position.x, y },
    };
  });
  return changed ? next : nodes;
}

function CnnFlowGraphInner(props: CnnFlowGraphProps) {
  const {
    pipeline,
    selectedNodeId,
    paintGeneration,
    featureMaps,
    onSelectNode,
    loss,
    probability,
    fillHeight = false,
    height = 420,
    refitViewKey,
  } = props;
  const t = useCnnMessages();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { setViewport } = useReactFlow();
  const nodesInitialized = useNodesInitialized();
  const [measured, setMeasured] = useState({ width: 0, height: 0 });
  const fittedSizeRef = useRef({ width: 0, height: 0 });
  const topologyFitRef = useRef("");
  const refitViewKeyRef = useRef<number | undefined>(undefined);

  const labelFor = useCallback((layer: Parameters<typeof formatCnnNodeLabel>[0]) => formatCnnNodeLabel(layer, t), [t]);

  const mapped = useMemo(
    () =>
      cnnPipelineToFlow(pipeline, {
        selectedNodeId,
        paintGeneration,
        featureMaps,
        loss,
        probability,
        labelFor,
      }),
    [pipeline, selectedNodeId, paintGeneration, featureMaps, loss, probability, labelFor],
  );

  const nodeKey = mapped.nodes.map((n) => n.id).join(",");
  const [nodes, setNodes, onNodesChange] = useNodesState(mapped.nodes);

  const realign = useCallback(() => {
    setNodes((prev) => alignNodesToMidline(prev));
  }, [setNodes]);

  // Sync pipeline → RF nodes (x from layout), then align Y from DOM heights.
  useLayoutEffect(() => {
    setNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      const merged = mapped.nodes.map((n) => {
        const cur = prevById.get(n.id);
        return {
          ...n,
          // Keep prior measured box until DOM align overwrites height.
          measured: cur?.measured,
          position: { x: n.position.x, y: cur?.position.y ?? n.position.y },
        };
      });
      return alignNodesToMidline(merged);
    });
  }, [mapped.nodes, setNodes]);

  // Re-align after RF mount and whenever feature maps repaint (content height changes).
  useLayoutEffect(() => {
    if (!nodesInitialized) return;
    realign();
    const t1 = window.setTimeout(realign, 32);
    const t2 = window.setTimeout(realign, 100);
    const t3 = window.setTimeout(realign, 250);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
    };
  }, [nodesInitialized, nodeKey, paintGeneration, realign]);

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<CnnNodeData>>[]) => {
      onNodesChange(changes);
      if (changes.some((c) => c.type === "dimensions")) {
        requestAnimationFrame(realign);
      }
    },
    [onNodesChange, realign],
  );

  useEffect(() => {
    if (!wrapperRef.current) return;
    const el = wrapperRef.current;
    const sync = () => {
      const rect = el.getBoundingClientRect();
      const width = Math.floor(rect.width);
      const heightPx = Math.floor(rect.height);
      setMeasured((prev) =>
        Math.abs(prev.width - width) > 1 || Math.abs(prev.height - heightPx) > 1
          ? { width, height: heightPx }
          : prev,
      );
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (measured.width <= 0 || measured.height <= 0 || !nodes.length) return;

    const topologyPending = topologyFitRef.current !== nodeKey;
    const refitPending = refitViewKey !== undefined && refitViewKeyRef.current !== refitViewKey;
    const sizeDrift =
      Math.abs(measured.width - fittedSizeRef.current.width) > 24 ||
      Math.abs(measured.height - fittedSizeRef.current.height) > 24;
    const resizeRefit =
      fittedSizeRef.current.width > 0 && sizeDrift && topologyFitRef.current !== "";

    if (!topologyPending && !refitPending && !resizeRefit) return;

    const viewport = viewportForNodes(nodes, measured, fillHeight);
    if (!viewport) return;

    const hadPriorFit = topologyFitRef.current !== "";
    void setViewport(viewport, {
      duration: (refitPending || resizeRefit) && hadPriorFit ? 200 : 0,
    });
    fittedSizeRef.current = measured;
    if (topologyPending) topologyFitRef.current = nodeKey;
    if (refitPending) refitViewKeyRef.current = refitViewKey;
  }, [measured, nodes, nodeKey, refitViewKey, fillHeight, setViewport]);

  const onNodeClick = (_: React.MouseEvent, node: Node<CnnNodeData>) => {
    onSelectNode(node.id);
  };
  const onPaneClick = () => onSelectNode(null);

  const canvasHeight = fillHeight ? measured.height || height : height;

  return (
    <div
      className={`nn-flow-wrap${fillHeight ? " nn-flow-wrap--fill" : ""}`}
      ref={wrapperRef}
      style={fillHeight ? undefined : { height: canvasHeight }}
    >
      <div
        className="nn-flow-canvas"
        style={{ width: "100%", height: fillHeight ? "100%" : canvasHeight }}
      >
        <ReactFlow
          nodes={nodes}
          edges={mapped.edges}
          nodeTypes={cnnNodeTypes}
          edgeTypes={cnnEdgeTypes}
          onNodesChange={handleNodesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
        >
          <Background gap={20} size={1} color="var(--nn-border)" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
      {props.children && <div className="nn-flow-overlays">{props.children}</div>}
    </div>
  );
}

export function CnnFlowGraph(props: CnnFlowGraphProps) {
  return (
    <FeatureMapRefContext.Provider value={props.featureMapRef}>
      <TrainingStatsRefContext.Provider value={props.statsRef}>
        <TrainingLiveRefContext.Provider value={props.trainingLiveRef}>
          <PaintGenerationContext.Provider value={props.paintGeneration}>
            <ReactFlowProvider>
              <CnnFlowGraphInner {...props} />
            </ReactFlowProvider>
          </PaintGenerationContext.Provider>
        </TrainingLiveRefContext.Provider>
      </TrainingStatsRefContext.Provider>
    </FeatureMapRefContext.Provider>
  );
}
