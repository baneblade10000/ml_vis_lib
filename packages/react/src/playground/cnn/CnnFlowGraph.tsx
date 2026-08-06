import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import {
  Background,
  Controls,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useNodesInitialized,
  useReactFlow,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { CnnEngine, FeatureMapSnapshot } from "@ml-vis/core";
import { cnnPipelineToFlow, type CnnNodeData } from "./cnnAdapter";
import { cnnEdgeTypes, cnnNodeTypes } from "./CnnFlowNodes";
import {
  FeatureMapRefContext,
  PaintGenerationContext,
  TrainingLiveRefContext,
  TrainingStatsRefContext,
  type CnnTrainingStats,
  type FeatureMapStore,
} from "./featureMapContext";

export interface CnnFlowGraphProps {
  engine: CnnEngine;
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

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 1.5;

function CnnFlowGraphInner(props: CnnFlowGraphProps) {
  const {
    engine,
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
  const wrapperRef = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();
  const nodesInitialized = useNodesInitialized();
  const [measuredHeight, setMeasuredHeight] = useState(height);
  /** Measured vertical offsets so every node center sits on one axis. */
  const [centerYs, setCenterYs] = useState<Record<string, number>>({});

  const mapped = useMemo(
    () =>
      cnnPipelineToFlow(engine, {
        selectedNodeId,
        paintGeneration,
        featureMaps,
        loss,
        probability,
      }),
    [engine, selectedNodeId, paintGeneration, featureMaps, loss, probability],
  );

  const nodeKey = mapped.nodes.map((n) => n.id).join(",");

  // Track measured container height for fill mode.
  useEffect(() => {
    if (!wrapperRef.current || !fillHeight) return;
    const el = wrapperRef.current;
    let last = 0;
    const sync = () => {
      const h = Math.floor(el.getBoundingClientRect().height);
      if (h > 0 && Math.abs(h - last) > 1) {
        last = h;
        setMeasuredHeight(h);
      }
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, [fillHeight]);

  // After React Flow measures real node boxes, re-center on a shared midline.
  useLayoutEffect(() => {
    if (!nodesInitialized) return;
    const align = () => {
      const live = rf.getNodes();
      if (!live.length) return;
      const heights = live.map((n) => n.measured?.height ?? n.height ?? 0);
      const maxH = Math.max(0, ...heights);
      if (maxH < 1) return;
      const next: Record<string, number> = {};
      for (const n of live) {
        const h = n.measured?.height ?? n.height ?? 0;
        next[n.id] = (maxH - h) / 2;
      }
      setCenterYs((prev) => {
        const same =
          Object.keys(next).length === Object.keys(prev).length &&
          Object.entries(next).every(([id, y]) => Math.abs((prev[id] ?? NaN) - y) < 0.5);
        return same ? prev : next;
      });
    };
    align();
    const t = window.setTimeout(align, 60);
    return () => window.clearTimeout(t);
    // Remeasure when topology or feature-map content size may change node boxes.
  }, [nodesInitialized, nodeKey, featureMaps, rf]);

  const nodes = useMemo(
    () =>
      mapped.nodes.map((n) => ({
        ...n,
        position: { x: n.position.x, y: centerYs[n.id] ?? n.position.y },
      })),
    [mapped.nodes, centerYs],
  );

  // Refit the view to show all layers whenever topology changes or on demand.
  const prevNodeKey = useRef("");
  useEffect(() => {
    if (prevNodeKey.current === nodeKey && refitViewKey === undefined) return;
    prevNodeKey.current = nodeKey;
    const t = window.setTimeout(() => rf.fitView({ padding: 0.2, maxZoom: 1 }), 30);
    return () => window.clearTimeout(t);
  }, [nodeKey, refitViewKey, rf]);

  const onNodeClick = (_: React.MouseEvent, node: Node<CnnNodeData>) => {
    onSelectNode(node.id);
  };
  const onPaneClick = () => onSelectNode(null);

  const canvasHeight = fillHeight ? measuredHeight : height;

  return (
    <div
      className={`tf-flow-wrap${fillHeight ? " tf-flow-wrap--fill" : ""}`}
      ref={wrapperRef}
      style={fillHeight ? undefined : { height: canvasHeight }}
    >
      <div className="tf-flow-canvas" style={{ width: "100%", height: fillHeight ? measuredHeight : canvasHeight }}>
        <ReactFlow
          nodes={nodes}
          edges={mapped.edges}
          nodeTypes={cnnNodeTypes}
          edgeTypes={cnnEdgeTypes}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
        >
          <Background gap={20} size={1} color="var(--tf-border)" />
          <Controls showInteractive={false} position="bottom-right" />
          <Panel position="bottom-right" className="tf-flow-panel-legend">
            <div className="tf-weight-legend" role="img" aria-label="Weight magnitude: violet (neg), magenta (pos)">
              <span className="tf-weight-legend__title">Weights</span>
              <div className="tf-weight-legend__bar" />
              <div className="tf-weight-legend__scale">
                <span>−</span>
                <span>0</span>
                <span>+</span>
              </div>
            </div>
          </Panel>
        </ReactFlow>
      </div>
      {props.children && <div className="tf-flow-overlays">{props.children}</div>}
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
