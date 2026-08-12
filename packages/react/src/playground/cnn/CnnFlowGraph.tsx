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
  useNodesState,
  useReactFlow,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { FeatureMapSnapshot } from "@ml-vis/core";
import {
  CNN_COL_GAP,
  CNN_ORIGIN_X,
  cnnPipelineToFlow,
  formatCnnNodeLabel,
  type CnnNodeData,
  type CnnPipelineView,
} from "./cnnAdapter";
import { cnnEdgeTypes, cnnNodeTypes } from "./CnnFlowNodes";
import {
  CnnPlayVizRefContext,
  FeatureMapRefContext,
  KernelExpandContext,
  PaintGenerationContext,
  ReceptiveFieldContext,
  TrainingLiveRefContext,
  TrainingStatsRefContext,
  type CnnPlayViz,
  type CnnTrainingStats,
  type FeatureMapStore,
  type KernelExpandApi,
  type KernelExpandSelection,
  type ReceptiveFieldApi,
} from "./featureMapContext";
import { useCnnMessages } from "./messages";
import {
  buildRfLayerMetas,
  type RfSelection,
} from "./receptiveField";

export interface CnnFlowGraphProps {
  pipeline: CnnPipelineView;
  selectedNodeId: string | null;
  paintGeneration: number;
  featureMaps: FeatureMapSnapshot[];
  /** Config kernel sizes by layer id — layout must react before WASM dumps catch up. */
  layerKernelSizes?: Record<string, number>;
  onSelectNode: (nodeId: string | null) => void;
  featureMapRef: RefObject<FeatureMapStore>;
  statsRef: RefObject<CnnTrainingStats>;
  trainingLiveRef: RefObject<boolean>;
  playVizRef: RefObject<CnnPlayViz>;
  loss?: number;
  probability?: number;
  height?: number;
  fillHeight?: boolean;
  children?: ReactNode;
  refitViewKey?: number;
}

/** Position is the left-center of each node — vertical centers share one Y. */
const NODE_ORIGIN: [number, number] = [0, 0.5];
const PIPELINE_Y = 0;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 1.5;
const FIT_MAX_ZOOM = 1.05;
const LABEL_MARGIN = 20;

/**
 * Viewport that fits the pipeline between immersive side docks.
 * Accounts for nodeOrigin [0, 0.5] (position.y is the vertical center).
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
    const top = node.position.y - h / 2;
    const bottom = node.position.y + h / 2 + LABEL_MARGIN;
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, top);
    maxX = Math.max(maxX, node.position.x + w);
    maxY = Math.max(maxY, bottom);
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

/** Pack columns left→right from live widths; keep vertical centers on the midline. */
function layoutPipelineNodes(nodes: Node<CnnNodeData>[]): Node<CnnNodeData>[] {
  let x = CNN_ORIGIN_X;
  return nodes.map((n) => {
    const width = n.measured?.width ?? n.width ?? 0;
    const height = n.measured?.height ?? n.height ?? 0;
    const next = {
      ...n,
      width,
      height,
      position: { x, y: PIPELINE_Y },
    };
    x += width + CNN_COL_GAP;
    return next;
  });
}

function CnnFlowGraphInner(props: CnnFlowGraphProps) {
  const {
    pipeline,
    selectedNodeId,
    featureMaps,
    layerKernelSizes,
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
  const [measured, setMeasured] = useState({ width: 0, height: 0 });
  const fittedSizeRef = useRef({ width: 0, height: 0 });
  const topologyFitRef = useRef("");
  const refitViewKeyRef = useRef<number | undefined>(undefined);

  const labelFor = useCallback(
    (layer: Parameters<typeof formatCnnNodeLabel>[0]) => formatCnnNodeLabel(layer, t),
    [t],
  );

  const kernelSizeByLayerId = useMemo(() => {
    const out: Record<string, number> = { ...(layerKernelSizes ?? {}) };
    for (const m of featureMaps) {
      if (out[m.layerId] != null) continue;
      const k2 = m.kernels2d?.[0]?.length;
      const k1 = m.kernels1d?.[0]?.length;
      if (k2 && k2 > 0) out[m.layerId] = k2;
      else if (k1 && k1 > 0) out[m.layerId] = k1;
    }
    return out;
  }, [featureMaps, layerKernelSizes]);

  const rfLayers = useMemo(
    () => buildRfLayerMetas(pipeline.layers, kernelSizeByLayerId),
    [pipeline.layers, kernelSizeByLayerId],
  );

  const [rfSelection, setRfSelection] = useState<RfSelection | null>(null);
  const [rfPinned, setRfPinned] = useState(false);
  const [kernelExpand, setKernelExpand] = useState<KernelExpandSelection | null>(null);

  const rfApi = useMemo<ReceptiveFieldApi>(
    () => ({
      layers: rfLayers,
      selection: rfSelection,
      pinned: rfPinned,
      setHover: (next) => {
        if (!rfPinned) setRfSelection(next);
      },
      pin: (next) => {
        if (
          next &&
          rfPinned &&
          rfSelection &&
          rfSelection.sourceLayerId === next.sourceLayerId &&
          rfSelection.channel === next.channel &&
          rfSelection.outY === next.outY &&
          rfSelection.outX === next.outX
        ) {
          setRfPinned(false);
          setRfSelection(null);
          return;
        }
        setRfPinned(!!next);
        setRfSelection(next);
      },
      clear: () => {
        setRfPinned(false);
        setRfSelection(null);
      },
    }),
    [rfLayers, rfSelection, rfPinned],
  );

  const kernelExpandApi = useMemo<KernelExpandApi>(
    () => ({
      selection: kernelExpand,
      prevLayerId: (layerId: string) => {
        const idx = pipeline.layers.findIndex((l) => l.id === layerId);
        if (idx <= 0) return null;
        return pipeline.layers[idx - 1]?.id ?? null;
      },
      selectLayer: (layerId: string) => onSelectNode(layerId),
      toggle: (next) => {
        setKernelExpand((cur) =>
          cur &&
          cur.layerId === next.layerId &&
          cur.filter === next.filter
            ? null
            : next,
        );
      },
      clear: () => setKernelExpand(null),
    }),
    [kernelExpand, pipeline.layers, onSelectNode],
  );

  void loss;
  void probability;
  const mapped = useMemo(
    () =>
      cnnPipelineToFlow(pipeline, {
        selectedNodeId,
        featureMaps,
        kernelSizeByLayerId,
        labelFor,
      }),
    [pipeline, selectedNodeId, featureMaps, kernelSizeByLayerId, labelFor],
  );

  const nodeKey = mapped.nodes.map((n) => n.id).join(",");
  /** Include sizes so kernel / filter changes re-fit the viewport. */
  const layoutKey = mapped.nodes
    .map(
      (n) =>
        `${n.id}:${n.data.channels}:${n.data.length ?? 0}:${n.data.kernelSize ?? 0}:${n.width ?? 0}:${n.height ?? 0}`,
    )
    .join(",");

  useEffect(() => {
    setRfPinned(false);
    setRfSelection(null);
    setKernelExpand(null);
  }, [nodeKey]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutPipelineNodes(mapped.nodes));

  // Sync pipeline → RF nodes; re-pack X from widths; keep vertical centers on the midline.
  useLayoutEffect(() => {
    setNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      const merged = mapped.nodes.map((n) => {
        const cur = prevById.get(n.id);
        // Same-pad keeps rows/cols fixed when kernel size changes — must key on kernelSize.
        const sizeDirty =
          !!cur &&
          (cur.data.channels !== n.data.channels ||
            cur.data.length !== n.data.length ||
            cur.data.rows !== n.data.rows ||
            cur.data.cols !== n.data.cols ||
            cur.data.kernelSize !== n.data.kernelSize);
        return {
          ...n,
          measured: sizeDirty ? undefined : cur?.measured,
          width: sizeDirty
            ? n.width
            : (cur?.measured?.width ?? cur?.width ?? n.width),
          height: sizeDirty
            ? n.height
            : (cur?.measured?.height ?? cur?.height ?? n.height),
        };
      });
      return layoutPipelineNodes(merged);
    });
  }, [mapped.nodes, setNodes]);

  const handleNodesChange = useCallback(
    (changes: NodeChange<Node<CnnNodeData>>[]) => {
      onNodesChange(changes);
      // After RF measures DOM, re-pack X only when widths actually moved.
      if (changes.some((c) => c.type === "dimensions")) {
        requestAnimationFrame(() => {
          setNodes((prev) => {
            let dirty = false;
            const sized = prev.map((node) => {
              const mh = node.measured?.height;
              const mw = node.measured?.width;
              if (mh == null && mw == null) return node;
              const width = mw ?? node.width;
              const height = mh ?? node.height;
              if (width !== node.width || height !== node.height) dirty = true;
              return width === node.width && height === node.height
                ? node
                : { ...node, width, height };
            });
            const next = layoutPipelineNodes(sized);
            if (!dirty && next.every((n, i) => n.position.x === prev[i]?.position.x)) {
              return prev;
            }
            return next;
          });
        });
      }
    },
    [onNodesChange, setNodes],
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

    const topologyPending = topologyFitRef.current !== layoutKey;
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
      duration: (refitPending || resizeRefit || hadPriorFit) && hadPriorFit ? 200 : 0,
    });
    fittedSizeRef.current = measured;
    if (topologyPending) topologyFitRef.current = layoutKey;
    if (refitPending) refitViewKeyRef.current = refitViewKey;
  }, [measured, nodes, layoutKey, refitViewKey, fillHeight, setViewport]);

  const onNodeClick = (_: React.MouseEvent, node: Node<CnnNodeData>) => {
    onSelectNode(node.id);
  };
  const onPaneClick = () => {
    onSelectNode(null);
    rfApi.clear();
    kernelExpandApi.clear();
  };

  const canvasHeight = fillHeight ? measured.height || height : height;

  return (
    <ReceptiveFieldContext.Provider value={rfApi}>
    <KernelExpandContext.Provider value={kernelExpandApi}>
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
          nodeOrigin={NODE_ORIGIN}
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
    </KernelExpandContext.Provider>
    </ReceptiveFieldContext.Provider>
  );
}

export function CnnFlowGraph(props: CnnFlowGraphProps) {
  return (
    <FeatureMapRefContext.Provider value={props.featureMapRef}>
      <TrainingStatsRefContext.Provider value={props.statsRef}>
        <TrainingLiveRefContext.Provider value={props.trainingLiveRef}>
          <CnnPlayVizRefContext.Provider value={props.playVizRef}>
            <PaintGenerationContext.Provider value={props.paintGeneration}>
              <ReactFlowProvider>
                <CnnFlowGraphInner {...props} />
              </ReactFlowProvider>
            </PaintGenerationContext.Provider>
          </CnnPlayVizRefContext.Provider>
        </TrainingLiveRefContext.Provider>
      </TrainingStatsRefContext.Provider>
    </FeatureMapRefContext.Provider>
  );
}
