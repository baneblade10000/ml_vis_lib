import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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
  type OnNodeDrag,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AutogradGraph } from "@ml-vis/core/autograd";
import {
  autogradKindFromDrag,
  autogradToFlow,
  AUTOGRAD_DRAG_TYPE,
  type AutogradEdgeData,
  type AutogradNodeData,
} from "./adapter";
import { autogradEdgeTypes, autogradNodeTypes } from "./AutogradFlowNodes";

export interface AutogradFlowGraphProps {
  graph: AutogradGraph;
  version: number;
  showGrad: boolean;
  showValues: boolean;
  children?: ReactNode;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSelectNode: (id: string | null) => void;
  onSelectEdge: (id: string | null) => void;
  onConnect: (sourceId: string, targetId: string) => void;
  onDropNode: (kind: string, position: { x: number; y: number }) => void;
  onMoveNode: (nodeId: string, position: { x: number; y: number }) => void;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (edgeId: string) => void;
  fitViewKey: string;
}

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 1.75;
const FIT_MAX_ZOOM = 1.2;
const LABEL_MARGIN = 24;

function viewportForNodes(
  nodes: Node<AutogradNodeData>[],
  container: { width: number; height: number },
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
  const pad = { top: 32, bottom: 32, left: 290, right: 64 };
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

function AutogradFlowGraphInner({
  graph,
  version,
  showGrad,
  showValues,
  children,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onConnect,
  onDropNode,
  onMoveNode,
  onRemoveNode,
  onRemoveEdge,
  fitViewKey,
}: AutogradFlowGraphProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();
  const fitKeyRef = useRef("");
  const draggingRef = useRef(new Set<string>());

  const { nodes: computedNodes, edges: computedEdges } = useMemo(
    () => autogradToFlow(graph, { selectedNodeId, selectedEdgeId, showGrad, showValues }),
    // version bumps whenever the engine mutates the graph.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph, version, selectedNodeId, selectedEdgeId, showGrad, showValues],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(computedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges);
  const [measured, setMeasured] = useState({ width: 0, height: 0 });

  useEffect(() => {
    setNodes((prev) => {
      const posById = new Map(prev.map((n) => [n.id, n.position]));
      return computedNodes.map((n) =>
        draggingRef.current.has(n.id) ? { ...n, position: posById.get(n.id) ?? n.position } : n,
      );
    });
    setEdges(computedEdges);
  }, [computedNodes, computedEdges, setNodes, setEdges]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => setMeasured({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (fitKeyRef.current === fitViewKey) return;
    if (measured.width <= 0) return;
    const vp = viewportForNodes(computedNodes, measured);
    if (vp) {
      rf.setViewport(vp);
      fitKeyRef.current = fitViewKey;
    }
  }, [fitViewKey, measured, computedNodes, rf]);

  const onNodeDragStart: OnNodeDrag<Node<AutogradNodeData>> = useCallback((_, node) => {
    draggingRef.current.add(node.id);
  }, []);

  const onNodeDragStop: OnNodeDrag<Node<AutogradNodeData>> = useCallback(
    (_, node) => {
      draggingRef.current.delete(node.id);
      onMoveNode(node.id, node.position);
    },
    [onMoveNode],
  );

  const handleConnect: OnConnect = useCallback(
    (c: Connection) => {
      if (c.source && c.target) onConnect(c.source, c.target);
    },
    [onConnect],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const raw =
        event.dataTransfer.getData(AUTOGRAD_DRAG_TYPE) || event.dataTransfer.getData("text/plain");
      const kind = autogradKindFromDrag(raw);
      if (!kind) return;
      const position = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      onDropNode(kind, position);
    },
    [onDropNode, rf],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<AutogradNodeData>) => {
      onSelectEdge(null);
      onSelectNode(node.id);
    },
    [onSelectNode, onSelectEdge],
  );

  const onEdgeClick = useCallback(
    (_: React.MouseEvent, edge: Edge<AutogradEdgeData>) => {
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
        onRemoveEdge(selectedEdgeId);
      } else if (selectedNodeId) {
        onRemoveNode(selectedNodeId);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedEdgeId, selectedNodeId, onRemoveEdge, onRemoveNode]);

  return (
    <div className="nn-flow-wrap nn-flow-wrap--fill">
      <div ref={wrapRef} className="nn-flow-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={autogradNodeTypes}
          edgeTypes={autogradEdgeTypes}
          onNodesChange={onNodesChange}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={onNodeDragStop}
          onEdgesChange={onEdgesChange}
          onConnect={handleConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
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
          <Background gap={20} size={1} color="var(--nn-border)" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
      {children && <div className="nn-flow-overlays">{children}</div>}
    </div>
  );
}

export function AutogradFlowGraph(props: AutogradFlowGraphProps) {
  return (
    <ReactFlowProvider>
      <AutogradFlowGraphInner {...props} />
    </ReactFlowProvider>
  );
}
