import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useI18n } from "@ml-vis/react";
import { playgroundMessages, usePlaygroundT } from "../i18n";
import { getVisualizationById, visualizations, type VisualizationId } from "../visualizations";
import { catalogKnowledgeTree, type CatalogTreeNode } from "./tree";
import { computeKnowledgeTreeLayout, withAlpha } from "./layout";
import { searchCatalog } from "./search";
import {
  catalogNodeTypes,
  type CatalogTreeFlowNode,
  type CatalogVizFlowNode,
  type TreeNodeData,
  type VizCardData,
} from "./CatalogFlowNodes";

type CatalogFlowNode = CatalogTreeFlowNode | CatalogVizFlowNode;

const MIN_ZOOM = 0.12;
const MAX_ZOOM = 1.5;
const FIT_MAX_ZOOM = 0.9;

interface BuiltGraph {
  nodes: CatalogFlowNode[];
  edges: Edge[];
  /** The query is not blank and lit nothing. */
  noMatches: boolean;
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
}

/** Fit viewport for given flow bounds into a container — no DOM measurement needed. */
function viewportForBounds(
  bounds: BuiltGraph["bounds"],
  width: number,
  height: number,
): Viewport {
  if (width <= 0 || height <= 0) return { x: 0, y: 0, zoom: 1 };
  // Rendered boxes run slightly larger than the declared size hints.
  const boundsWidth = Math.max(bounds.maxX - bounds.minX, 1) * 1.1;
  const boundsHeight = Math.max(bounds.maxY - bounds.minY, 1) * 1.2;
  const zoom = Math.min(
    Math.max(Math.min((width - 32) / boundsWidth, (height - 32) / boundsHeight), MIN_ZOOM),
    FIT_MAX_ZOOM,
  );
  return {
    x: 16 + (width - 32 - (bounds.maxX - bounds.minX) * zoom) / 2 - bounds.minX * zoom,
    y: 16 + (height - 32 - (bounds.maxY - bounds.minY) * zoom) / 2 - bounds.minY * zoom,
    zoom,
  };
}

function collectTree(
  node: CatalogTreeNode,
  depth: number,
  visit: (node: CatalogTreeNode, depth: number) => void,
) {
  visit(node, depth);
  for (const child of node.children ?? []) collectTree(child, depth + 1, visit);
}

function buildGraph(locale: "en" | "ru", query: string, collapsed: Set<string>): BuiltGraph {
  const messages = playgroundMessages[locale];
  const layout = computeKnowledgeTreeLayout(collapsed);
  const { searching, matchedNodes, matchedViz, lit } = searchCatalog(locale, query);

  const labels = new Map<string, string>();
  collectTree(catalogKnowledgeTree, 0, (node) => {
    labels.set(node.id, locale === "ru" ? node.label.ru : node.label.en);
  });

  const vizById = new Map(visualizations.map((viz) => [viz.id, viz]));
  const isNodeDimmed = (id: string) => searching && !lit.has(id);
  const isCardDimmed = (cardNodeId: string) => searching && !lit.has(cardNodeId);

  // Known CSS box sizes: lets React Flow lay out and fit the view without
  // waiting for DOM measurement (which some embedded webviews never deliver).
  const NODE_SIZE: Record<string, { width: number; height: number }> = {
    root: { width: 250, height: 64 },
    section: { width: 264, height: 52 },
    subtopic: { width: 300, height: 46 },
    viz: { width: 268, height: 88 },
  };

  const nodes: CatalogFlowNode[] = [];

  for (const placement of layout.nodes) {
    const data: TreeNodeData = {
      kind: placement.kind,
      label: labels.get(placement.id) ?? placement.id,
      accent: placement.accent,
      ghost: placement.kind === "subtopic" && placement.vizCount === 0,
      collapsed: placement.collapsed,
      vizCount: placement.vizCount,
      dimmed: isNodeDimmed(placement.id),
      matched: searching && matchedNodes.has(placement.id),
      ghostHint:
        locale === "ru"
          ? "Раздел запланирован — визуализации появятся позже"
          : "Planned branch — visualizations are coming later",
    };
    nodes.push({
      id: placement.id,
      type: placement.kind,
      position: { x: placement.x, y: placement.y },
      ...NODE_SIZE[placement.kind],
      data,
    });
  }

  for (const card of layout.cards) {
    const viz = vizById.get(card.vizId);
    if (!viz) continue;
    const data: VizCardData = {
      title: messages[viz.titleKey],
      description: messages[viz.descriptionKey],
      accent: card.accent,
      dimmed: isCardDimmed(card.nodeId),
      matched: searching && (matchedViz.has(card.vizId) || matchedNodes.has(card.nodeId)),
    };
    nodes.push({
      id: card.vizId,
      type: "viz",
      position: { x: card.x, y: card.y },
      ...NODE_SIZE.viz,
      data,
    });
  }

  const edges: Edge[] = layout.edges.map((link) => ({
    id: `${link.source}->${link.target}`,
    source: link.source,
    target: link.target,
    className:
      searching && isNodeDimmed(link.source) && isNodeDimmed(link.target) && isCardDimmed(link.source)
        ? "is-dimmed"
        : undefined,
    style: {
      stroke: link.source === catalogKnowledgeTree.id ? "#94a3b8" : withAlpha(link.accent, 0.55),
      strokeWidth: link.toCard ? 1.6 : 1.8,
    },
  }));

  return { nodes, edges, noMatches: searching && lit.size === 0, bounds: computeBounds(nodes) };
}

function computeBounds(nodes: CatalogFlowNode[]): BuiltGraph["bounds"] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    const width = node.width ?? 268;
    const height = node.height ?? 88;
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + width);
    maxY = Math.max(maxY, node.position.y + height);
  }
  return { minX, minY, maxX, maxY };
}

interface CatalogTopicGraphProps {
  query: string;
}

function CatalogTopicGraphInner({ query }: CatalogTopicGraphProps) {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const navigate = useNavigate();
  const rf = useReactFlow();
  const canvasRef = useRef<HTMLDivElement>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const { nodes: computedNodes, edges: computedEdges, noMatches, bounds } = useMemo(
    () => buildGraph(locale, query, collapsed),
    [locale, query, collapsed],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<CatalogFlowNode>(computedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges);

  // The canvas has a known CSS height, so the tree bounds alone determine the
  // fit viewport — React Flow's own fitView would wait for DOM measurement.
  const fitCanvas = useCallback(
    (animated: boolean) => {
      const el = canvasRef.current;
      const width = el?.clientWidth || window.innerWidth - 64;
      const height = el?.clientHeight || Math.min(window.innerHeight * 0.62, 660);
      const viewport = viewportForBounds(bounds, width, height);
      rf.setViewport(viewport, animated ? { duration: 420 } : undefined);
    },
    [bounds, rf],
  );

  const initialViewport = useMemo(
    () =>
      viewportForBounds(
        bounds,
        typeof window === "undefined" ? 1280 : window.innerWidth - 48,
        typeof window === "undefined" ? 620 : Math.min(window.innerHeight * 0.62, 660),
      ),
    // Only the very first viewport comes from window metrics; effects refine it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    fitCanvas(false);
    // Late layout settling (fonts, styles) changes the canvas box; re-fit at a
    // few delays instead of relying on ResizeObserver delivery.
    const timers = [100, 400, 1000, 1800].map((delay) =>
      window.setTimeout(() => fitCanvas(false), delay),
    );
    window.addEventListener("resize", handleResize);
    function handleResize() {
      fitCanvas(true);
    }
    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
      window.removeEventListener("resize", handleResize);
    };
    // Re-fit when the tree footprint changes (fold/unfold), not on search/locale.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collapsed]);

  // Recomputed layouts keep dragged positions; labels refresh on locale change.
  useEffect(() => {
    setNodes((prev) => {
      const posById = new Map(prev.map((node) => [node.id, node.position]));
      return computedNodes.map((node) => ({
        ...node,
        position: posById.get(node.id) ?? node.position,
      }));
    });
    setEdges(computedEdges);
  }, [computedNodes, computedEdges, setNodes, setEdges]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: CatalogFlowNode) => {
      if (node.type === "section") {
        setCollapsed((prev) => {
          const next = new Set(prev);
          if (next.has(node.id)) next.delete(node.id);
          else next.add(node.id);
          return next;
        });
        return;
      }
      if (node.type === "viz") {
        const entry = getVisualizationById(node.id as VisualizationId);
        if (entry) navigate(entry.path);
      }
    },
    [navigate],
  );

  return (
    <div className="catalog-graph">
      <div ref={canvasRef} className="catalog-graph-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={catalogNodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          defaultViewport={initialViewport}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnDoubleClick={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} size={1} color="var(--nn-border)" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
        {noMatches && <p className="catalog-graph-empty">{t("catalogSearchEmpty")}</p>}
      </div>
    </div>
  );
}

export function CatalogTopicGraph(props: CatalogTopicGraphProps) {
  return (
    <ReactFlowProvider>
      <CatalogTopicGraphInner {...props} />
    </ReactFlowProvider>
  );
}
