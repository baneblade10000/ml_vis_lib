import { getVisualizationById, type VisualizationId } from "../visualizations";
import { catalogKnowledgeTree, type CatalogTreeNode } from "./tree";

export function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  if (value.length !== 6) return hex;
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export type CatalogNodeKind = "root" | "section" | "subtopic";

export interface CatalogTreePlacement {
  id: string;
  parentId: string | null;
  kind: CatalogNodeKind;
  accent: string;
  /** Sections only: whether the subtree is folded away. */
  collapsed: boolean;
  /** Number of visualization cards hanging off this node. */
  vizCount: number;
  x: number;
  y: number;
}

export interface CatalogVizCardPlacement {
  vizId: VisualizationId;
  nodeId: string;
  accent: string;
  x: number;
  y: number;
}

export interface CatalogTreeLayout {
  nodes: CatalogTreePlacement[];
  cards: CatalogVizCardPlacement[];
  /** Parent→child pairs (node ids) and node→card pairs, in draw order. */
  edges: Array<{ source: string; target: string; accent: string; toCard: boolean }>;
}

/* Columns by depth: root → sections → subtopics → visualization cards. */
const COLUMN_X = [0, 470, 1030, 1590];
const SLOT_HEIGHT = 64;
/** CSS heights of the node boxes (keep in sync with network-playground.css). */
const SUBTOPIC_HEIGHT = 46;
const CARD_HEIGHT = 88;
const CARD_STACK_GAP = 8;

function vizCountOf(node: CatalogTreeNode): number {
  return node.vizIds?.length ?? 0;
}

/**
 * Deterministic tidy-tree layout (left → right). Leaves take consecutive
 * vertical slots; every parent is centered on its children. A collapsed
 * section behaves like a leaf. Visualization cards occupy extra slots
 * right of their node.
 */
export function computeKnowledgeTreeLayout(
  collapsed: Set<string>,
  resolveViz: (vizId: VisualizationId) => { exists: boolean } = (id) => ({ exists: getVisualizationById(id) !== undefined }),
): CatalogTreeLayout {
  const nodes: CatalogTreePlacement[] = [];
  const cards: CatalogVizCardPlacement[] = [];
  const edges: CatalogTreeLayout["edges"] = [];
  let slot = 0;
  const root = catalogKnowledgeTree;

  const walk = (
    node: CatalogTreeNode,
    depth: number,
    parentId: string | null,
    inheritedAccent: string,
  ): number => {
    const accent = node.accent ?? inheritedAccent;
    const kind: CatalogNodeKind = depth === 0 ? "root" : depth === 1 ? "section" : "subtopic";
    const children = node.children ?? [];
    const isCollapsed = kind === "section" && collapsed.has(node.id);
    const x = COLUMN_X[Math.min(depth, COLUMN_X.length - 2)];

    if (children.length > 0 && !isCollapsed) {
      const childYs = children.map((child) => walk(child, depth + 1, node.id, accent));
      const y = (Math.min(...childYs) + Math.max(...childYs)) / 2;
      nodes.push({ id: node.id, parentId, kind, accent, collapsed: false, vizCount: vizCountOf(node), x, y });
      for (const child of children) edges.push({ source: node.id, target: child.id, accent, toCard: false });
      return y;
    }

    const count = vizCountOf(node);
    const ownY = slot * SLOT_HEIGHT;
    slot += 1;
    nodes.push({
      id: node.id,
      parentId,
      kind,
      accent,
      collapsed: isCollapsed,
      vizCount: count,
      x,
      y: ownY,
    });

    for (let index = 0; index < count; index += 1) {
      const vizId = node.vizIds![index];
      if (!resolveViz(vizId).exists) continue;
      // First card is center-aligned with its subtopic row, extras stack below.
      const y = ownY + (SUBTOPIC_HEIGHT - CARD_HEIGHT) / 2 + index * (CARD_HEIGHT + CARD_STACK_GAP);
      cards.push({ vizId, nodeId: node.id, accent, x: COLUMN_X[3], y });
      edges.push({ source: node.id, target: vizId, accent, toCard: true });
    }
    return ownY;
  };

  walk(root, 0, null, "#334155");
  return { nodes, cards, edges };
}
