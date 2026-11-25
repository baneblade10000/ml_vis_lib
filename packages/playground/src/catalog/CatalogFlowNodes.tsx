import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { withAlpha, type CatalogNodeKind } from "./layout";

export interface TreeNodeData extends Record<string, unknown> {
  kind: CatalogNodeKind;
  label: string;
  accent: string;
  ghost: boolean;
  collapsed: boolean;
  vizCount: number;
  dimmed: boolean;
  matched: boolean;
  /** Tooltip text for planned (empty) branches. */
  ghostHint?: string;
}

export interface VizCardData extends Record<string, unknown> {
  title: string;
  description: string;
  accent: string;
  dimmed: boolean;
  matched: boolean;
}

export type CatalogTreeFlowNode = Node<TreeNodeData, "root" | "section" | "subtopic">;
export type CatalogVizFlowNode = Node<VizCardData, "viz">;

export function RootFlowNode({ data }: NodeProps<CatalogTreeFlowNode>) {
  return (
    <div
      className="catalog-tree-node catalog-root-node"
      style={{ "--topic-accent": data.accent } as React.CSSProperties}
    >
      <Handle type="source" position={Position.Right} className="catalog-graph-handle" />
      <span className="catalog-root-node-label">{data.label}</span>
    </div>
  );
}

export function SectionFlowNode({ data }: NodeProps<CatalogTreeFlowNode>) {
  return (
    <div
      className={`catalog-tree-node catalog-section-node${data.dimmed ? " is-dimmed" : ""}${data.matched ? " is-matched" : ""}`}
      style={
        {
          "--topic-accent": data.accent,
          "--topic-accent-soft": withAlpha(data.accent, 0.13),
        } as React.CSSProperties
      }
    >
      <Handle type="target" position={Position.Left} className="catalog-graph-handle" />
      <Handle type="source" position={Position.Right} className="catalog-graph-handle" />
      <span className="catalog-node-dot" />
      <span className="catalog-section-node-label">{data.label}</span>
      <span className="catalog-section-node-meta">
        <span className="catalog-section-node-count">{data.vizCount}</span>
        <span className="catalog-section-node-chevron">{data.collapsed ? "▸" : "▾"}</span>
      </span>
    </div>
  );
}

export function SubtopicFlowNode({ data }: NodeProps<CatalogTreeFlowNode>) {
  return (
    <div
      className={`catalog-tree-node catalog-subtopic-node${data.ghost ? " is-ghost" : ""}${data.dimmed ? " is-dimmed" : ""}${data.matched ? " is-matched" : ""}`}
      style={
        {
          "--topic-accent": data.accent,
          "--topic-accent-soft": withAlpha(data.accent, 0.13),
        } as React.CSSProperties
      }
      title={data.ghost ? data.ghostHint : undefined}
    >
      <Handle type="target" position={Position.Left} className="catalog-graph-handle" />
      <span className="catalog-subtopic-node-label">{data.label}</span>
      {data.vizCount > 0 && <span className="catalog-subtopic-node-count">{data.vizCount}</span>}
      <Handle type="source" position={Position.Right} className="catalog-graph-handle" />
    </div>
  );
}

export function VizFlowNode({ data }: NodeProps<CatalogVizFlowNode>) {
  return (
    <div
      className={`catalog-tree-node catalog-viz-node${data.matched ? " is-highlighted" : ""}${data.dimmed ? " is-dimmed" : ""}`}
      style={
        {
          "--topic-accent": data.accent,
          "--topic-accent-soft": withAlpha(data.accent, 0.13),
        } as React.CSSProperties
      }
    >
      <Handle type="target" position={Position.Left} className="catalog-graph-handle" />
      <h4 className="catalog-viz-node-title">{data.title}</h4>
      <p className="catalog-viz-node-description">{data.description}</p>
    </div>
  );
}

export const catalogNodeTypes = {
  root: RootFlowNode,
  section: SectionFlowNode,
  subtopic: SubtopicFlowNode,
  viz: VizFlowNode,
};
