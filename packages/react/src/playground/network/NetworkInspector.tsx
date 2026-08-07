import type { ComputationalGraph } from "@ml-vis/core";
import { linkPartialDerivative } from "./graphAdapter";
import { useNetworkMessages } from "./messages";

export function NetworkInspector({
  graph,
  selectedNodeId,
  selectedEdgeId,
  onRemoveNode,
  onRemoveEdge,
}: {
  graph: ComputationalGraph;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (edgeId: string) => void;
}) {
  const t = useNetworkMessages();
  const node = selectedNodeId ? graph.getNode(selectedNodeId) : null;
  const edge = selectedEdgeId ? graph.getAllLinks().find((l) => l.id === selectedEdgeId) : null;

  if (!node && !edge) {
    return (
      <div className="tf-network-inspector" id="network-inspector">
        <div className="tf-network-inspector-empty">{t.inspectorEmpty}</div>
      </div>
    );
  }

  if (edge) {
    return (
      <div className="tf-network-inspector" id="network-inspector">
        <div className="tf-network-inspector-title">{t.inspectorEdge}</div>
        <div className="tf-network-inspector-row">
          <span className="label">{t.inspectorFrom}</span>
          <span className="value">{edge.source.id}</span>
        </div>
        <div className="tf-network-inspector-row">
          <span className="label">{t.inspectorTo}</span>
          <span className="value">{edge.dest.id}</span>
        </div>
        <div className="tf-network-inspector-row">
          <span className="label">{t.inspectorWeight}</span>
          <span className="value">{edge.weight.toFixed(4)}</span>
        </div>
        <div className="tf-network-inspector-row">
          <span className="label">{t.inspectorGradient}</span>
          <span className="value">{linkPartialDerivative(edge).toFixed(6)}</span>
        </div>
        <button type="button" className="tf-btn tf-btn--ghost tf-btn--sm" onClick={() => onRemoveEdge(edge.id)}>
          {t.removeEdge}
        </button>
      </div>
    );
  }

  if (!node) return null;

  return (
    <div className="tf-network-inspector" id="network-inspector">
      <div className="tf-network-inspector-title">{node.label ?? node.kind}</div>
      <div className="tf-network-inspector-row">
        <span className="label">{t.inspectorKind}</span>
        <span className="value">{node.kind}</span>
      </div>
      <div className="tf-network-inspector-row">
        <span className="label">{t.inspectorOutput}</span>
        <span className="value">{node.output.toFixed(4)}</span>
      </div>
      {node.kind !== "input" && node.kind !== "sum" && (
        <div className="tf-network-inspector-row">
          <span className="label">{t.inspectorBias}</span>
          <span className="value">{node.bias.toFixed(4)}</span>
        </div>
      )}
      {node.kind !== "input" && node.kind !== "output" && (
        <button type="button" className="tf-btn tf-btn--ghost tf-btn--sm" onClick={() => onRemoveNode(node.id)}>
          {t.removeNode}
        </button>
      )}
    </div>
  );
}
