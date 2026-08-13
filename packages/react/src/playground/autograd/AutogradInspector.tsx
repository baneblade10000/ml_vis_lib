import { OP_SPECS, type AutogradGraph } from "@ml-vis/core/autograd";
import { useAutogradMessages } from "./messages";

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "∞";
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(3);
}

export function AutogradInspector({
  graph,
  version,
  showGrad,
  showValues,
  selectedNodeId,
  selectedEdgeId,
  onSetNodeValue,
  onSetOutput,
  onRemoveNode,
  onRemoveEdge,
}: {
  graph: AutogradGraph;
  version: number;
  showGrad: boolean;
  showValues: boolean;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSetNodeValue: (id: string, value: number) => void;
  onSetOutput: (id: string) => void;
  onRemoveNode: (id: string) => void;
  onRemoveEdge: (id: string) => void;
}) {
  void version;
  const t = useAutogradMessages();
  const node = selectedNodeId ? graph.getNode(selectedNodeId) : undefined;
  const edge = selectedEdgeId ? graph.edges.get(selectedEdgeId) : undefined;

  if (!node && !edge) {
    return (
      <div className="nn-network-inspector">
        <div className="nn-network-inspector-empty">{t.inspectorEmpty}</div>
      </div>
    );
  }

  if (edge) {
    return (
      <div className="nn-network-inspector">
        <div className="nn-network-inspector-title">{t.inspectorEdge}</div>
        <div className="nn-network-inspector-row">
          <span className="label">{t.inspectorFrom}</span>
          <span className="value">{graph.getNode(edge.source)?.label ?? edge.source}</span>
        </div>
        <div className="nn-network-inspector-row">
          <span className="label">{t.inspectorTo}</span>
          <span className="value">{graph.getNode(edge.target)?.label ?? edge.target}</span>
        </div>
        {showGrad && (
          <div className="nn-network-inspector-row">
            <span className="label">{t.inspectorLocalDer}</span>
            <span className="value">{fmt(edge.localDer)}</span>
          </div>
        )}
        <button type="button" className="nn-btn nn-btn--ghost nn-btn--sm" onClick={() => onRemoveEdge(edge.id)}>
          {t.removeEdge}
        </button>
      </div>
    );
  }

  if (!node) return null;

  const isOutput = graph.outputId === node.id;

  return (
    <div className="nn-network-inspector">
      <div className="nn-network-inspector-title">{node.label ?? OP_SPECS[node.op].label}</div>
      <div className="nn-network-inspector-row">
        <span className="label">{t.inspectorOp}</span>
        <span className="value">{t.opLabels[node.op]}</span>
      </div>

      {node.isLeaf ? (
        <label className="ag-value-field">
          <span className="label">{t.inspectorValue}</span>
          <input
            type="number"
            step="0.1"
            className="nn-select nn-select--dock ag-value-input"
            value={Number.isFinite(node.value) ? node.value : 0}
            onChange={(e) => onSetNodeValue(node.id, Number(e.target.value))}
          />
        </label>
      ) : (
        showValues && (
          <div className="nn-network-inspector-row">
            <span className="label">{t.value}</span>
            <span className="value">{fmt(node.value)}</span>
          </div>
        )
      )}

      {showGrad && (
        <div className="nn-network-inspector-row">
          <span className="label">{t.grad}</span>
          <span className="value">{fmt(node.grad)}</span>
        </div>
      )}

      {!isOutput && !node.isLeaf && (
        <button type="button" className="nn-btn nn-btn--ghost nn-btn--sm" onClick={() => onSetOutput(node.id)}>
          {t.setAsOutput}
        </button>
      )}
      <button type="button" className="nn-btn nn-btn--ghost nn-btn--sm" onClick={() => onRemoveNode(node.id)}>
        {t.removeNode}
      </button>
    </div>
  );
}
