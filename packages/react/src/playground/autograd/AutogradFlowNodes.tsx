import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  Handle,
  Position,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { gradColor, type AutogradEdgeData, type AutogradNodeData } from "./adapter";

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "∞";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001) return value.toExponential(1);
  return value.toFixed(Math.abs(value) < 1 ? 3 : 2);
}

function NodeBody({ data }: { data: AutogradNodeData }) {
  return (
    <>
      <div className="ag-node-head">
        <span className="ag-node-symbol">{data.symbol}</span>
        <span className="ag-node-label">{data.label}</span>
      </div>
      <div className="ag-node-stats">
        <span className="ag-node-value" title="value">{fmt(data.value)}</span>
        {data.showGrad && (
          <span className="ag-node-grad" style={{ color: gradColor(data.grad) }} title="grad">
            ∂ {fmt(data.grad)}
          </span>
        )}
      </div>
    </>
  );
}

export function LeafFlowNode({ data }: NodeProps<Node<AutogradNodeData>>) {
  return (
    <div className={`ag-node ag-node--leaf ag-node--${data.op}${data.selected ? " selected" : ""}`}>
      <NodeBody data={data} />
      <Handle type="source" position={Position.Right} className="ag-handle" />
    </div>
  );
}

export function OpFlowNode({ data }: NodeProps<Node<AutogradNodeData>>) {
  return (
    <div className={`ag-node ag-node--op${data.selected ? " selected" : ""}`}>
      <Handle type="target" position={Position.Left} className="ag-handle" />
      <NodeBody data={data} />
      <Handle type="source" position={Position.Right} className="ag-handle" />
    </div>
  );
}

export function SinkFlowNode({ data }: NodeProps<Node<AutogradNodeData>>) {
  return (
    <div className={`ag-node ag-node--op ag-node--sink${data.selected ? " selected" : ""}`}>
      <Handle type="target" position={Position.Left} className="ag-handle" />
      <NodeBody data={data} />
    </div>
  );
}

export const autogradNodeTypes = {
  leaf: LeafFlowNode,
  op: OpFlowNode,
  sink: SinkFlowNode,
};

function edgeFmt(value: number): string {
  if (!Number.isFinite(value)) return "∞";
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs < 0.001 && abs > 0)) return value.toExponential(1);
  return value.toFixed(2);
}

export function AutogradFlowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
}: EdgeProps<Edge<AutogradEdgeData>>) {
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  const localDer = data?.localDer ?? 0;
  const showGrad = data?.showGrad ?? false;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke: selected ? "#4f46e5" : "#94a3b8",
          strokeWidth: selected ? 2.5 : 1.75,
        }}
      />
      {showGrad && localDer !== 0 && (
        <EdgeLabelRenderer>
          <div
            className="ag-edge-label"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            }}
          >
            ∂ {edgeFmt(localDer)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const autogradEdgeTypes = {
  autograd: AutogradFlowEdge,
};
