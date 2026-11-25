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
import {
  gradColor,
  type AutogradEdgeData,
  type AutogradNodeData,
  type AutogradStepRole,
} from "./adapter";
import { useAutogradMessages } from "./messages";

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "∞";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001) return value.toExponential(1);
  return value.toFixed(Math.abs(value) < 1 ? 3 : 2);
}

function nodeClass(base: string, data: AutogradNodeData): string {
  const step = data.step ? ` ag-node--${data.step.state}` : "";
  return `${base}${data.selected ? " selected" : ""}${step}`;
}

/** Tooltip naming what a step does, or will do, to a node. */
function useStepTitle(step: AutogradStepRole | null): string | undefined {
  const t = useAutogradMessages();
  if (!step) return undefined;
  const titles = {
    active: { forward: t.stepActiveForward, backward: t.stepActiveBackward },
    next: { forward: t.stepNextForward, backward: t.stepNextBackward },
  };
  return titles[step.state][step.pass];
}

function NodeBody({ data }: { data: AutogradNodeData }) {
  return (
    <>
      <div className="ag-node-head">
        <span className="ag-node-symbol">{data.symbol}</span>
        <span className="ag-node-label">{data.label}</span>
      </div>
      <div className="ag-node-stats">
        {data.computed && (
          <span className="ag-node-value" title="value">{fmt(data.value)}</span>
        )}
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
  const stepTitle = useStepTitle(data.step);
  return (
    <div className={nodeClass(`ag-node ag-node--leaf ag-node--${data.op}`, data)} title={stepTitle}>
      <NodeBody data={data} />
      <Handle type="source" position={Position.Right} className="ag-handle" />
    </div>
  );
}

export function OpFlowNode({ data }: NodeProps<Node<AutogradNodeData>>) {
  const stepTitle = useStepTitle(data.step);
  return (
    <div className={nodeClass("ag-node ag-node--op", data)} title={stepTitle}>
      <Handle type="target" position={Position.Left} className="ag-handle" />
      <NodeBody data={data} />
      <Handle type="source" position={Position.Right} className="ag-handle" />
    </div>
  );
}

export function SinkFlowNode({ data }: NodeProps<Node<AutogradNodeData>>) {
  const stepTitle = useStepTitle(data.step);
  return (
    <div className={nodeClass("ag-node ag-node--op ag-node--sink", data)} title={stepTitle}>
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
  const hasLocalDer = data?.hasLocalDer ?? false;
  const stepInput = data?.stepInput ?? false;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke: selected ? "#4f46e5" : stepInput ? "var(--nn-crimson)" : "#94a3b8",
          strokeWidth: selected || stepInput ? 2.5 : 1.75,
        }}
      />
      {hasLocalDer && localDer !== 0 && (
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
