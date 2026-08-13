import { useCallback, useRef, useState } from "react";
import { AUTOGRAD_PRESETS, CompGraphEngine, type AutogradOp, type AutogradPresetId, type CompGraphConfig } from "@ml-vis/core/autograd";
import { AutogradFlowGraph } from "./AutogradFlowGraph";
import { AutogradPalette } from "./AutogradPalette";
import { AutogradInspector } from "./AutogradInspector";
import { useAutogradMessages } from "./messages";

export interface ComputationalGraphPlaygroundProps {
  initialConfig?: Partial<CompGraphConfig>;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "∞";
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(3);
}

export function ComputationalGraphPlayground({
  initialConfig,
  toolbarStart,
  toolbarEnd,
}: ComputationalGraphPlaygroundProps) {
  const engineRef = useRef<CompGraphEngine>(undefined as unknown as CompGraphEngine);
  if (!engineRef.current) {
    engineRef.current = new CompGraphEngine(initialConfig);
  }
  const engine = engineRef.current;
  const t = useAutogradMessages();

  const [version, setVersion] = useState(0);
  const [fitKey, setFitKey] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [preset, setPreset] = useState<AutogradPresetId>(engine.config.preset);
  // Gradients are revealed only after the user runs the backward pass; any edit
  // or a fresh forward pass invalidates them until Backward is pressed again.
  const [gradsVisible, setGradsVisible] = useState(false);
  // Computed op-node values are revealed only after the user runs a pass; on
  // initial load, reset, or preset change the graph shows leaf values alone.
  const [valuesVisible, setValuesVisible] = useState(false);
  const addSlotRef = useRef(0);

  const bump = useCallback(() => setVersion((v) => v + 1), []);

  const nextDropPosition = useCallback(() => {
    const n = addSlotRef.current++;
    return { x: 60 + (n % 4) * 130, y: 40 + Math.floor(n / 4) * 90 };
  }, []);

  // A structural/value edit keeps leaf scalars in place but stales the computed
  // op-node values and gradients — both stay hidden until the next pass.
  const afterEdit = useCallback(() => {
    engine.recompute();
    setValuesVisible(false);
    setGradsVisible(false);
    bump();
  }, [engine, bump]);

  const onAddOp = useCallback(
    (op: AutogradOp) => {
      engine.addPaletteNode(op, nextDropPosition());
      afterEdit();
    },
    [engine, afterEdit, nextDropPosition],
  );

  const onConnect = useCallback((s: string, tId: string) => {
    engine.connectNodes(s, tId);
    afterEdit();
  }, [engine, afterEdit]);

  const onDropNode = useCallback((kind: string, position: { x: number; y: number }) => {
    addSlotRef.current += 1;
    engine.addPaletteNode(kind as never, position);
    afterEdit();
  }, [engine, afterEdit]);

  const onMoveNode = useCallback((id: string, position: { x: number; y: number }) => {
    engine.setNodePosition(id, position);
  }, [engine]);

  const onRemoveNode = useCallback((id: string) => {
    engine.removeNode(id);
    setSelectedNodeId(null);
    afterEdit();
  }, [engine, afterEdit]);

  const onRemoveEdge = useCallback((id: string) => {
    engine.removeEdge(id);
    setSelectedEdgeId(null);
    afterEdit();
  }, [engine, afterEdit]);

  const onSetNodeValue = useCallback((id: string, value: number) => {
    engine.setNodeValue(id, value);
    afterEdit();
  }, [engine, afterEdit]);

  const onSetOutput = useCallback((id: string) => {
    engine.setOutput(id);
    afterEdit();
  }, [engine, afterEdit]);

  const onForward = useCallback(() => {
    engine.runForward();
    setValuesVisible(true);
    setGradsVisible(false);
    bump();
  }, [engine, bump]);

  const onBackward = useCallback(() => {
    engine.runBackward();
    setValuesVisible(true);
    setGradsVisible(true);
    bump();
  }, [engine, bump]);

  const onReset = useCallback(() => {
    engine.reset();
    addSlotRef.current = 0;
    setValuesVisible(false);
    setGradsVisible(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setFitKey((k) => k + 1);
    bump();
  }, [engine, bump]);

  const onPresetChange = useCallback((next: AutogradPresetId) => {
    engine.loadPreset(next);
    addSlotRef.current = 0;
    setPreset(next);
    setValuesVisible(false);
    setGradsVisible(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setFitKey((k) => k + 1);
    bump();
  }, [engine, bump]);

  const presetLabels: Record<AutogradPresetId, string> = {
    expr: t.presetExpr,
    neuron: t.presetNeuron,
  };

  const outputNode = engine.graph.getNode(engine.graph.outputId);

  return (
    <div className="nn-playground nn-playground--immersive">
      <div className="nn-immersive-toolbar">
        <div className="nn-toolbar-group nn-toolbar-group--actions">
          {toolbarStart}
          <button type="button" className="nn-btn nn-btn--ghost" onClick={onReset}>
            {t.reset}
          </button>
          <button type="button" className="nn-btn nn-btn--primary" onClick={onForward}>
            {t.forward}
          </button>
          <button type="button" className="nn-btn nn-btn--secondary" onClick={onBackward}>
            {t.backward}
          </button>
          <label className="network-training-field ag-toolbar-preset">
            <span className="network-training-label">{t.preset}</span>
            <select
              className="nn-select nn-select--dock"
              value={preset}
              onChange={(e) => onPresetChange(e.target.value as AutogradPresetId)}
            >
              {AUTOGRAD_PRESETS.map((id) => (
                <option key={id} value={id}>
                  {presetLabels[id]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="nn-toolbar-group nn-toolbar-group--params">
          <div className="nn-toolbar-stat">
            <span className="label">{t.value}</span>
            <span className="value">
              {valuesVisible && outputNode ? fmt(outputNode.value) : "—"}
            </span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="nn-immersive-body">
        <AutogradFlowGraph
          graph={engine.graph}
          version={version}
          showGrad={gradsVisible}
          showValues={valuesVisible}
          fitViewKey={`${preset}:${fitKey}`}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onSelectNode={setSelectedNodeId}
          onSelectEdge={setSelectedEdgeId}
          onConnect={onConnect}
          onDropNode={onDropNode}
          onMoveNode={onMoveNode}
          onRemoveNode={onRemoveNode}
          onRemoveEdge={onRemoveEdge}
        >
          <aside className="nn-flow-dock nn-flow-dock--left">
            <AutogradPalette onAddOp={onAddOp} />
            <AutogradInspector
              graph={engine.graph}
              version={version}
              showGrad={gradsVisible}
              showValues={valuesVisible}
              selectedNodeId={selectedNodeId}
              selectedEdgeId={selectedEdgeId}
              onSetNodeValue={onSetNodeValue}
              onSetOutput={onSetOutput}
              onRemoveNode={onRemoveNode}
              onRemoveEdge={onRemoveEdge}
            />
          </aside>
        </AutogradFlowGraph>
      </div>
    </div>
  );
}
