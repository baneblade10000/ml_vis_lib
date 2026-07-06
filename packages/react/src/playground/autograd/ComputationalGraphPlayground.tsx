import { useCallback, useRef, useState } from "react";
import {
  AUTOGRAD_PRESETS,
  CompGraphEngine,
  type AutogradPresetId,
  type CompGraphConfig,
} from "@ml-vis/core";
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

  const bump = useCallback(() => setVersion((v) => v + 1), []);

  // A structural/value edit keeps values fresh (forward) but stales gradients.
  const afterEdit = useCallback(() => {
    engine.recompute();
    setGradsVisible(false);
    bump();
  }, [engine, bump]);

  const onConnect = useCallback((s: string, tId: string) => {
    engine.connectNodes(s, tId);
    afterEdit();
  }, [engine, afterEdit]);

  const onDropNode = useCallback((kind: string, position: { x: number; y: number }) => {
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
    setGradsVisible(false);
    bump();
  }, [engine, bump]);

  const onBackward = useCallback(() => {
    engine.runBackward();
    setGradsVisible(true);
    bump();
  }, [engine, bump]);

  const onReset = useCallback(() => {
    engine.reset();
    engine.runForward();
    setGradsVisible(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setFitKey((k) => k + 1);
    bump();
  }, [engine, bump]);

  const onPresetChange = useCallback((next: AutogradPresetId) => {
    engine.loadPreset(next);
    engine.runForward();
    setPreset(next);
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
    <div className="tf-playground tf-playground--immersive">
      <div className="tf-immersive-toolbar">
        <div className="tf-toolbar-group tf-toolbar-group--actions">
          {toolbarStart}
          <button type="button" className="tf-btn tf-btn--ghost" onClick={onReset}>
            {t.reset}
          </button>
          <button type="button" className="tf-btn tf-btn--primary" onClick={onForward}>
            {t.forward}
          </button>
          <button type="button" className="tf-btn tf-btn--secondary" onClick={onBackward}>
            {t.backward}
          </button>
          <label className="tf-training-field ag-toolbar-preset">
            <span className="tf-training-label">{t.preset}</span>
            <select
              className="tf-select tf-select--dock"
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

        <div className="tf-toolbar-group tf-toolbar-group--params">
          <div className="tf-toolbar-stat">
            <span className="label">{t.value}</span>
            <span className="value">{outputNode ? fmt(outputNode.value) : "—"}</span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="tf-immersive-body">
        <AutogradFlowGraph
          graph={engine.graph}
          version={version}
          showGrad={gradsVisible}
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
          <aside className="tf-flow-dock tf-flow-dock--left">
            <AutogradPalette />
            <AutogradInspector
              graph={engine.graph}
              version={version}
              showGrad={gradsVisible}
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
