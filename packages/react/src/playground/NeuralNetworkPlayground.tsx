import { memo, useCallback, useEffect, useRef, useState, type RefObject } from "react";
import {
  CLASS_0_HEX,
  CLASS_1_HEX,
  DEFAULT_TF_CONFIG,
  PlaygroundEngine,
  TF_DATASETS,
  type TfDatasetId,
  type TfPlaygroundConfig,
} from "@ml-vis/core";
import { NetworkInspector } from "./network/NetworkInspector";
import { NetworkArchitecturePanel } from "./network/NetworkArchitecturePanel";
import { NetworkPalette } from "./network/NetworkPalette";
import { NetworkTrainingPanel } from "./network/NetworkTrainingPanel";
import { ReactFlowNetworkGraph } from "./network/ReactFlowNetworkGraph";
import { paintAllBoundaries, paintAllBoundariesAfterCommit, paintBoundaryNode } from "./network/boundaryPaint";
import type { TrainingStats } from "./network/NetworkBoundaryContext";
import { useNetworkMessages } from "./network/messages";

function DatasetThumbnail({
  datasetId,
  label,
  selected,
  onSelect,
}: {
  datasetId: TfDatasetId;
  label: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const size = 128;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);
    const points = TF_DATASETS[datasetId](120, 0);
    for (const p of points) {
      const px = ((p.x + 6) / 12) * size;
      const py = ((6 - p.y) / 12) * size;
      ctx.beginPath();
      ctx.arc(px, py, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
      ctx.fill();
    }
  }, [datasetId]);

  return (
    <div className={`dataset ${selected ? "selected" : ""}`}>
      <button
        type="button"
        className={`data-thumbnail-btn ${selected ? "selected" : ""}`}
        onClick={onSelect}
        aria-label={label}
        aria-pressed={selected}
      >
        <canvas ref={canvasRef} className="data-thumbnail" />
        {selected && <span className="data-thumbnail-check" aria-hidden="true">✓</span>}
      </button>
      <span className="dataset-label">{label}</span>
    </div>
  );
}

export interface NeuralNetworkPlaygroundProps {
  initialConfig?: Partial<TfPlaygroundConfig>;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

type GraphPaneProps = {
  engineRef: RefObject<PlaygroundEngine | null>;
  boundaryRef: RefObject<Record<string, number[][]>>;
  statsRef: RefObject<TrainingStats>;
  trainingLiveRef: RefObject<boolean>;
  paintGeneration: number;
  tick: number;
  playing: boolean;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  refitViewKey: number;
  onSelectNode: (nodeId: string | null) => void;
  onSelectEdge: (edgeId: string | null) => void;
  onToggleFeature: (id: string) => void;
  onConnect: (sourceId: string, targetId: string) => void;
  onDropNode: (kind: Parameters<PlaygroundEngine["addPaletteNode"]>[0], position: { x: number; y: number }) => void;
  onMoveNode: (nodeId: string, position: { x: number; y: number }) => void;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (sourceId: string, targetId: string) => void;
  onAddLayer: () => void;
  onRemoveLayer: () => void;
  onAddNeuron: (layerIdx: number) => void;
  onRemoveNeuron: (layerIdx: number) => void;
  onNormalizeLayout: () => void;
  onLearningRateChange: (learningRate: number) => void;
  onActivationChange: (activation: TfPlaygroundConfig["activation"]) => void;
  onBatchSizeChange: (batchSize: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (percTrainData: number) => void;
  onDiscretizeChange: (discretize: boolean) => void;
  onRegenerateData: () => void;
  onDatasetChange: (dataset: TfDatasetId) => void;
  onResetWeights: () => void;
};

const NetworkGraphPane = memo(function NetworkGraphPane({
  engineRef,
  boundaryRef,
  statsRef,
  trainingLiveRef,
  paintGeneration,
  tick,
  playing,
  selectedNodeId,
  selectedEdgeId,
  refitViewKey,
  onSelectNode,
  onSelectEdge,
  onToggleFeature,
  onConnect,
  onDropNode,
  onMoveNode,
  onRemoveNode,
  onRemoveEdge,
  onAddLayer,
  onRemoveLayer,
  onAddNeuron,
  onRemoveNeuron,
  onNormalizeLayout,
  onLearningRateChange,
  onActivationChange,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onDiscretizeChange,
  onRegenerateData,
  onDatasetChange,
  onResetWeights,
}: GraphPaneProps) {
  const engine = engineRef.current!;
  const cfg = engine.config;
  const t = useNetworkMessages();
  const fitViewKey = engine.graph.inputIds.join(",");

  return (
    <ReactFlowNetworkGraph
      graph={engine.graph}
      enabledFeatures={cfg.enabledFeatures}
      trainData={engine.trainData}
      lossTest={engine.lossTest}
      lossTrain={engine.lossTrain}
      fillHeight
      trainingLive={playing}
      trainingLiveRef={trainingLiveRef}
      paintGeneration={paintGeneration}
      boundaryRef={boundaryRef}
      statsRef={statsRef}
      selectedNodeId={selectedNodeId}
      selectedEdgeId={selectedEdgeId}
      onSelectNode={onSelectNode}
      onSelectEdge={onSelectEdge}
      onToggleFeature={onToggleFeature}
      onConnect={onConnect}
      onDropNode={onDropNode}
      onMoveNode={onMoveNode}
      onRemoveNode={onRemoveNode}
      onRemoveEdge={onRemoveEdge}
      fitViewKey={fitViewKey}
      refitViewKey={refitViewKey}
      layoutKey={tick}
      discretize={cfg.discretize}
    >
      <aside className="tf-flow-dock tf-flow-dock--left">
        <NetworkPalette />
        <NetworkArchitecturePanel
          numHiddenLayers={cfg.numHiddenLayers}
          networkShape={cfg.networkShape}
          onAddLayer={onAddLayer}
          onRemoveLayer={onRemoveLayer}
          onAddNeuron={onAddNeuron}
          onRemoveNeuron={onRemoveNeuron}
        />
        <div className="tf-flow-dock-section">
          <button type="button" className="tf-btn tf-btn--secondary tf-reset-weights" onClick={onResetWeights}>
            {t.resetWeights}
          </button>
          <button type="button" className="tf-btn tf-btn--secondary tf-layout-normalize" onClick={onNormalizeLayout}>
            <span className="tf-layout-normalize-icon" aria-hidden="true">⊞</span>
            {t.arrangeLayout}
          </button>
        </div>
        <div className="tf-flow-dock-section">
          <h4 className="tf-flow-dock-title">{t.dataset}</h4>
          <div className="dataset-list dataset-list--compact">
            {(["circle", "xor", "gauss", "spiral"] as TfDatasetId[]).map((id) => (
              <DatasetThumbnail
                key={id}
                datasetId={id}
                label={t.datasetLabels[id]}
                selected={cfg.dataset === id}
                onSelect={() => onDatasetChange(id)}
              />
            ))}
          </div>
        </div>
        <NetworkInspector
          graph={engine.graph}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onRemoveNode={onRemoveNode}
          onRemoveEdge={(edgeId) => {
            const link = engine.graph.getAllLinks().find((l) => l.id === edgeId);
            if (link) onRemoveEdge(link.source.id, link.dest.id);
          }}
        />
      </aside>
      <aside className="tf-flow-dock tf-flow-dock--right">
        <NetworkTrainingPanel
          learningRate={cfg.learningRate}
          activation={cfg.activation}
          batchSize={cfg.batchSize}
          noise={cfg.noise}
          percTrainData={cfg.percTrainData}
          discretize={cfg.discretize ?? false}
          onLearningRateChange={onLearningRateChange}
          onActivationChange={onActivationChange}
          onBatchSizeChange={onBatchSizeChange}
          onNoiseChange={onNoiseChange}
          onTrainRatioChange={onTrainRatioChange}
          onDiscretizeChange={onDiscretizeChange}
          onRegenerateData={onRegenerateData}
        />
      </aside>
    </ReactFlowNetworkGraph>
  );
});

export function NeuralNetworkPlayground({ initialConfig, toolbarStart, toolbarEnd }: NeuralNetworkPlaygroundProps) {
  // Lazy init: the constructor runs data generation + boundary computation,
  // so it must not execute on every render.
  const engineRef = useRef<PlaygroundEngine>(undefined as unknown as PlaygroundEngine);
  if (!engineRef.current) {
    engineRef.current = new PlaygroundEngine(initialConfig ?? DEFAULT_TF_CONFIG);
  }
  const t = useNetworkMessages();
  const boundaryRef = useRef(engineRef.current.boundary);
  const statsRef = useRef<TrainingStats>({
    epoch: engineRef.current.epoch,
    lossTrain: engineRef.current.lossTrain,
    lossTest: engineRef.current.lossTest,
  });

  const [tick, setTick] = useState(0);
  const [refitViewKey, setRefitViewKey] = useState(0);
  const [stats, setStats] = useState<TrainingStats>(() => ({
    epoch: engineRef.current.epoch,
    lossTrain: engineRef.current.lossTrain,
    lossTest: engineRef.current.lossTest,
  }));
  const bump = useCallback(() => setTick((n) => n + 1), []);
  const [paintGeneration, setPaintGeneration] = useState(0);
  const requestPaint = useCallback(() => setPaintGeneration((n) => n + 1), []);

  useEffect(() => {
    paintAllBoundariesAfterCommit();
  }, [paintGeneration, tick]);

  const [playing, setPlaying] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const wasPlayingRef = useRef(false);
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;

  const syncRuntimeRefs = useCallback(() => {
    const engine = engineRef.current;
    boundaryRef.current = engine.boundary;
    const next = { epoch: engine.epoch, lossTrain: engine.lossTrain, lossTest: engine.lossTest };
    statsRef.current = next;
    setStats(next);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const engine = engineRef.current;
    engine.refreshOutputBoundaryFast();
    boundaryRef.current = engine.boundary;
    paintAllBoundaries();
  }, [playing]);

  useEffect(() => {
    if (!playing) return;
    /** Target training rate during Play (~60–90 epochs/s). */
    const epochsPerSec = 75;
    const maxEpochsPerFrame = 2;
    let tick = 0;
    let raf = 0;
    let lastTime = performance.now();
    let lastPaint = 0;
    let epochBank = 0;
    const paintIntervalMs = 1000 / 30;
    const loop = (now: number) => {
      const engine = engineRef.current;
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      epochBank = Math.min(epochBank + dt * epochsPerSec, maxEpochsPerFrame);
      const steps = Math.floor(epochBank);
      epochBank -= steps;

      if (steps > 0) {
        for (let i = 0; i < steps; i++) engine.trainEpoch(false);
        if (now - lastPaint >= paintIntervalMs) {
          // Hidden nodes are coarse (10×10), so refreshing them every paint
          // frame is cheap and keeps them in sync with the output node.
          engine.refreshOutputBoundaryFast();
          engine.refreshHiddenBoundariesFast();
          boundaryRef.current = engine.boundary;
          // Imperative paint only — requestPaint() during Play re-renders React Flow and wipes canvases.
          paintBoundaryNode(engine.outputNodeId);
          paintAllBoundaries();
          lastPaint = now;
        }
      }

      tick++;
      if (steps > 0 && tick % 16 === 0) {
        engine.lossTrain = engine.getLoss(engine.trainData);
      }
      if (steps > 0 && tick % 32 === 0) {
        engine.lossTest = engine.getLoss(engine.testData);
      }

      statsRef.current = {
        epoch: engine.epoch,
        lossTrain: engine.lossTrain,
        lossTest: engine.lossTest,
      };
      if (tick % 8 === 0) {
        setStats({ ...statsRef.current });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    if (wasPlayingRef.current && !playing) {
      const engine = engineRef.current;
      engine.refreshMetrics();
      engine.refreshBoundary();
      boundaryRef.current = engine.boundary;
      statsRef.current = {
        epoch: engine.epoch,
        lossTrain: engine.lossTrain,
        lossTest: engine.lossTest,
      };
      setStats({ ...statsRef.current });
      requestPaint();
      bump();
    }
    wasPlayingRef.current = playing;
  }, [playing, bump, requestPaint]);

  const reset = () => {
    engineRef.current.resetToInitial();
    setPlaying(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setRefitViewKey((n) => n + 1);
    syncRuntimeRefs();
    requestPaint();
    bump();
  };

  const resetWeights = () => {
    engineRef.current.resetWeights();
    setPlaying(false);
    syncRuntimeRefs();
    requestPaint();
    bump();
  };

  const step = () => {
    engineRef.current.step();
    syncRuntimeRefs();
    paintAllBoundariesAfterCommit();
    bump();
    requestPaint();
  };

  // Hyperparameters are read live by trainEpoch — no engine reset needed.
  const onLearningRateChange = useCallback((learningRate: number) => {
    engineRef.current.config.learningRate = learningRate;
    bump();
  }, [bump]);

  const onBatchSizeChange = useCallback((batchSize: number) => {
    engineRef.current.config.batchSize = batchSize;
    bump();
  }, [bump]);

  const onActivationChange = useCallback((activation: TfPlaygroundConfig["activation"]) => {
    engineRef.current.setActivation(activation);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onDatasetChange = useCallback((dataset: TfDatasetId) => {
    engineRef.current.setDataset(dataset);
    setPlaying(false);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onToggleFeature = useCallback((id: string) => {
    engineRef.current.toggleFeature(id);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onConnect = useCallback((sourceId: string, targetId: string) => {
    engineRef.current.connectNodes(sourceId, targetId);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onDropNode = useCallback((kind: Parameters<PlaygroundEngine["addPaletteNode"]>[0], position: { x: number; y: number }) => {
    engineRef.current.addPaletteNode(kind, position);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onMoveNode = useCallback((nodeId: string, position: { x: number; y: number }) => {
    engineRef.current.setNodePosition(nodeId, position);
  }, []);

  const onRemoveNode = useCallback((nodeId: string) => {
    engineRef.current.removeGraphNode(nodeId);
    setSelectedNodeId(null);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onRemoveEdge = useCallback((sourceId: string, targetId: string) => {
    engineRef.current.disconnectNodes(sourceId, targetId);
    setSelectedEdgeId(null);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onAddLayer = useCallback(() => {
    engineRef.current.addLayer();
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onRemoveLayer = useCallback(() => {
    engineRef.current.removeLayer();
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onAddNeuron = useCallback((layerIdx: number) => {
    engineRef.current.addNeuron(layerIdx);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onRemoveNeuron = useCallback((layerIdx: number) => {
    engineRef.current.removeNeuron(layerIdx);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onNormalizeLayout = useCallback(() => {
    engineRef.current.normalizeLayout();
    setRefitViewKey((n) => n + 1);
    bump();
  }, [bump]);

  const onDiscretizeChange = useCallback((discretize: boolean) => {
    engineRef.current.config.discretize = discretize;
    requestPaint();
    bump();
  }, [bump, requestPaint]);

  const onRegenerateData = useCallback(() => {
    engineRef.current.regenerateData();
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  // Data params only regenerate the dataset; the network keeps its progress.
  const onNoiseChange = useCallback((noise: number) => {
    engineRef.current.updateDataParams({ noise });
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onTrainRatioChange = useCallback((percTrainData: number) => {
    engineRef.current.updateDataParams({ percTrainData });
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  return (
    <div className="tf-playground tf-playground--immersive">
      <div className="tf-immersive-toolbar">
        <div className="tf-toolbar-group tf-toolbar-group--actions">
          {toolbarStart}
          <button type="button" className="tf-btn tf-btn--ghost" onClick={reset}>
            {t.reset}
          </button>
          <button
            type="button"
            className={`tf-btn tf-btn--primary${playing ? " playing" : ""}`}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? t.pause : t.play}
          </button>
          <button type="button" className="tf-btn tf-btn--secondary" onClick={step}>
            {t.step}
          </button>
        </div>

        <div className="tf-toolbar-group tf-toolbar-group--params">
          <div className="tf-toolbar-stat">
            <span className="label">{t.epoch}</span>
            <span className="value">{stats.epoch.toLocaleString()}</span>
          </div>
          <div className="tf-toolbar-stat">
            <span className="label">{t.testLoss}</span>
            <span className="value" id="loss-test">
              {stats.lossTest.toFixed(3)}
            </span>
          </div>
          <div className="tf-toolbar-stat tf-toolbar-stat--train">
            <span className="label">{t.trainLoss}</span>
            <span className="value" id="loss-train">
              {stats.lossTrain.toFixed(3)}
            </span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="tf-immersive-body">
        <NetworkGraphPane
          engineRef={engineRef}
          boundaryRef={boundaryRef}
          statsRef={statsRef}
          trainingLiveRef={trainingLiveRef}
          paintGeneration={paintGeneration}
          tick={tick}
          playing={playing}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          refitViewKey={refitViewKey}
          onSelectNode={setSelectedNodeId}
          onSelectEdge={setSelectedEdgeId}
          onToggleFeature={onToggleFeature}
          onConnect={onConnect}
          onDropNode={onDropNode}
          onMoveNode={onMoveNode}
          onRemoveNode={onRemoveNode}
          onRemoveEdge={onRemoveEdge}
          onAddLayer={onAddLayer}
          onRemoveLayer={onRemoveLayer}
          onAddNeuron={onAddNeuron}
          onRemoveNeuron={onRemoveNeuron}
          onNormalizeLayout={onNormalizeLayout}
          onLearningRateChange={onLearningRateChange}
          onActivationChange={onActivationChange}
          onBatchSizeChange={onBatchSizeChange}
          onNoiseChange={onNoiseChange}
          onTrainRatioChange={onTrainRatioChange}
          onDiscretizeChange={onDiscretizeChange}
          onRegenerateData={onRegenerateData}
          onDatasetChange={onDatasetChange}
          onResetWeights={resetWeights}
        />
      </div>
    </div>
  );
}
