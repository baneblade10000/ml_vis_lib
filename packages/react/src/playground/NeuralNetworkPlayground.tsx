import { memo, useCallback, useEffect, useRef, useState, type RefObject } from "react";
import {
  CLASS_0_HEX,
  CLASS_1_HEX,
  DEFAULT_TF_CONFIG,
  PlaygroundEngine,
  TF_DATASETS,
  TF_DATASETS_1D_CLASSIFICATION,
  TF_DATASETS_1D_REGRESSION,
  valueToRgb,
  type Dataset1DId,
  type LossHistoryPoint,
  type TfAnyDatasetId,
  type TfDataMode,
  type TfDatasetId,
  type TfPlaygroundConfig,
  type TfProblemType,
} from "@ml-vis/core";
import { NetworkInspector } from "./network/NetworkInspector";
import { NetworkArchitecturePanel } from "./network/NetworkArchitecturePanel";
import { NetworkPalette } from "./network/NetworkPalette";
import { NetworkDataPanel } from "./network/NetworkDataPanel";
import { NetworkTrainingPanel } from "./network/NetworkTrainingPanel";
import { NetworkLossChart } from "./network/NetworkLossChart";
import { ReactFlowNetworkGraph } from "./network/ReactFlowNetworkGraph";
import { paintAllBoundaries, paintAllBoundariesAfterCommit, paintBoundaryNode } from "./network/boundaryPaint";
import type { CurveStore, TrainingStats } from "./network/NetworkBoundaryContext";
import type { EdgeVizMode } from "./network/graphAdapter";
import { useNetworkMessages } from "./network/messages";

const DATASETS_2D_CLASSIFICATION: TfDatasetId[] = ["circle", "xor", "gauss", "spiral"];
const DATASETS_2D_REGRESSION: TfDatasetId[] = ["sinSin"];
const DATASETS_1D_CLASSIFICATION: Dataset1DId[] = ["gauss1d", "threshold", "twoClusters"];
const DATASETS_1D_REGRESSION: Dataset1DId[] = ["sine", "linear", "cubic", "step"];

function DatasetThumbnail({
  datasetId,
  label,
  selected,
  onSelect,
  mode,
  problemType = "classification",
}: {
  datasetId: TfAnyDatasetId;
  label: string;
  selected: boolean;
  onSelect: () => void;
  mode: TfDataMode;
  problemType?: TfProblemType;
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

    if (mode === "1d") {
      const gen =
        datasetId in TF_DATASETS_1D_CLASSIFICATION
          ? TF_DATASETS_1D_CLASSIFICATION[datasetId as keyof typeof TF_DATASETS_1D_CLASSIFICATION]
          : TF_DATASETS_1D_REGRESSION[datasetId as keyof typeof TF_DATASETS_1D_REGRESSION];
      if (!gen) return;
      const points = gen(80, 0);
      // Fit Y to the actual series — a fixed [-1.5, 1.5] clips linear/cubic hard.
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const p of points) {
        if (p.y < yMin) yMin = p.y;
        if (p.y > yMax) yMax = p.y;
      }
      if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
        yMin = -1.2;
        yMax = 1.2;
      } else {
        const pad = (yMax - yMin) * 0.12;
        yMin -= pad;
        yMax += pad;
      }
      // Keep zero in view when the series crosses / sits near it.
      yMin = Math.min(yMin, -0.15);
      yMax = Math.max(yMax, 0.15);
      const mapY = (y: number) => ((yMax - y) / (yMax - yMin)) * size;
      const isRegression = problemType === "regression";

      ctx.strokeStyle = "rgba(0,0,0,0.12)";
      ctx.beginPath();
      ctx.moveTo(0, mapY(0));
      ctx.lineTo(size, mapY(0));
      ctx.stroke();
      for (const p of points) {
        const px = ((p.x + 6) / 12) * size;
        const py = mapY(p.y);
        ctx.beginPath();
        ctx.arc(px, py, 2.4, 0, Math.PI * 2);
        // Regression has no classes — one neutral color. Classification keeps ±1 hues.
        ctx.fillStyle = isRegression
          ? "rgb(100, 116, 139)"
          : p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
        ctx.fill();
      }
      return;
    }

    const gen = TF_DATASETS[datasetId as TfDatasetId];
    if (!gen) return;
    const isRegression = problemType === "regression";
    if (isRegression) {
      // Continuous labels as sparse dots are unreadable in a tiny thumb —
      // paint a dense target-field heatmap instead (same palette as the canvas).
      const side = 48;
      const cell = size / side;
      const field = gen(side * side, 0);
      for (const p of field) {
        const { r, g, b } = valueToRgb(p.label);
        const px = ((p.x + 6) / 12) * size;
        const py = ((6 - p.y) / 12) * size;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(px - cell / 2, py - cell / 2, cell + 0.6, cell + 0.6);
      }
      return;
    }
    const points = gen(120, 0);
    for (const p of points) {
      const px = ((p.x + 6) / 12) * size;
      const py = ((6 - p.y) / 12) * size;
      ctx.beginPath();
      ctx.arc(px, py, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
      ctx.fill();
    }
  }, [datasetId, mode, problemType]);

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
  curvesRef: RefObject<CurveStore>;
  targetCurveRef: RefObject<number[] | null>;
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
  onOptimizerChange: (optimizer: TfPlaygroundConfig["optimizer"]) => void;
  onActivationChange: (activation: TfPlaygroundConfig["activation"]) => void;
  onWeightInitChange: (weightInit: TfPlaygroundConfig["weightInit"]) => void;
  onRegularizationChange: (regularization: TfPlaygroundConfig["regularization"]) => void;
  onRegularizationRateChange: (regularizationRate: number) => void;
  onBatchSizeChange: (batchSize: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (percTrainData: number) => void;
  onDiscretizeChange: (discretize: boolean) => void;
  onRegenerateData: () => void;
  onDatasetChange: (dataset: TfAnyDatasetId) => void;
  onProblemTypeChange: (problemType: TfProblemType) => void;
  onResetWeights: () => void;
  edgeVizMode: EdgeVizMode;
  onEdgeVizModeChange: (mode: EdgeVizMode) => void;
  /** Snapshot of loss history — new array reference when the curve should redraw. */
  lossHistory: LossHistoryPoint[];
  lossTrain: number;
  lossTest: number;
};

const NetworkGraphPane = memo(function NetworkGraphPane({
  engineRef,
  boundaryRef,
  curvesRef,
  targetCurveRef,
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
  onOptimizerChange,
  onActivationChange,
  onWeightInitChange,
  onRegularizationChange,
  onRegularizationRateChange,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onDiscretizeChange,
  onRegenerateData,
  onDatasetChange,
  onProblemTypeChange,
  onResetWeights,
  edgeVizMode,
  onEdgeVizModeChange,
  lossHistory,
  lossTrain,
  lossTest,
}: GraphPaneProps) {
  const engine = engineRef.current!;
  const cfg = engine.config;
  const t = useNetworkMessages();
  const fitViewKey = engine.graph.inputIds.join(",");
  const datasets1d =
    cfg.problemType === "regression" ? DATASETS_1D_REGRESSION : DATASETS_1D_CLASSIFICATION;
  const datasets2d =
    cfg.problemType === "regression" ? DATASETS_2D_REGRESSION : DATASETS_2D_CLASSIFICATION;
  const datasetIds = cfg.dataMode === "1d" ? datasets1d : datasets2d;

  return (
    <ReactFlowNetworkGraph
      graph={engine.graph}
      enabledFeatures={cfg.enabledFeatures}
      dataMode={cfg.dataMode}
      problemType={cfg.problemType}
      trainData={engine.trainData}
      lossTest={engine.lossTest}
      lossTrain={engine.lossTrain}
      fillHeight
      trainingLive={playing}
      trainingLiveRef={trainingLiveRef}
      paintGeneration={paintGeneration}
      edgeVizMode={edgeVizMode}
      learningRate={cfg.learningRate}
      boundaryRef={boundaryRef}
      curvesRef={curvesRef}
      targetCurveRef={targetCurveRef}
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
          <h4 className="tf-flow-dock-title">{t.edgeViz}</h4>
          <div className="tf-flat-switch" role="group" aria-label={t.edgeViz}>
            {([
              ["weight", t.edgeVizWeight],
              ["gradient", t.edgeVizGradient],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={`tf-flat-switch__btn${edgeVizMode === id ? " selected" : ""}`}
                onClick={() => onEdgeVizModeChange(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="tf-flow-dock-section">
          <h4 className="tf-flow-dock-title">{t.problemType}</h4>
          <div className="tf-flat-switch" role="group" aria-label={t.problemType}>
            {(["classification", "regression"] as TfProblemType[]).map((id) => (
              <button
                key={id}
                type="button"
                className={`tf-flat-switch__btn${cfg.problemType === id ? " selected" : ""}`}
                onClick={() => onProblemTypeChange(id)}
              >
                {t.problemTypeLabels[id]}
              </button>
            ))}
          </div>
        </div>
        <div className="tf-flow-dock-section">
          <h4 className="tf-flow-dock-title">{t.dataset}</h4>
          <div className="dataset-list dataset-list--compact">
            {datasetIds.map((id) => (
              <DatasetThumbnail
                key={id}
                datasetId={id}
                label={
                  cfg.dataMode === "1d"
                    ? t.dataset1dLabels[id as Dataset1DId]
                    : t.datasetLabels[id as TfDatasetId]
                }
                selected={cfg.dataset === id}
                onSelect={() => onDatasetChange(id)}
                mode={cfg.dataMode}
                problemType={cfg.problemType}
              />
            ))}
          </div>
          <NetworkDataPanel
            batchSize={cfg.batchSize}
            noise={cfg.noise}
            percTrainData={cfg.percTrainData}
            onBatchSizeChange={onBatchSizeChange}
            onNoiseChange={onNoiseChange}
            onTrainRatioChange={onTrainRatioChange}
            onRegenerateData={onRegenerateData}
          />
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
        <NetworkLossChart
          history={lossHistory}
          title={t.learningCurve}
          trainLabel={t.trainLoss}
          testLabel={t.testLoss}
          lossTrain={lossTrain}
          lossTest={lossTest}
        />
        <NetworkTrainingPanel
          learningRate={cfg.learningRate}
          optimizer={cfg.optimizer}
          activation={cfg.activation}
          weightInit={cfg.weightInit}
          regularization={cfg.regularization}
          regularizationRate={cfg.regularizationRate}
          discretize={cfg.discretize ?? false}
          onLearningRateChange={onLearningRateChange}
          onOptimizerChange={onOptimizerChange}
          onActivationChange={onActivationChange}
          onWeightInitChange={onWeightInitChange}
          onRegularizationChange={onRegularizationChange}
          onRegularizationRateChange={onRegularizationRateChange}
          onDiscretizeChange={onDiscretizeChange}
        />
        <div
          className="tf-weight-legend tf-weight-legend--dock"
          role="img"
          aria-label="Weight color scale from −1 (violet) to +1 (magenta)"
        >
          <span className="tf-weight-legend__title">Weight</span>
          <div className="tf-weight-legend__bar" />
          <div className="tf-weight-legend__scale">
            <span>−1</span>
            <span>0</span>
            <span>+1</span>
          </div>
        </div>
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
  const curvesRef = useRef<CurveStore>(engineRef.current.curves);
  const targetCurveRef = useRef<number[] | null>(engineRef.current.targetCurve);
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
  const [lossHistory, setLossHistory] = useState<LossHistoryPoint[]>(() =>
    engineRef.current.lossHistory.map((p) => ({ ...p })),
  );
  const bump = useCallback(() => setTick((n) => n + 1), []);
  const [paintGeneration, setPaintGeneration] = useState(0);
  const requestPaint = useCallback(() => setPaintGeneration((n) => n + 1), []);

  useEffect(() => {
    paintAllBoundariesAfterCommit();
  }, [paintGeneration, tick]);

  const [playing, setPlaying] = useState(false);
  const [edgeVizMode, setEdgeVizMode] = useState<EdgeVizMode>("weight");
  const edgeVizModeRef = useRef<EdgeVizMode>("weight");
  edgeVizModeRef.current = edgeVizMode;
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const wasPlayingRef = useRef(false);
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;

  const syncRuntimeRefs = useCallback(() => {
    const engine = engineRef.current;
    boundaryRef.current = engine.boundary;
    curvesRef.current = engine.curves;
    targetCurveRef.current = engine.targetCurve;
    const next = { epoch: engine.epoch, lossTrain: engine.lossTrain, lossTest: engine.lossTest };
    statsRef.current = next;
    setStats(next);
    setLossHistory(engine.lossHistory.map((p) => ({ ...p })));
  }, []);

  useEffect(() => {
    if (!playing) return;
    const engine = engineRef.current;
    engine.refreshOutputBoundaryFast();
    boundaryRef.current = engine.boundary;
    curvesRef.current = engine.curves;
    targetCurveRef.current = engine.targetCurve;
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
    let lastTrainLoss = 0;
    let lastTestLoss = 0;
    let lastHistory = 0;
    let epochBank = 0;
    const paintIntervalMs = 1000 / 30;
    /** Learning-curve sample rate — high enough to look continuous while Play runs. */
    const historyIntervalMs = 1000 / 20;
    const trainLossIntervalMs = 1000 / 20;
    const testLossIntervalMs = 1000 / 8;
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
          curvesRef.current = engine.curves;
          targetCurveRef.current = engine.targetCurve;
          // Imperative paint only — requestPaint() during Play re-renders React Flow and wipes canvases.
          paintBoundaryNode(engine.outputNodeId);
          paintAllBoundaries();
          lastPaint = now;
        }

        if (now - lastTrainLoss >= trainLossIntervalMs) {
          engine.lossTrain = engine.getLoss(engine.trainData);
          lastTrainLoss = now;
        }
        if (now - lastTestLoss >= testLossIntervalMs) {
          engine.lossTest = engine.getLoss(engine.testData);
          lastTestLoss = now;
        }
        if (now - lastHistory >= historyIntervalMs) {
          engine.pushLossHistory();
          setLossHistory(engine.lossHistory.map((p) => ({ ...p })));
          lastHistory = now;
        }
      }

      tick++;
      statsRef.current = {
        epoch: engine.epoch,
        lossTrain: engine.lossTrain,
        lossTest: engine.lossTest,
      };
      if (tick % 4 === 0) {
        setStats({ ...statsRef.current });
        // Gradient strokes change every batch — refresh edges without a full
        // paintGeneration bump (that would remount boundary canvases).
        if (edgeVizModeRef.current === "gradient") bump();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, bump]);

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
      engine.pushLossHistory();
      setLossHistory(engine.lossHistory.map((p) => ({ ...p })));
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

  const onOptimizerChange = useCallback((optimizer: TfPlaygroundConfig["optimizer"]) => {
    engineRef.current.setOptimizer(optimizer);
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

  const onWeightInitChange = useCallback((weightInit: TfPlaygroundConfig["weightInit"]) => {
    engineRef.current.setWeightInit(weightInit);
    setPlaying(false);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onRegularizationChange = useCallback((regularization: TfPlaygroundConfig["regularization"]) => {
    engineRef.current.setRegularization(regularization);
    bump();
  }, [bump]);

  const onRegularizationRateChange = useCallback((regularizationRate: number) => {
    engineRef.current.config.regularizationRate = regularizationRate;
    bump();
  }, [bump]);

  const onDatasetChange = useCallback((dataset: TfAnyDatasetId) => {
    engineRef.current.setDataset(dataset);
    setPlaying(false);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onDataModeChange = useCallback((dataMode: TfDataMode) => {
    engineRef.current.setDataMode(dataMode);
    setPlaying(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    syncRuntimeRefs();
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

  const onProblemTypeChange = useCallback((problemType: TfProblemType) => {
    engineRef.current.setProblemType(problemType);
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
          <div className="tf-flat-switch" role="group" aria-label={t.mode}>
            <button
              type="button"
              className={`tf-flat-switch__btn${engineRef.current.config.dataMode === "1d" ? " selected" : ""}`}
              onClick={() => onDataModeChange("1d")}
            >
              {t.mode1D}
            </button>
            <button
              type="button"
              className={`tf-flat-switch__btn${engineRef.current.config.dataMode === "2d" ? " selected" : ""}`}
              onClick={() => onDataModeChange("2d")}
            >
              {t.mode2D}
            </button>
          </div>
        </div>

        <p className="tf-inspired-by">
          {t.inspiredBy}{" "}
          <a
            href="https://playground.tensorflow.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t.inspiredBySource}
          </a>
        </p>

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
          curvesRef={curvesRef}
          targetCurveRef={targetCurveRef}
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
          onOptimizerChange={onOptimizerChange}
          onActivationChange={onActivationChange}
          onWeightInitChange={onWeightInitChange}
          onRegularizationChange={onRegularizationChange}
          onRegularizationRateChange={onRegularizationRateChange}
          onBatchSizeChange={onBatchSizeChange}
          onNoiseChange={onNoiseChange}
          onTrainRatioChange={onTrainRatioChange}
          onDiscretizeChange={onDiscretizeChange}
          onRegenerateData={onRegenerateData}
          onDatasetChange={onDatasetChange}
          onProblemTypeChange={onProblemTypeChange}
          onResetWeights={resetWeights}
          edgeVizMode={edgeVizMode}
          onEdgeVizModeChange={setEdgeVizMode}
          lossHistory={lossHistory}
          lossTrain={stats.lossTrain}
          lossTest={stats.lossTest}
        />
      </div>
    </div>
  );
}
