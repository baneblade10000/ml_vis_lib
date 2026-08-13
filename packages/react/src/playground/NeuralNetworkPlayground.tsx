import { memo, useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { CLASS_0_HEX, CLASS_1_HEX, DEFAULT_NETWORK_CONFIG, PlaygroundEngine, NetworkTrainClient, canUseTrainWorkers, valueToRgb, type Dataset1DId, type LossHistoryPoint, type NetworkAnyDatasetId, type NetworkDataMode, type DatasetId as NetworkDatasetId, type NetworkPlaygroundConfig, type NetworkProblemType, type NetworkTrainSnapshot } from "@ml-vis/core/network";
import { DATASETS as NETWORK_DATASETS, DATASETS_1D_CLASSIFICATION as NETWORK_DATASETS_1D_CLASSIFICATION, DATASETS_1D_REGRESSION as NETWORK_DATASETS_1D_REGRESSION } from "@ml-vis/core/network";
import { createNetworkTrainWorker } from "@ml-vis/core/workers/createWorkers";
import { NetworkInspector } from "./network/NetworkInspector";
import { NetworkArchitecturePanel } from "./network/NetworkArchitecturePanel";
import { NetworkPalette } from "./network/NetworkPalette";
import { NetworkDataPanel } from "./network/NetworkDataPanel";
import { NetworkTrainingPanel } from "./network/NetworkTrainingPanel";
import { NetworkLossChart } from "./network/NetworkLossChart";
import { ReactFlowNetworkGraph } from "./network/ReactFlowNetworkGraph";
import { paintAllBoundaries, paintAllBoundariesAfterCommit, paintBoundaryNode } from "./network/boundaryPaint";
import type { CurveStore, TrainingStats } from "./network/NetworkBoundaryContext";
import type { EdgeVizMode, LayoutVizMode } from "./network/graphAdapter";
import { useNetworkMessages } from "./network/messages";

const DATASETS_2D_CLASSIFICATION: NetworkDatasetId[] = ["circle", "xor", "gauss", "spiral"];
const DATASETS_2D_REGRESSION: NetworkDatasetId[] = ["sinSin"];
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
  datasetId: NetworkAnyDatasetId;
  label: string;
  selected: boolean;
  onSelect: () => void;
  mode: NetworkDataMode;
  problemType?: NetworkProblemType;
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
        datasetId in NETWORK_DATASETS_1D_CLASSIFICATION
          ? NETWORK_DATASETS_1D_CLASSIFICATION[datasetId as keyof typeof NETWORK_DATASETS_1D_CLASSIFICATION]
          : NETWORK_DATASETS_1D_REGRESSION[datasetId as keyof typeof NETWORK_DATASETS_1D_REGRESSION];
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

    const gen = NETWORK_DATASETS[datasetId as NetworkDatasetId];
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
  initialConfig?: Partial<NetworkPlaygroundConfig>;
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
  onOptimizerChange: (optimizer: NetworkPlaygroundConfig["optimizer"]) => void;
  onActivationChange: (activation: NetworkPlaygroundConfig["activation"]) => void;
  onWeightInitChange: (weightInit: NetworkPlaygroundConfig["weightInit"]) => void;
  onRegularizationChange: (regularization: NetworkPlaygroundConfig["regularization"]) => void;
  onRegularizationRateChange: (regularizationRate: number) => void;
  onBatchSizeChange: (batchSize: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (percTrainData: number) => void;
  onDiscretizeChange: (discretize: boolean) => void;
  onRegenerateData: () => void;
  onDatasetChange: (dataset: NetworkAnyDatasetId) => void;
  onProblemTypeChange: (problemType: NetworkProblemType) => void;
  onResetWeights: () => void;
  edgeVizMode: EdgeVizMode;
  onEdgeVizModeChange: (mode: EdgeVizMode) => void;
  layoutVizMode: LayoutVizMode;
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
  layoutVizMode,
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
      layoutVizMode={layoutVizMode}
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
      <aside className="nn-flow-dock nn-flow-dock--left">
        <NetworkPalette />
        <NetworkArchitecturePanel
          numHiddenLayers={cfg.numHiddenLayers}
          networkShape={cfg.networkShape}
          onAddLayer={onAddLayer}
          onRemoveLayer={onRemoveLayer}
          onAddNeuron={onAddNeuron}
          onRemoveNeuron={onRemoveNeuron}
        />
        <div className="nn-flow-dock-section">
          <button type="button" className="nn-btn nn-btn--secondary nn-reset-weights" onClick={onResetWeights}>
            {t.resetWeights}
          </button>
          <button type="button" className="nn-btn nn-btn--secondary nn-layout-normalize" onClick={onNormalizeLayout}>
            <span className="nn-layout-normalize-icon" aria-hidden="true">⊞</span>
            {t.arrangeLayout}
          </button>
        </div>
        <div className="nn-flow-dock-section">
          <h4 className="nn-flow-dock-title">{t.edgeViz}</h4>
          <div className="nn-flat-switch" role="group" aria-label={t.edgeViz}>
            {([
              ["weight", t.edgeVizWeight],
              ["gradient", t.edgeVizGradient],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={`nn-flat-switch__btn${edgeVizMode === id ? " selected" : ""}`}
                onClick={() => onEdgeVizModeChange(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="nn-flow-dock-section">
          <h4 className="nn-flow-dock-title">{t.problemType}</h4>
          <div className="nn-flat-switch" role="group" aria-label={t.problemType}>
            {(["classification", "regression"] as NetworkProblemType[]).map((id) => (
              <button
                key={id}
                type="button"
                className={`nn-flat-switch__btn${cfg.problemType === id ? " selected" : ""}`}
                onClick={() => onProblemTypeChange(id)}
              >
                {t.problemTypeLabels[id]}
              </button>
            ))}
          </div>
        </div>
        <div className="nn-flow-dock-section">
          <h4 className="nn-flow-dock-title">{t.dataset}</h4>
          <div className="dataset-list dataset-list--compact">
            {datasetIds.map((id) => (
              <DatasetThumbnail
                key={id}
                datasetId={id}
                label={
                  cfg.dataMode === "1d"
                    ? t.dataset1dLabels[id as Dataset1DId]
                    : t.datasetLabels[id as NetworkDatasetId]
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
      </aside>
      <aside className="nn-flow-dock nn-flow-dock--right">
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
          className="nn-weight-legend nn-weight-legend--dock"
          role="img"
          aria-label="Weight color scale from −1 (deep blue) through 0 (mid blue) to +1 (sky cyan)"
        >
          <span className="nn-weight-legend__title">Weight</span>
          <div className="nn-weight-legend__bar" />
          <div className="nn-weight-legend__scale">
            <span>−1</span>
            <span>0</span>
            <span>+1</span>
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
    </ReactFlowNetworkGraph>
  );
});

export function NeuralNetworkPlayground({ initialConfig, toolbarStart, toolbarEnd }: NeuralNetworkPlaygroundProps) {
  // Lazy init: the constructor runs data generation + boundary computation,
  // so it must not execute on every render.
  const engineRef = useRef<PlaygroundEngine>(undefined as unknown as PlaygroundEngine);
  if (!engineRef.current) {
    engineRef.current = new PlaygroundEngine(initialConfig ?? DEFAULT_NETWORK_CONFIG);
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
  const [layoutVizMode, setLayoutVizMode] = useState<LayoutVizMode>("graph");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const wasPlayingRef = useRef(false);
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;
  const trainClientRef = useRef<NetworkTrainClient | null>(null);
  const trainReadyRef = useRef<Promise<unknown> | null>(null);
  const edgeVizBumpRef = useRef(0);
  /** Bumped on main-thread topology edits; worker rebuild catches up. */
  const topologyEpochRef = useRef(0);
  const syncedTopologyEpochRef = useRef(0);
  const rebuildPromiseRef = useRef<Promise<void> | null>(null);

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

  const applyNetworkTick = useCallback(
    (snap: NetworkTrainSnapshot) => {
      const engine = engineRef.current;
      // Ignore stale ticks from a pre-edit worker topology — otherwise their
      // boundary map (old node ids) wipes heatmaps and leaves new neurons white.
      if (
        snap.graphSnapshot.outputId !== engine.graph.outputId ||
        snap.graphSnapshot.nodes.length !== engine.graph.nodes.size ||
        snap.graphSnapshot.nodes.some((n) => !engine.graph.nodes.has(n.id))
      ) {
        return;
      }
      engine.applyWorkerViz(snap);
      boundaryRef.current = engine.boundary;
      curvesRef.current = engine.curves;
      targetCurveRef.current = engine.targetCurve;
      statsRef.current = {
        epoch: engine.epoch,
        lossTrain: engine.lossTrain,
        lossTest: engine.lossTest,
      };
      setStats({ ...statsRef.current });
      setLossHistory(engine.lossHistory.map((p) => ({ ...p })));
      paintBoundaryNode(engine.outputNodeId);
      paintAllBoundaries();
      edgeVizBumpRef.current += 1;
      if (edgeVizModeRef.current === "gradient" && edgeVizBumpRef.current % 4 === 0) {
        bump();
      }
    },
    [bump],
  );

  const workerPayload = useCallback(() => {
    const engine = engineRef.current;
    return {
      config: structuredClone(engine.config),
      graphSnapshot: engine.graph.toSnapshot(),
      trainData: engine.trainData.map((p) => ({ x: p.x, y: p.y, label: p.label })),
      testData: engine.testData.map((p) => ({ x: p.x, y: p.y, label: p.label })),
    };
  }, []);

  // Boot TF train worker; main engine remains the topology/UI source of truth.
  useEffect(() => {
    if (!canUseTrainWorkers()) {
      console.warn("[network] Web Workers unavailable; falling back to main-thread train");
      return;
    }
    const client = new NetworkTrainClient({
      createWorker: createNetworkTrainWorker,
      onTick: applyNetworkTick,
      onError: (message) => console.error("[network train worker]", message),
    });
    trainClientRef.current = client;
    trainReadyRef.current = client
      .init(workerPayload())
      .then((snap) => {
        syncedTopologyEpochRef.current = topologyEpochRef.current;
        return snap;
      })
      .catch((err) => {
        console.error("[network train worker] init failed", err);
      });
    return () => {
      client.dispose();
      trainClientRef.current = null;
      trainReadyRef.current = null;
    };
  }, [applyNetworkTick, workerPayload]);

  const syncWorkerFromMain = useCallback(
    (reason: "topology" | "dataset" | "reset" | "resetWeights" | "mode" = "topology") => {
      const client = trainClientRef.current;
      if (!client) {
        syncedTopologyEpochRef.current = topologyEpochRef.current;
        return Promise.resolve();
      }
      const epochAtStart = topologyEpochRef.current;
      const p = client.rebuild(reason, workerPayload()).then(() => {
        if (rebuildPromiseRef.current === p) rebuildPromiseRef.current = null;
        // Only mark synced if no newer edit landed while we waited.
        if (topologyEpochRef.current === epochAtStart) {
          syncedTopologyEpochRef.current = epochAtStart;
        }
      });
      rebuildPromiseRef.current = p;
      return p;
    },
    [workerPayload],
  );

  /** Wait until worker matches main topology (and any in-flight rebuild finishes). */
  const ensureWorkerSynced = useCallback(async () => {
    if (syncedTopologyEpochRef.current !== topologyEpochRef.current) {
      await syncWorkerFromMain("topology");
    } else if (rebuildPromiseRef.current) {
      await rebuildPromiseRef.current;
    }
  }, [syncWorkerFromMain]);

  /** Topology edits on main must rebuild the train worker or ticks desync heatmaps. */
  const afterTopologyEdit = useCallback(() => {
    topologyEpochRef.current += 1;
    setPlaying(false);
    syncRuntimeRefs();
    void syncWorkerFromMain("topology");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  // Play/pause → worker owns trainEpoch; display engine only applies ticks.
  useEffect(() => {
    const client = trainClientRef.current;
    if (!client) {
      // Fallback: previous main-thread loop when Workers are missing.
      if (!playing) return;
      const epochsPerSec = 75;
      let raf = 0;
      let lastTime = performance.now();
      let epochBank = 0;
      const loop = (now: number) => {
        const engine = engineRef.current;
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;
        epochBank = Math.min(epochBank + dt * epochsPerSec, 2);
        const steps = Math.floor(epochBank);
        epochBank -= steps;
        for (let i = 0; i < steps; i++) engine.trainEpoch(false);
        if (steps > 0) {
          engine.refreshOutputBoundaryFast();
          engine.refreshHiddenBoundariesFast();
          engine.lossTrain = engine.getLoss(engine.trainData);
          engine.lossTest = engine.getLoss(engine.testData);
          engine.pushLossHistory();
          syncRuntimeRefs();
          paintBoundaryNode(engine.outputNodeId);
          paintAllBoundaries();
        }
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(raf);
    }
    if (playing) {
      const ready = trainReadyRef.current ?? Promise.resolve();
      let cancelled = false;
      void ready
        .then(() => ensureWorkerSynced())
        .then(() => {
          if (cancelled || !trainClientRef.current || trainClientRef.current !== client) return;
          client.play(75);
        });
      return () => {
        cancelled = true;
      };
    }
    client.pause();
  }, [playing, ensureWorkerSynced, syncRuntimeRefs]);

  useEffect(() => {
    if (wasPlayingRef.current && !playing) {
      const engine = engineRef.current;
      // Prefer last worker tick; refresh locally if worker missing.
      if (!trainClientRef.current) {
        engine.refreshMetrics();
        engine.refreshBoundary();
      }
      boundaryRef.current = engine.boundary;
      statsRef.current = {
        epoch: engine.epoch,
        lossTrain: engine.lossTrain,
        lossTest: engine.lossTest,
      };
      setStats({ ...statsRef.current });
      setLossHistory(engine.lossHistory.map((p) => ({ ...p })));
      requestPaint();
      bump();
    }
    wasPlayingRef.current = playing;
  }, [playing, bump, requestPaint]);

  const reset = () => {
    engineRef.current.resetToInitial();
    topologyEpochRef.current += 1;
    setPlaying(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setRefitViewKey((n) => n + 1);
    syncRuntimeRefs();
    syncWorkerFromMain("reset");
    requestPaint();
    bump();
  };

  const resetWeights = () => {
    engineRef.current.resetWeights();
    setPlaying(false);
    syncRuntimeRefs();
    syncWorkerFromMain("resetWeights");
    requestPaint();
    bump();
  };

  const step = () => {
    setPlaying(false);
    const client = trainClientRef.current;
    if (client) {
      void ensureWorkerSynced().then(() => {
        if (trainClientRef.current !== client) return;
        client.step();
      });
    } else {
      engineRef.current.step();
      syncRuntimeRefs();
      paintAllBoundariesAfterCommit();
    }
    bump();
    requestPaint();
  };

  // Hyperparameters: update display engine + live worker config during Play.
  const onLearningRateChange = useCallback((learningRate: number) => {
    engineRef.current.config.learningRate = learningRate;
    trainClientRef.current?.command("setLearningRate", { lr: learningRate });
    bump();
  }, [bump]);

  const onOptimizerChange = useCallback((optimizer: NetworkPlaygroundConfig["optimizer"]) => {
    engineRef.current.setOptimizer(optimizer);
    trainClientRef.current?.command("setOptimizer", { optimizer });
    bump();
  }, [bump]);

  const onBatchSizeChange = useCallback((batchSize: number) => {
    engineRef.current.config.batchSize = batchSize;
    trainClientRef.current?.command("setBatchSize", { bs: batchSize });
    bump();
  }, [bump]);

  const onActivationChange = useCallback((activation: NetworkPlaygroundConfig["activation"]) => {
    engineRef.current.setActivation(activation);
    topologyEpochRef.current += 1;
    setPlaying(false);
    syncRuntimeRefs();
    syncWorkerFromMain("topology");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  const onWeightInitChange = useCallback((weightInit: NetworkPlaygroundConfig["weightInit"]) => {
    engineRef.current.setWeightInit(weightInit);
    topologyEpochRef.current += 1;
    setPlaying(false);
    syncRuntimeRefs();
    syncWorkerFromMain("resetWeights");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  const onRegularizationChange = useCallback((regularization: NetworkPlaygroundConfig["regularization"]) => {
    engineRef.current.setRegularization(regularization);
    trainClientRef.current?.command("setRegularization", { regularization });
    bump();
  }, [bump]);

  const onRegularizationRateChange = useCallback((regularizationRate: number) => {
    engineRef.current.config.regularizationRate = regularizationRate;
    trainClientRef.current?.command("setRegularizationRate", { rate: regularizationRate });
    bump();
  }, [bump]);

  const onDatasetChange = useCallback((dataset: NetworkAnyDatasetId) => {
    engineRef.current.setDataset(dataset);
    topologyEpochRef.current += 1;
    setPlaying(false);
    syncRuntimeRefs();
    syncWorkerFromMain("dataset");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  const onDataModeChange = useCallback((dataMode: NetworkDataMode) => {
    engineRef.current.setDataMode(dataMode);
    topologyEpochRef.current += 1;
    setPlaying(false);
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    syncRuntimeRefs();
    syncWorkerFromMain("mode");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  const onProblemTypeChange = useCallback((problemType: NetworkProblemType) => {
    engineRef.current.setProblemType(problemType);
    topologyEpochRef.current += 1;
    setPlaying(false);
    syncRuntimeRefs();
    syncWorkerFromMain("dataset");
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  const onToggleFeature = useCallback((id: string) => {
    engineRef.current.toggleFeature(id);
    afterTopologyEdit();
  }, [afterTopologyEdit]);

  const onConnect = useCallback((sourceId: string, targetId: string) => {
    engineRef.current.connectNodes(sourceId, targetId);
    afterTopologyEdit();
  }, [afterTopologyEdit]);

  const onDropNode = useCallback((kind: Parameters<PlaygroundEngine["addPaletteNode"]>[0], position: { x: number; y: number }) => {
    engineRef.current.addPaletteNode(kind, position);
    afterTopologyEdit();
  }, [afterTopologyEdit]);

  const onMoveNode = useCallback((nodeId: string, position: { x: number; y: number }) => {
    engineRef.current.setNodePosition(nodeId, position);
  }, []);

  const onRemoveNode = useCallback((nodeId: string) => {
    engineRef.current.removeGraphNode(nodeId);
    setSelectedNodeId(null);
    afterTopologyEdit();
  }, [afterTopologyEdit]);

  const onRemoveEdge = useCallback((sourceId: string, targetId: string) => {
    engineRef.current.disconnectNodes(sourceId, targetId);
    setSelectedEdgeId(null);
    afterTopologyEdit();
  }, [afterTopologyEdit]);

  const afterMlpShapeEdit = useCallback(() => {
    afterTopologyEdit();
    // Trigger the same glide + viewport refit as "Arrange layout".
    setRefitViewKey((n) => n + 1);
  }, [afterTopologyEdit]);

  const onAddLayer = useCallback(() => {
    engineRef.current.addLayer();
    afterMlpShapeEdit();
  }, [afterMlpShapeEdit]);

  const onRemoveLayer = useCallback(() => {
    engineRef.current.removeLayer();
    afterMlpShapeEdit();
  }, [afterMlpShapeEdit]);

  const onAddNeuron = useCallback((layerIdx: number) => {
    engineRef.current.addNeuron(layerIdx);
    afterMlpShapeEdit();
  }, [afterMlpShapeEdit]);

  const onRemoveNeuron = useCallback((layerIdx: number) => {
    engineRef.current.removeNeuron(layerIdx);
    afterMlpShapeEdit();
  }, [afterMlpShapeEdit]);

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
    <div className="nn-playground nn-playground--immersive">
      <div className="nn-immersive-toolbar">
        <div className="nn-toolbar-group nn-toolbar-group--actions">
          {toolbarStart}
          <button type="button" className="nn-btn nn-btn--ghost" onClick={reset}>
            {t.reset}
          </button>
          <button
            type="button"
            className={`nn-btn nn-btn--primary${playing ? " playing" : ""}`}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? t.pause : t.play}
          </button>
          <button type="button" className="nn-btn nn-btn--secondary" onClick={step}>
            {t.step}
          </button>
          <div className="nn-flat-switch" role="group" aria-label={t.mode}>
            <button
              type="button"
              className={`nn-flat-switch__btn${engineRef.current.config.dataMode === "1d" ? " selected" : ""}`}
              onClick={() => onDataModeChange("1d")}
            >
              {t.mode1D}
            </button>
            <button
              type="button"
              className={`nn-flat-switch__btn${engineRef.current.config.dataMode === "2d" ? " selected" : ""}`}
              onClick={() => onDataModeChange("2d")}
            >
              {t.mode2D}
            </button>
          </div>
          <div className="nn-toolbar-weight-viz" role="group" aria-label={t.layoutViz}>
            <span className="nn-toolbar-weight-viz__label">{t.layoutViz}</span>
            <div className="nn-flat-switch">
              {([
                ["graph", t.layoutVizGraph],
                ["matrix", t.layoutVizMatrix],
              ] as const).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={`nn-flat-switch__btn${layoutVizMode === id ? " selected" : ""}`}
                  onClick={() => {
                    setLayoutVizMode(id);
                    setRefitViewKey((n) => n + 1);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="nn-inspired-by">
          {t.inspiredBy}{" "}
          <a
            href="https://playground.tensorflow.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t.inspiredBySource}
          </a>
        </p>

        <div className="nn-toolbar-group nn-toolbar-group--params">
          <div className="nn-toolbar-stat">
            <span className="label">{t.epoch}</span>
            <span className="value">{stats.epoch.toLocaleString()}</span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.testLoss}</span>
            <span className="value" id="loss-test">
              {stats.lossTest.toFixed(3)}
            </span>
          </div>
          <div className="nn-toolbar-stat nn-toolbar-stat--train">
            <span className="label">{t.trainLoss}</span>
            <span className="value" id="loss-train">
              {stats.lossTrain.toFixed(3)}
            </span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="nn-immersive-body">
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
          layoutVizMode={layoutVizMode}
          lossHistory={lossHistory}
          lossTrain={stats.lossTrain}
          lossTest={stats.lossTest}
        />
      </div>
    </div>
  );
}
