import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_NETWORK_CONFIG,
  HEATMAP_PRESET_IDS,
  HEATMAP_PRESETS,
  PlaygroundEngine,
  NetworkTrainClient,
  canUseTrainWorkers,
  type HeatmapPresetId,
  type LossHistoryPoint,
  type NetworkAnyDatasetId,
  type NetworkDataMode,
  type NetworkPlaygroundConfig,
  type NetworkProblemType,
  type NetworkTrainSnapshot,
} from "@ml-vis/core/network";
import { createNetworkTrainWorker } from "@ml-vis/core/workers/createWorkers";
import { paintAllBoundaries, paintAllBoundariesAfterCommit, paintBoundaryNode } from "./network/boundaryPaint";
import type { CurveStore, TrainingStats } from "./network/NetworkBoundaryContext";
import type { EdgeVizMode, LayoutVizMode } from "./network/graphAdapter";
import { useNetworkMessages } from "./network/messages";
import { NetworkGraphPane } from "./network/NetworkGraphPane";
import { dismissStartingAfterPaint } from "./PlayStartingOverlay";


export interface NeuralNetworkPlaygroundProps {
  initialConfig?: Partial<NetworkPlaygroundConfig>;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

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
  const [starting, setStarting] = useState(false);
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
  /** True after Play is sent to the worker, until the first live frame is on screen. */
  const awaitingFirstPlayPaintRef = useRef(false);

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
      if (awaitingFirstPlayPaintRef.current && trainingLiveRef.current) {
        awaitingFirstPlayPaintRef.current = false;
        dismissStartingAfterPaint(setStarting);
      }
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
    requestPaint();
    bump();
    if (trainClientRef.current) {
      void syncWorkerFromMain("topology");
      return;
    }
    const epoch = topologyEpochRef.current;
    window.setTimeout(() => {
      if (topologyEpochRef.current !== epoch) return;
      engineRef.current.refreshBoundary();
      syncRuntimeRefs();
      requestPaint();
    }, 0);
  }, [bump, requestPaint, syncRuntimeRefs, syncWorkerFromMain]);

  // Play/pause → worker owns trainEpoch; display engine only applies ticks.
  useEffect(() => {
    const client = trainClientRef.current;
    if (!client) {
      // Fallback: previous main-thread loop when Workers are missing.
      if (!playing) return;
      awaitingFirstPlayPaintRef.current = true;
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
          if (awaitingFirstPlayPaintRef.current) {
            awaitingFirstPlayPaintRef.current = false;
            dismissStartingAfterPaint(setStarting);
          }
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
          awaitingFirstPlayPaintRef.current = true;
          client.play(75);
        });
      return () => {
        cancelled = true;
      };
    }
    client.pause();
  }, [playing, ensureWorkerSynced, syncRuntimeRefs]);

  useEffect(() => {
    if (!playing) {
      awaitingFirstPlayPaintRef.current = false;
      setStarting(false);
    }
  }, [playing]);

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
    afterTopologyEdit();
  }, [afterTopologyEdit]);

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

  const onHeatmapPresetChange = useCallback((preset: HeatmapPresetId) => {
    engineRef.current.setHeatmapPreset(preset);
    setPlaying(false);
    syncRuntimeRefs();
    trainClientRef.current?.command("setHeatmapPreset", { preset });
    requestPaint();
    bump();
  }, [bump, requestPaint, syncRuntimeRefs]);

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
            className={`nn-btn nn-btn--primary${starting ? " starting" : playing ? " playing" : ""}`}
            onClick={() => {
              setPlaying((p) => {
                const next = !p;
                setStarting(next);
                return next;
              });
            }}
          >
            {starting ? (
              <>
                <span className="nn-btn__spinner" aria-hidden />
                {t.starting}
              </>
            ) : playing ? (
              t.pause
            ) : (
              t.play
            )}
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
          {engineRef.current.config.dataMode !== "1d" && (
            <div className="nn-toolbar-weight-viz" role="group" aria-label={t.heatmapViz}>
              <span className="nn-toolbar-weight-viz__label">{t.heatmapViz}</span>
              <div className="nn-flat-switch">
                {HEATMAP_PRESET_IDS.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`nn-flat-switch__btn${engineRef.current.config.heatmapPreset === id ? " selected" : ""}`}
                    onClick={() => onHeatmapPresetChange(id)}
                  >
                    {HEATMAP_PRESETS[id].output}
                  </button>
                ))}
              </div>
            </div>
          )}
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
          playStarting={starting}
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
