import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CNN_DATASET_IDS_1D,
  CNN_DATASET_IDS_2D,
  DEFAULT_CNN_CONFIG,
  TrainWorkerClient,
  canUseTrainWorkers,
  type CnnActivationId,
  type CnnMode,
  type CnnRegularizationId,
  type CnnTrainSnapshot,
  type FeatureMapSnapshot,
  type ImageExample,
  type LayerShape,
  type LossHistoryPoint,
  type PlaygroundOptimizerId,
  type SignalExample,
  type TrainSnapshot,
} from "@ml-vis/core";
import { createCnnTrainWorker } from "@ml-vis/core/workers/createWorkers";
import { NetworkLossChart } from "../network/NetworkLossChart";
import { CnnFlowGraph } from "./CnnFlowGraph";
import { CnnArchitecturePanel } from "./CnnArchitecturePanel";
import { CnnTrainingPanel } from "./CnnTrainingPanel";
import { CnnDataPanel } from "./CnnDataPanel";
import { CnnInspector } from "./CnnInspector";
import { CnnGallery } from "./CnnGallery";
import { formatCnnNodeLabel } from "./cnnAdapter";
import { useCnnMessages } from "./messages";
import {
  paintAllFeatureMaps,
  paintAllFeatureMapsAfterCommit,
} from "./featureMapPaint";
import type { CnnTrainingStats, FeatureMapStore } from "./featureMapContext";

export interface ConvolutionalNetworkPlaygroundProps {
  initialMode?: CnnMode;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

function shapeLabel(shape: LayerShape): string {
  if (shape.kind === "2d") return `${shape.channels}×${shape.rows}×${shape.cols}`;
  return `${shape.channels}×${shape.length}`;
}

function specKindLabel(kind: string, t: ReturnType<typeof useCnnMessages>): string {
  switch (kind) {
    case "conv2d":
    case "conv1d":
      return t.paletteConv;
    case "pool2d":
    case "pool1d":
      return t.palettePool;
    case "flatten":
      return t.flatten;
    case "dense":
      return t.paletteDense;
    default:
      return kind;
  }
}

function isCnnSnapshot(s: TrainSnapshot | null): s is CnnTrainSnapshot {
  return s?.kind === "cnn";
}

export function ConvolutionalNetworkPlayground({
  initialMode = "2d",
  toolbarStart,
  toolbarEnd,
}: ConvolutionalNetworkPlaygroundProps) {
  const t = useCnnMessages();
  const initialConfig = initialMode === "1d" ? DEFAULT_CNN_CONFIG["1d"] : DEFAULT_CNN_CONFIG["2d"];

  const [snapshot, setSnapshot] = useState<CnnTrainSnapshot | null>(null);
  const [playing, setPlaying] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [paintGeneration, setPaintGeneration] = useState(0);
  const requestPaint = useCallback(() => setPaintGeneration((n) => n + 1), []);

  const featureMapRef = useRef<FeatureMapStore>({});
  const statsRef = useRef<CnnTrainingStats>({
    epoch: 0,
    lossTrain: 0,
    lossTest: 0,
    accTrain: 0,
    accTest: 0,
  });
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;
  const clientRef = useRef<TrainWorkerClient | null>(null);
  const playingRef = useRef(false);
  playingRef.current = playing;

  const applySnapshot = useCallback(
    (snap: CnnTrainSnapshot) => {
      setSnapshot(snap);
      statsRef.current = { ...snap.stats };
      const store: FeatureMapStore = {};
      for (const m of snap.featureMaps) store[m.layerId] = m;
      featureMapRef.current = store;
      requestPaint();
      paintAllFeatureMaps();
    },
    [requestPaint],
  );

  // Boot train worker (or fall back to a same-thread shim via Worker if unavailable — skip).
  useEffect(() => {
    if (!canUseTrainWorkers()) {
      console.warn("[cnn] Web Workers unavailable; CNN playground requires Workers");
      return;
    }
    const client = new TrainWorkerClient({
      createWorker: createCnnTrainWorker,
      onTick: (s) => {
        if (isCnnSnapshot(s)) applySnapshot(s);
      },
      onError: (message) => console.error("[cnn train worker]", message),
    });
    clientRef.current = client;
    let cancelled = false;
    void client.init(structuredClone(initialConfig)).then((s) => {
      if (!cancelled && isCnnSnapshot(s)) applySnapshot(s);
    });
    return () => {
      cancelled = true;
      client.dispose();
      clientRef.current = null;
    };
    // Only boot once for the initial mode; mode switches go through commands.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    paintAllFeatureMapsAfterCommit();
  }, [paintGeneration, snapshot]);

  // Drive play/pause on the worker.
  useEffect(() => {
    const client = clientRef.current;
    if (!client) return;
    if (playing) client.play(12);
    else client.pause();
  }, [playing]);

  const cmd = useCallback((name: string, args?: unknown) => {
    clientRef.current?.command(name, args);
  }, []);

  const reset = useCallback(() => {
    setPlaying(false);
    setSelectedLayerId(null);
    clientRef.current?.rebuild("reset");
  }, []);

  const resetWeights = useCallback(() => {
    setPlaying(false);
    clientRef.current?.rebuild("resetWeights");
  }, []);

  const step = useCallback(() => {
    setPlaying(false);
    clientRef.current?.step();
  }, []);

  const onModeChange = useCallback((mode: CnnMode) => {
    setPlaying(false);
    setSelectedLayerId(null);
    clientRef.current?.rebuild("mode", mode);
  }, []);

  const onDatasetChange = useCallback((id: string) => {
    setPlaying(false);
    cmd("setDataset", { dataset: id });
  }, [cmd]);

  const onActivationChange = useCallback(
    (a: CnnActivationId) => {
      cmd("setActivation", { activation: a });
    },
    [cmd],
  );

  const onLearningRateChange = useCallback(
    (lr: number) => {
      cmd("setLearningRate", { lr });
    },
    [cmd],
  );

  const onOptimizerChange = useCallback(
    (optimizer: PlaygroundOptimizerId) => {
      cmd("setOptimizer", { optimizer });
    },
    [cmd],
  );

  const onBatchSizeChange = useCallback(
    (bs: number) => {
      cmd("setBatchSize", { bs });
    },
    [cmd],
  );

  const onRegularizationChange = useCallback(
    (regularization: CnnRegularizationId) => {
      cmd("setRegularization", { regularization });
    },
    [cmd],
  );

  const onRegularizationRateChange = useCallback(
    (rate: number) => {
      cmd("setRegularizationRate", { rate });
    },
    [cmd],
  );

  const onNoiseChange = useCallback(
    (n: number) => {
      setPlaying(false);
      cmd("updateDataParams", { noise: n });
    },
    [cmd],
  );

  const onTrainRatioChange = useCallback(
    (r: number) => {
      setPlaying(false);
      cmd("updateDataParams", { percTrainData: r });
    },
    [cmd],
  );

  const onRegenerateData = useCallback(() => {
    setPlaying(false);
    cmd("regenerateData");
  }, [cmd]);

  const selectedSpecIndex = useMemo(() => {
    if (!snapshot || !selectedLayerId) return null;
    const idx = snapshot.layers.findIndex((l) => l.id === selectedLayerId) - 1;
    return idx >= 0 && idx < snapshot.config.layers.length ? idx : null;
  }, [snapshot, selectedLayerId]);

  const onRemoveLayer = useCallback(
    (index: number) => {
      setPlaying(false);
      setSelectedLayerId(null);
      cmd("removeLayer", { index });
    },
    [cmd],
  );

  const onSetFilters = useCallback(
    (index: number, filters: number) => {
      setPlaying(false);
      cmd("setLayerFilters", { index, filters });
    },
    [cmd],
  );

  const onSetKernelSize = useCallback(
    (index: number, kernelSize: number) => {
      setPlaying(false);
      cmd("setLayerKernelSize", { index, kernelSize });
    },
    [cmd],
  );

  const onSetUnits = useCallback(
    (index: number, units: number) => {
      setPlaying(false);
      cmd("setLayerUnits", { index, units });
    },
    [cmd],
  );

  const onSetPoolKind = useCallback(
    (index: number, poolKind: "max" | "avg") => {
      setPlaying(false);
      cmd("setLayerPoolKind", { index, poolKind });
    },
    [cmd],
  );

  const onSelectExample = useCallback(
    (index: number) => {
      clientRef.current?.inspect(index);
    },
    [],
  );

  if (!snapshot) {
    return (
      <div className="nn-playground nn-playground--immersive">
        <div className="nn-immersive-toolbar">
          <span className="nn-toolbar-stat">
            <span className="label">{t.training}</span>
            <span className="value">…</span>
          </span>
        </div>
      </div>
    );
  }

  const mode = snapshot.config.mode;
  const datasetIds = mode === "2d" ? CNN_DATASET_IDS_2D : CNN_DATASET_IDS_1D;
  const datasetLabels = mode === "2d" ? t.datasetLabels2D : t.datasetLabels1D;
  const galleryExamples = snapshot.galleryExamples as Array<ImageExample | SignalExample>;
  const galleryPredictions = snapshot.galleryPredictions;
  const inspectedIdx = snapshot.inspectedExampleIndex;
  const featureMaps: FeatureMapSnapshot[] = snapshot.featureMaps;
  const stats = snapshot.stats;
  const lossHistory: LossHistoryPoint[] = snapshot.lossHistory;
  const loss = snapshot.loss;
  const probability = snapshot.probability;
  const pipeline = { mode, layers: snapshot.layers };

  const inspectorInfo = (() => {
    if (!selectedLayerId) return null;
    const idx = snapshot.layers.findIndex((l) => l.id === selectedLayerId);
    if (idx < 0) return null;
    const layer = snapshot.layers[idx]!;
    const inShape = snapshot.layers[idx - 1]?.shape;
    const outShape = layer.shape;
    return {
      kind: specKindLabel(layer.kind, t),
      label: formatCnnNodeLabel({ kind: layer.kind, label: () => layer.label }, t),
      inputShape: inShape ? shapeLabel(inShape) : "—",
      outputShape: outShape ? shapeLabel(outShape) : "—",
      params: layer.params,
    };
  })();

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
              className={`nn-flat-switch__btn${mode === "1d" ? " selected" : ""}`}
              onClick={() => onModeChange("1d")}
            >
              {t.mode1D}
            </button>
            <button
              type="button"
              className={`nn-flat-switch__btn${mode === "2d" ? " selected" : ""}`}
              onClick={() => onModeChange("2d")}
            >
              {t.mode2D}
            </button>
          </div>
        </div>

        <div className="nn-toolbar-group nn-toolbar-group--params">
          <div className="nn-toolbar-stat">
            <span className="label">{t.epoch}</span>
            <span className="value">{stats.epoch.toLocaleString()}</span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.testAcc}</span>
            <span className="value">{(stats.accTest * 100).toFixed(0)}%</span>
          </div>
          <div className="nn-toolbar-stat nn-toolbar-stat--train">
            <span className="label">{t.trainAcc}</span>
            <span className="value">{(stats.accTrain * 100).toFixed(0)}%</span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.testLoss}</span>
            <span className="value">{stats.lossTest.toFixed(3)}</span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="nn-immersive-body">
        <CnnFlowGraph
          pipeline={pipeline}
          selectedNodeId={selectedLayerId}
          paintGeneration={paintGeneration}
          featureMaps={featureMaps}
          onSelectNode={setSelectedLayerId}
          featureMapRef={featureMapRef}
          statsRef={statsRef}
          trainingLiveRef={trainingLiveRef}
          loss={loss}
          probability={probability}
          fillHeight
        >
          <aside className="nn-flow-dock nn-flow-dock--left nn-flow-dock--wide">
            <CnnArchitecturePanel
              layers={snapshot.config.layers}
              selectedIndex={selectedSpecIndex}
              onSelectLayer={(idx) => {
                const layer = snapshot.layers[idx + 1];
                setSelectedLayerId(layer ? layer.id : null);
              }}
              onRemoveLayer={onRemoveLayer}
              onSetFilters={onSetFilters}
              onSetKernelSize={onSetKernelSize}
              onSetUnits={onSetUnits}
              onSetPoolKind={onSetPoolKind}
            />
            <div className="nn-flow-dock-section">
              <button
                type="button"
                className="nn-btn nn-btn--secondary nn-reset-weights"
                onClick={resetWeights}
              >
                {t.resetWeights}
              </button>
            </div>
            <div className="nn-flow-dock-section">
              <CnnGallery
                mode={mode}
                examples={galleryExamples}
                predictions={galleryPredictions}
                inspectedIndex={inspectedIdx}
                onSelectExample={onSelectExample}
                datasetId={snapshot.config.dataset}
                onSelectDataset={onDatasetChange}
                datasetIds={datasetIds}
                datasetLabels={datasetLabels}
              />
              <CnnDataPanel
                batchSize={snapshot.config.batchSize}
                noise={snapshot.config.noise}
                percTrainData={snapshot.config.percTrainData}
                onBatchSizeChange={onBatchSizeChange}
                onNoiseChange={onNoiseChange}
                onTrainRatioChange={onTrainRatioChange}
                onRegenerateData={onRegenerateData}
              />
            </div>
            <CnnInspector
              selectedLayerId={selectedLayerId}
              kernels={snapshot.kernels}
              info={inspectorInfo}
            />
          </aside>
          <aside className="nn-flow-dock nn-flow-dock--right">
            <NetworkLossChart
              history={lossHistory}
              title={t.learningCurve}
              trainLabel={t.trainLoss}
              testLabel={t.testLoss}
              lossTrain={stats.lossTrain}
              lossTest={stats.lossTest}
            />
            <CnnTrainingPanel
              learningRate={snapshot.config.learningRate}
              optimizer={snapshot.config.optimizer}
              activation={snapshot.config.activation}
              regularization={snapshot.config.regularization}
              regularizationRate={snapshot.config.regularizationRate}
              onLearningRateChange={onLearningRateChange}
              onOptimizerChange={onOptimizerChange}
              onActivationChange={onActivationChange}
              onRegularizationChange={onRegularizationChange}
              onRegularizationRateChange={onRegularizationRateChange}
            />
            <div
              className="nn-weight-legend nn-weight-legend--dock"
              role="img"
              aria-label={t.weightsLegendAria}
            >
              <span className="nn-weight-legend__title">{t.weightsLegend}</span>
              <div className="nn-weight-legend__bar" />
              <div className="nn-weight-legend__scale">
                <span>−1</span>
                <span>0</span>
                <span>+1</span>
              </div>
            </div>
          </aside>
        </CnnFlowGraph>
      </div>
    </div>
  );
}
