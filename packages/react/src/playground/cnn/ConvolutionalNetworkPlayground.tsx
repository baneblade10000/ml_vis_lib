import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CNN_DATASET_IDS_1D, CNN_DATASET_IDS_2D, DEFAULT_CNN_CONFIG, CnnTrainClient, canUseTrainWorkers, type CnnActivationId, type CnnCommandArgs, type CnnConfig, type CnnMode, type CnnRegularizationId, type CnnTrainSnapshot, type FeatureMapSnapshot, type ImageExample, type LayerShape, type LossHistoryPoint, type PlaygroundOptimizerId, type SignalExample } from "@ml-vis/core/cnn";
import { NetworkLossChart } from "../network/NetworkLossChart";
import { CnnFlowGraph } from "./CnnFlowGraph";
import { CnnArchitecturePanel } from "./CnnArchitecturePanel";
import { CnnTrainingPanel } from "./CnnTrainingPanel";
import { CnnDataPanel } from "./CnnDataPanel";
import { CnnInspector } from "./CnnInspector";
import { CnnGallery } from "./CnnGallery";
import { formatCnnNodeLabel } from "./cnnAdapter";
import { useCnnMessages } from "./messages";
import { paintAllFeatureMapsAfterCommit } from "./featureMapPaint";
import type { CnnPlayViz, CnnTrainingStats, FeatureMapStore } from "./featureMapContext";
import { PlayStartingOverlay, dismissStartingAfterPaint } from "../PlayStartingOverlay";

export interface ConvolutionalNetworkPlaygroundProps {
  initialMode?: CnnMode;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
  /** Burn WASM train worker factory (required — JS CNN compute removed). */
  createWorker: () => Worker;
}

function shapeLabel(shape: LayerShape): string {
  if (shape.kind === "2d") return `${shape.channels}×${shape.rows}×${shape.cols}`;
  return `${shape.channels}×${shape.length}`;
}

/** Config kernel sizes keyed by live layer id (skip input at layers[0]). */
function kernelSizesFromSnapshot(snap: CnnTrainSnapshot): Record<string, number> {
  const out: Record<string, number> = {};
  const specs = snap.config.layers;
  for (let i = 0; i < specs.length; i++) {
    const spec = specs[i]!;
    const view = snap.layers[i + 1];
    if (!view) continue;
    if (spec.kind === "conv2d" || spec.kind === "conv1d") {
      out[view.id] = Math.max(1, spec.kernelSize ?? 3);
    }
  }
  return out;
}

function kernelSpatialSize(kernels: unknown): number | undefined {
  if (!Array.isArray(kernels) || kernels.length === 0) return undefined;
  let cur: unknown = kernels[0];
  // Walk nested filter/in-channel wrappers until we hit a numeric row (2d) or vector (1d).
  while (Array.isArray(cur) && cur.length > 0 && Array.isArray(cur[0])) {
    cur = cur[0];
  }
  if (Array.isArray(cur)) return cur.length;
  return undefined;
}

/** Keep previous kernels only when spatial size still matches config (avoid stale k after resize). */
function mergeKernels<T>(
  next: T | undefined,
  prev: T | undefined,
  expectedK: number | undefined,
): T | undefined {
  const nextArr = next as unknown[] | undefined;
  if (nextArr && nextArr.length > 0) {
    const spatial = kernelSpatialSize(next);
    if (expectedK == null || spatial == null || spatial === expectedK) return next;
  }
  const prevArr = prev as unknown[] | undefined;
  if (!prevArr?.length) return nextArr?.length ? next : undefined;
  const prevSpatial = kernelSpatialSize(prev);
  if (expectedK != null && prevSpatial != null && prevSpatial !== expectedK) return undefined;
  return prev;
}

function specKindLabel(kind: string, t: ReturnType<typeof useCnnMessages>): string {
  switch (kind) {
    case "conv2d":
    case "conv1d":
      return t.paletteConv;
    case "pool2d":
    case "pool1d":
      return t.palettePool;
    case "gap2d":
    case "gap1d":
      return t.paletteGap;
    case "flatten":
      return t.flatten;
    case "dense":
      return t.paletteDense;
    default:
      return kind;
  }
}

export function ConvolutionalNetworkPlayground({
  initialMode = "2d",
  toolbarStart,
  toolbarEnd,
  createWorker,
}: ConvolutionalNetworkPlaygroundProps) {
  const t = useCnnMessages();
  const initialConfig = initialMode === "1d" ? DEFAULT_CNN_CONFIG["1d"] : DEFAULT_CNN_CONFIG["2d"];

  const [snapshot, setSnapshot] = useState<CnnTrainSnapshot | null>(null);
  const [playing, setPlaying] = useState(false);
  const [starting, setStarting] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [paintGeneration, setPaintGeneration] = useState(0);
  /** Curve + scalars — updated every worker tick (not throttled with RF snapshot). */
  const [lossHistory, setLossHistory] = useState<LossHistoryPoint[]>([]);
  const [chartLoss, setChartLoss] = useState({ train: 0, test: 0 });
  const requestPaint = useCallback(() => setPaintGeneration((n) => n + 1), []);

  const featureMapRef = useRef<FeatureMapStore>({});
  const statsRef = useRef<CnnTrainingStats>({
    epoch: 0,
    lossTrain: 0,
    lossTest: 0,
    accTrain: 0,
    accTest: 0,
  });
  const playVizRef = useRef<CnnPlayViz>({ probability: 0.5, loss: 0 });
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;
  const clientRef = useRef<CnnTrainClient | null>(null);
  const playingRef = useRef(false);
  playingRef.current = playing;
  const lastReactFlushRef = useRef(0);
  const awaitingFirstPlayPaintRef = useRef(false);
  const epochValRef = useRef<HTMLSpanElement>(null);
  const testAccValRef = useRef<HTMLSpanElement>(null);
  const trainAccValRef = useRef<HTMLSpanElement>(null);
  const testLossValRef = useRef<HTMLSpanElement>(null);

  const flushToolbarDom = useCallback((stats: CnnTrainingStats) => {
    if (epochValRef.current) epochValRef.current.textContent = stats.epoch.toLocaleString();
    if (testAccValRef.current) testAccValRef.current.textContent = `${(stats.accTest * 100).toFixed(0)}%`;
    if (trainAccValRef.current) trainAccValRef.current.textContent = `${(stats.accTrain * 100).toFixed(0)}%`;
    if (testLossValRef.current) testLossValRef.current.textContent = stats.lossTest.toFixed(3);
  }, []);

  const applySnapshot = useCallback(
    (snap: CnnTrainSnapshot) => {
      statsRef.current = { ...snap.stats };
      playVizRef.current = { probability: snap.probability, loss: snap.loss };
      const expectedK = kernelSizesFromSnapshot(snap);
      const store: FeatureMapStore = { ...featureMapRef.current };
      if (snap.featureMaps.length > 0) {
        for (const m of snap.featureMaps) {
          const prev = store[m.layerId];
          const k = expectedK[m.layerId];
          // Play dumps often omit kernels — keep last known only if size still matches config.
          store[m.layerId] = {
            ...m,
            kernels2d: mergeKernels(m.kernels2d, prev?.kernels2d, k),
            kernels1d: mergeKernels(m.kernels1d, prev?.kernels1d, k),
            kernels2dIn: mergeKernels(m.kernels2dIn, prev?.kernels2dIn, k),
            kernels1dIn: mergeKernels(m.kernels1dIn, prev?.kernels1dIn, k),
            biases: m.biases?.length ? m.biases : prev?.biases,
            matrix: m.matrix?.length ? m.matrix : prev?.matrix,
          };
        }
        featureMapRef.current = store;
      }
      flushToolbarDom(statsRef.current);
      if (awaitingFirstPlayPaintRef.current && playingRef.current) {
        awaitingFirstPlayPaintRef.current = false;
        dismissStartingAfterPaint(setStarting);
      }

      // Chart follows every tick (history is pushed each paint in the worker).
      if (snap.lossHistory.length > 0) {
        setLossHistory(snap.lossHistory);
      }
      setChartLoss({ train: snap.stats.lossTrain, test: snap.stats.lossTest });

      const now = performance.now();
      const playingNow = playingRef.current;
      // During play, refresh RF node data ~12 Hz; canvases paint via paintGeneration.
      if (!playingNow || now - lastReactFlushRef.current >= 80) {
        lastReactFlushRef.current = now;
        setSnapshot((prev) => {
          if (!prev) return snap;
          const mergedMaps =
            snap.featureMaps.length > 0
              ? snap.featureMaps.map((m) => {
                  const old = prev.featureMaps.find((x) => x.layerId === m.layerId);
                  const k = expectedK[m.layerId];
                  return {
                    ...m,
                    kernels2d: mergeKernels(m.kernels2d, old?.kernels2d, k),
                    kernels1d: mergeKernels(m.kernels1d, old?.kernels1d, k),
                    kernels2dIn: mergeKernels(m.kernels2dIn, old?.kernels2dIn, k),
                    kernels1dIn: mergeKernels(m.kernels1dIn, old?.kernels1dIn, k),
                    biases: m.biases?.length ? m.biases : old?.biases,
                    matrix: m.matrix?.length ? m.matrix : old?.matrix,
                  };
                })
              : prev.featureMaps;
          return {
            ...snap,
            featureMaps: mergedMaps,
            kernels:
              Object.keys(snap.kernels).length > 0 ? snap.kernels : prev.kernels,
          };
        });
      }
      requestPaint();
    },
    [flushToolbarDom, requestPaint],
  );

  // Boot train worker (Burn WASM from playground, or legacy factory).
  useEffect(() => {
    if (!canUseTrainWorkers()) {
      console.warn("[cnn] Web Workers unavailable; CNN playground requires Workers");
      return;
    }
    const client = new CnnTrainClient({
      createWorker,
      onTick: applySnapshot,
      onError: (message) => console.error("[cnn train worker]", message),
    });
    clientRef.current = client;
    let cancelled = false;
    void client.init(structuredClone(initialConfig)).then((s) => {
      if (!cancelled) applySnapshot(s);
    });
    return () => {
      cancelled = true;
      client.dispose();
      clientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createWorker]);

  useEffect(() => {
    paintAllFeatureMapsAfterCommit();
  }, [paintGeneration]);

  // Drive play/pause on the worker.
  useEffect(() => {
    const client = clientRef.current;
    if (!client) return;
    // Single-thread Rust is faster than shard IPC on this tiny net.
    if (playing) {
      awaitingFirstPlayPaintRef.current = true;
      client.play(96);
    } else client.pause();
  }, [playing]);

  useEffect(() => {
    if (!playing) {
      awaitingFirstPlayPaintRef.current = false;
      setStarting(false);
    }
  }, [playing]);

  const cmd = useCallback(
    <K extends keyof CnnCommandArgs>(
      name: K,
      ...rest: CnnCommandArgs[K] extends void ? [] : [args: CnnCommandArgs[K]]
    ) => {
      clientRef.current?.command(name, ...rest);
    },
    [],
  );

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
    cmd("setDataset", { dataset: id as CnnConfig["dataset"] });
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

  const featureMaps: FeatureMapSnapshot[] = snapshot?.featureMaps ?? [];
  const convBiases = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const m of featureMaps) {
      if (m.biases && m.biases.length > 0) out[m.layerId] = m.biases;
    }
    return out;
  }, [featureMaps]);
  const layerKernelSizes = useMemo(
    () => (snapshot ? kernelSizesFromSnapshot(snapshot) : {}),
    [snapshot],
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
  const stats = snapshot.stats;
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
            <span ref={epochValRef} className="value">
              {stats.epoch.toLocaleString()}
            </span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.testAcc}</span>
            <span ref={testAccValRef} className="value">
              {(stats.accTest * 100).toFixed(0)}%
            </span>
          </div>
          <div className="nn-toolbar-stat nn-toolbar-stat--train">
            <span className="label">{t.trainAcc}</span>
            <span ref={trainAccValRef} className="value">
              {(stats.accTrain * 100).toFixed(0)}%
            </span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.testLoss}</span>
            <span ref={testLossValRef} className="value">
              {stats.lossTest.toFixed(3)}
            </span>
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
          layerKernelSizes={layerKernelSizes}
          onSelectNode={setSelectedLayerId}
          featureMapRef={featureMapRef}
          statsRef={statsRef}
          trainingLiveRef={trainingLiveRef}
          playVizRef={playVizRef}
          loss={loss}
          probability={probability}
          fillHeight
        >
          <PlayStartingOverlay visible={starting} label={t.startingHint} />
          <aside className="nn-flow-dock nn-flow-dock--left nn-flow-dock--wide">
            <CnnArchitecturePanel
              layers={snapshot.config.layers}
              selectedIndex={selectedSpecIndex}
              onSelectLayer={(idx) => {
                const layer = snapshot.layers[idx + 1];
                setSelectedLayerId(layer ? layer.id : null);
              }}
              onSetFilters={onSetFilters}
              onSetKernelSize={onSetKernelSize}
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
          </aside>
          <aside className="nn-flow-dock nn-flow-dock--right">
            <NetworkLossChart
              history={lossHistory}
              title={t.learningCurve}
              trainLabel={t.trainLoss}
              testLabel={t.testLoss}
              lossTrain={chartLoss.train}
              lossTest={chartLoss.test}
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
              <div className="nn-weight-legend__bar nn-weight-legend__bar--zero-white" />
              <div className="nn-weight-legend__scale">
                <span>−1</span>
                <span>0</span>
                <span>+1</span>
              </div>
            </div>
            <CnnInspector
              selectedLayerId={selectedLayerId}
              kernels={snapshot.kernels}
              biases={convBiases}
              info={inspectorInfo}
            />
          </aside>
        </CnnFlowGraph>
      </div>
    </div>
  );
}
