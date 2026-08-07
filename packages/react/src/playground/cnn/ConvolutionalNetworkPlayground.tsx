import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CnnEngine,
  DEFAULT_CNN_CONFIG,
  CNN_DATASET_IDS_2D,
  CNN_DATASET_IDS_1D,
  type CnnActivationId,
  type CnnMode,
  type CnnRegularizationId,
  type FeatureMapSnapshot,
  type ImageExample,
  type LayerShape,
  type LossHistoryPoint,
  type PlaygroundOptimizerId,
  type SignalExample,
} from "@ml-vis/core";
import { NetworkLossChart } from "../network/NetworkLossChart";
import { CnnFlowGraph } from "./CnnFlowGraph";
import { CnnPalette } from "./CnnPalette";
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

export function ConvolutionalNetworkPlayground({
  initialMode = "2d",
  toolbarStart,
  toolbarEnd,
}: ConvolutionalNetworkPlaygroundProps) {
  // Lazy init: engine construction runs dataset generation + a forward pass.
  const engineRef = useRef<CnnEngine | null>(null);
  if (engineRef.current === null) {
    const config = initialMode === "1d" ? DEFAULT_CNN_CONFIG["1d"] : DEFAULT_CNN_CONFIG["2d"];
    engineRef.current = new CnnEngine(structuredClone(config));
  }
  const t = useCnnMessages();

  const [version, setVersion] = useState(0);
  const bump = useCallback(() => setVersion((n) => n + 1), []);
  const [paintGeneration, setPaintGeneration] = useState(0);
  const requestPaint = useCallback(() => setPaintGeneration((n) => n + 1), []);

  const [playing, setPlaying] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);

  // Refs that ferry mutable engine state to the imperative canvas painters.
  const featureMapRef = useRef<FeatureMapStore>({});
  const statsRef = useRef<CnnTrainingStats>(engineRef.current.stats());
  const trainingLiveRef = useRef(false);
  trainingLiveRef.current = playing;

  const [stats, setStats] = useState<CnnTrainingStats>(() => engineRef.current!.stats());
  const [lossHistory, setLossHistory] = useState<LossHistoryPoint[]>(() =>
    engineRef.current!.lossHistory.map((p) => ({ ...p })),
  );

  const engine = engineRef.current!;

  const syncState = useCallback(() => {
    const e = engineRef.current!;
    statsRef.current = e.stats();
    setStats({ ...statsRef.current });
    setLossHistory(e.lossHistory.map((p) => ({ ...p })));
    // Refresh feature-map snapshots for the inspected example.
    const maps = e.featureMaps();
    const store: FeatureMapStore = {};
    for (const m of maps) store[m.layerId] = m;
    featureMapRef.current = store;
  }, []);

  // Initial paint on mount and whenever the version/paint generation bumps.
  useEffect(() => {
    syncState();
    paintAllFeatureMapsAfterCommit();
  }, [version, paintGeneration, syncState]);

  // Play loop: run a target number of epochs/sec, repainting periodically.
  useEffect(() => {
    if (!playing) return;
    const epochsPerSec = 12;
    let raf = 0;
    let lastTime = performance.now();
    let lastPaint = 0;
    let epochBank = 0;
    const paintIntervalMs = 1000 / 20;
    const loop = (now: number) => {
      const e = engineRef.current!;
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      epochBank = Math.min(epochBank + dt * epochsPerSec, 2);
      const steps = Math.floor(epochBank);
      epochBank -= steps;
      if (steps > 0) {
        for (let i = 0; i < steps; i++) e.trainEpoch();
        if (now - lastPaint >= paintIntervalMs) {
          e.refreshMetrics();
          e.pushLossHistory();
          const maps = e.featureMaps();
          const store: FeatureMapStore = {};
          for (const m of maps) store[m.layerId] = m;
          featureMapRef.current = store;
          statsRef.current = e.stats();
          setStats({ ...statsRef.current });
          setLossHistory(e.lossHistory.map((p) => ({ ...p })));
          paintAllFeatureMaps();
          lastPaint = now;
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // On pause, refresh full-quality metrics + snapshots.
  useEffect(() => {
    if (playing) return;
    const e = engineRef.current!;
    e.refreshMetrics();
    e.pushLossHistory();
    syncState();
    requestPaint();
    bump();
  }, [playing, syncState, requestPaint, bump]);

  const reset = useCallback(() => {
    engineRef.current!.reset();
    setPlaying(false);
    setSelectedLayerId(null);
    syncState();
    requestPaint();
    bump();
  }, [syncState, requestPaint, bump]);

  const resetWeights = useCallback(() => {
    engineRef.current!.resetWeights();
    setPlaying(false);
    syncState();
    requestPaint();
    bump();
  }, [syncState, requestPaint, bump]);

  const step = useCallback(() => {
    engineRef.current!.step();
    syncState();
    requestPaint();
    bump();
  }, [syncState, requestPaint, bump]);

  const onModeChange = useCallback(
    (mode: CnnMode) => {
      engineRef.current!.setMode(mode);
      setPlaying(false);
      setSelectedLayerId(null);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onDatasetChange = useCallback(
    (id: string) => {
      engineRef.current!.setDataset(id as never);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onActivationChange = useCallback(
    (a: CnnActivationId) => {
      engineRef.current!.setActivation(a);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onLearningRateChange = useCallback((lr: number) => {
    engineRef.current!.setLearningRate(lr);
  }, []);

  const onOptimizerChange = useCallback((optimizer: PlaygroundOptimizerId) => {
    engineRef.current!.setOptimizer(optimizer);
    bump();
  }, [bump]);

  const onBatchSizeChange = useCallback((bs: number) => {
    engineRef.current!.setBatchSize(bs);
  }, []);

  const onRegularizationChange = useCallback((regularization: CnnRegularizationId) => {
    engineRef.current!.setRegularization(regularization);
    bump();
  }, [bump]);

  const onRegularizationRateChange = useCallback((rate: number) => {
    engineRef.current!.setRegularizationRate(rate);
    bump();
  }, [bump]);

  const onNoiseChange = useCallback(
    (n: number) => {
      engineRef.current!.updateDataParams({ noise: n });
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onTrainRatioChange = useCallback(
    (r: number) => {
      engineRef.current!.updateDataParams({ percTrainData: r });
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onRegenerateData = useCallback(() => {
    engineRef.current!.regenerateData();
    syncState();
    requestPaint();
    bump();
  }, [syncState, requestPaint, bump]);

  // Architecture mutators — operate on config layer specs, then rebuild.
  const selectedSpecIndex = useMemo(() => {
    if (!selectedLayerId) return null;
    const layer = engine.layers.find((l) => l.id === selectedLayerId);
    if (!layer) return null;
    // The visible pipeline = input + config.layers + output; map back to spec idx.
    const idx = engine.layers.indexOf(layer) - 1;
    return idx >= 0 && idx < engine.config.layers.length ? idx : null;
  }, [engine, selectedLayerId]);

  const onAddLayer = useCallback(
    (kind: "conv" | "pool" | "dense") => {
      const mode = engineRef.current!.config.mode;
      const convKind = mode === "2d" ? "conv2d" : "conv1d";
      const poolKind = mode === "2d" ? "pool2d" : "pool1d";
      const specKind = kind === "conv" ? convKind : kind === "pool" ? poolKind : "dense";
      // Insert before the dense/output head if present, else at end.
      const cfg = engineRef.current!.config;
      let at = cfg.layers.length;
      const headIdx = cfg.layers.findIndex((l) => l.kind === "flatten" || l.kind === "dense");
      if (headIdx >= 0 && (specKind === "conv2d" || specKind === "conv1d" || specKind === "pool2d" || specKind === "pool1d")) {
        at = headIdx;
      }
      const spec =
        specKind === "conv2d" || specKind === "conv1d"
          ? { kind: specKind as "conv2d" | "conv1d", filters: 4, kernelSize: mode === "2d" ? 3 : 5, activation: cfg.activation }
          : specKind === "pool2d" || specKind === "pool1d"
            ? { kind: specKind as "pool2d" | "pool1d", poolKind: "max" as const }
            : { kind: "dense" as const, units: 1, activation: "linear" as const };
      engineRef.current!.addLayer(spec, at);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onRemoveLayer = useCallback(
    (index: number) => {
      engineRef.current!.removeLayer(index);
      setSelectedLayerId(null);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onSetFilters = useCallback(
    (index: number, filters: number) => {
      engineRef.current!.setLayerFilters(index, filters);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onSetKernelSize = useCallback(
    (index: number, kernelSize: number) => {
      const e = engineRef.current!;
      const spec = e.config.layers[index];
      if (!spec) return;
      spec.kernelSize = kernelSize;
      e.rebuildAfterSpecEdit(index);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onSetUnits = useCallback(
    (index: number, units: number) => {
      engineRef.current!.setLayerUnits(index, units);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onSetPoolKind = useCallback(
    (index: number, poolKind: "max" | "avg") => {
      const e = engineRef.current!;
      const spec = e.config.layers[index];
      if (!spec) return;
      spec.poolKind = poolKind;
      e.rebuildAfterSpecEdit(index);
      setPlaying(false);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  const onSelectExample = useCallback(
    (index: number) => {
      engineRef.current!.setInspectedExample(index);
      syncState();
      requestPaint();
      bump();
    },
    [syncState, requestPaint, bump],
  );

  // Derive display data.
  const mode = engine.config.mode;
  const datasetIds = mode === "2d" ? CNN_DATASET_IDS_2D : CNN_DATASET_IDS_1D;
  const datasetLabels = mode === "2d" ? t.datasetLabels2D : t.datasetLabels1D;
  const galleryExamples = (engine.testData.length ? engine.testData : engine.trainData) as (
    ImageExample | SignalExample
  )[];
  const galleryPredictions = galleryExamples.map((ex) => engine.predict(ex));
  const inspectedIdx = engine.inspectedExampleIndex;

  const featureMaps: FeatureMapSnapshot[] = Object.values(featureMapRef.current);
  const loss = stats.lossTest;
  const probability = (() => {
    const data = engine.trainData;
    if (!data.length) return 0.5;
    const idx = Math.min(inspectedIdx, data.length - 1);
    return engine.predict(data[idx]);
  })();

  // Inspector info for the selected layer.
  const inspectorInfo = useMemo(() => {
    if (!selectedLayerId) return null;
    const layer = engine.layers.find((l) => l.id === selectedLayerId);
    if (!layer) return null;
    const shapes = engine.pipelineShapes();
    const idx = engine.layers.indexOf(layer);
    const inShape = shapes[idx - 1];
    const outShape = shapes[idx];
    return {
      kind: specKindLabel(layer.kind, t),
      label: formatCnnNodeLabel(layer, t),
      inputShape: inShape ? shapeLabel(inShape) : "—",
      outputShape: outShape ? shapeLabel(outShape) : "—",
      params: layer.paramCount(),
    };
  }, [engine, selectedLayerId, t, version]);

  const kernelSnapshots = useMemo(() => engine.kernelSnapshots(), [engine, version]);

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
              className={`tf-flat-switch__btn${mode === "1d" ? " selected" : ""}`}
              onClick={() => onModeChange("1d")}
            >
              {t.mode1D}
            </button>
            <button
              type="button"
              className={`tf-flat-switch__btn${mode === "2d" ? " selected" : ""}`}
              onClick={() => onModeChange("2d")}
            >
              {t.mode2D}
            </button>
          </div>
        </div>

        <div className="tf-toolbar-group tf-toolbar-group--params">
          <div className="tf-toolbar-stat">
            <span className="label">{t.epoch}</span>
            <span className="value">{stats.epoch.toLocaleString()}</span>
          </div>
          <div className="tf-toolbar-stat">
            <span className="label">{t.testAcc}</span>
            <span className="value">{(stats.accTest * 100).toFixed(0)}%</span>
          </div>
          <div className="tf-toolbar-stat tf-toolbar-stat--train">
            <span className="label">{t.trainAcc}</span>
            <span className="value">{(stats.accTrain * 100).toFixed(0)}%</span>
          </div>
          <div className="tf-toolbar-stat">
            <span className="label">{t.testLoss}</span>
            <span className="value">{stats.lossTest.toFixed(3)}</span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="tf-immersive-body">
        <CnnFlowGraph
          engine={engine}
          selectedNodeId={selectedLayerId}
          paintGeneration={paintGeneration}
          featureMaps={featureMaps}
          onSelectNode={setSelectedLayerId}
          onDropLayer={onAddLayer}
          featureMapRef={featureMapRef}
          statsRef={statsRef}
          trainingLiveRef={trainingLiveRef}
          loss={loss}
          probability={probability}
          fillHeight
        >
          <aside className="tf-flow-dock tf-flow-dock--left tf-flow-dock--wide">
            <CnnPalette onAddLayer={onAddLayer} />
            <CnnArchitecturePanel
              layers={engine.config.layers}
              selectedIndex={selectedSpecIndex}
              onSelectLayer={(idx) => {
                const layer = engine.layers[idx + 1];
                setSelectedLayerId(layer ? layer.id : null);
              }}
              onRemoveLayer={onRemoveLayer}
              onSetFilters={onSetFilters}
              onSetKernelSize={onSetKernelSize}
              onSetUnits={onSetUnits}
              onSetPoolKind={onSetPoolKind}
            />
            <div className="tf-flow-dock-section">
              <button
                type="button"
                className="tf-btn tf-btn--secondary tf-reset-weights"
                onClick={resetWeights}
              >
                {t.resetWeights}
              </button>
            </div>
            <div className="tf-flow-dock-section">
              <CnnGallery
                mode={mode}
                examples={galleryExamples}
                predictions={galleryPredictions}
                inspectedIndex={inspectedIdx}
                onSelectExample={onSelectExample}
                datasetId={engine.config.dataset}
                onSelectDataset={onDatasetChange}
                datasetIds={datasetIds}
                datasetLabels={datasetLabels}
              />
              <CnnDataPanel
                batchSize={engine.config.batchSize}
                noise={engine.config.noise}
                percTrainData={engine.config.percTrainData}
                onBatchSizeChange={onBatchSizeChange}
                onNoiseChange={onNoiseChange}
                onTrainRatioChange={onTrainRatioChange}
                onRegenerateData={onRegenerateData}
              />
            </div>
            <CnnInspector
              selectedLayerId={selectedLayerId}
              kernels={kernelSnapshots}
              info={inspectorInfo}
            />
          </aside>
          <aside className="tf-flow-dock tf-flow-dock--right">
            <NetworkLossChart
              history={lossHistory}
              title={t.learningCurve}
              trainLabel={t.trainLoss}
              testLabel={t.testLoss}
              lossTrain={stats.lossTrain}
              lossTest={stats.lossTest}
            />
            <CnnTrainingPanel
              learningRate={engine.config.learningRate}
              optimizer={engine.config.optimizer}
              activation={engine.config.activation}
              regularization={engine.config.regularization}
              regularizationRate={engine.config.regularizationRate}
              onLearningRateChange={onLearningRateChange}
              onOptimizerChange={onOptimizerChange}
              onActivationChange={onActivationChange}
              onRegularizationChange={onRegularizationChange}
              onRegularizationRateChange={onRegularizationRateChange}
            />
            <div
              className="tf-weight-legend tf-weight-legend--dock"
              role="img"
              aria-label={t.weightsLegendAria}
            >
              <span className="tf-weight-legend__title">{t.weightsLegend}</span>
              <div className="tf-weight-legend__bar" />
              <div className="tf-weight-legend__scale">
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
