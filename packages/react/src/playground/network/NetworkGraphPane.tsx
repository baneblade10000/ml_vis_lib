import { memo, type RefObject } from "react";
import {
  PlaygroundEngine,
  type Dataset1DId,
  type LossHistoryPoint,
  type NetworkAnyDatasetId,
  type DatasetId as NetworkDatasetId,
  type NetworkPlaygroundConfig,
  type NetworkProblemType,
} from "@ml-vis/core/network";
import { ReactFlowNetworkGraph } from "./ReactFlowNetworkGraph";
import { NetworkPalette } from "./NetworkPalette";
import { NetworkArchitecturePanel } from "./NetworkArchitecturePanel";
import { NetworkDataPanel } from "./NetworkDataPanel";
import { NetworkLossChart } from "./NetworkLossChart";
import { NetworkTrainingPanel } from "./NetworkTrainingPanel";
import { NetworkInspector } from "./NetworkInspector";
import { DatasetThumbnail } from "./DatasetThumbnail";
import { useNetworkMessages } from "./messages";
import type { CurveStore, TrainingStats } from "./NetworkBoundaryContext";
import type { EdgeVizMode, LayoutVizMode } from "./graphAdapter";
import { PlayStartingOverlay } from "../PlayStartingOverlay";

const DATASETS_2D_CLASSIFICATION: NetworkDatasetId[] = ["circle", "xor", "gauss", "spiral"];
const DATASETS_2D_REGRESSION: NetworkDatasetId[] = ["sinSin"];
const DATASETS_1D_CLASSIFICATION: Dataset1DId[] = ["gauss1d", "threshold", "twoClusters"];
const DATASETS_1D_REGRESSION: Dataset1DId[] = ["sine", "linear", "cubic", "step"];

export type GraphPaneProps = {
  engineRef: RefObject<PlaygroundEngine | null>;
  boundaryRef: RefObject<Record<string, number[][]>>;
  curvesRef: RefObject<CurveStore>;
  targetCurveRef: RefObject<number[] | null>;
  statsRef: RefObject<TrainingStats>;
  trainingLiveRef: RefObject<boolean>;
  paintGeneration: number;
  tick: number;
  playing: boolean;
  playStarting?: boolean;
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

export const NetworkGraphPane = memo(function NetworkGraphPane({
  engineRef,
  boundaryRef,
  curvesRef,
  targetCurveRef,
  statsRef,
  trainingLiveRef,
  paintGeneration,
  tick,
  playing,
  playStarting = false,
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
      heatmapPreset={cfg.heatmapPreset}
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
      <PlayStartingOverlay visible={playStarting} label={t.startingHint} />
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
          aria-label="Weight color scale from −1 (indigo) through 0 (light-blue) to +1 (cyan-blue)"
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
