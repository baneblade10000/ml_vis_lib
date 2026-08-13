import { createContext, useContext, type RefObject } from "react";
import { FeatureMapSnapshot } from "@ml-vis/core/cnn";
import type { RfLayerMeta, RfSelection } from "./receptiveField";

/**
 * Ref-based contexts for the convolutional playground, mirroring the network
 * playground's pattern: mutable engine state (feature maps + training stats) is
 * ferried to the imperative canvas painters via refs, so painting during the
 * Play animation loop never triggers React re-renders.
 */
export type FeatureMapStore = Record<string, FeatureMapSnapshot>;

export type CnnTrainingStats = {
  epoch: number;
  lossTrain: number;
  lossTest: number;
  accTrain: number;
  accTest: number;
};

/** Live readout values updated every paint tick without React Flow remounts. */
export type CnnPlayViz = {
  probability: number;
  loss: number;
};

export type ReceptiveFieldApi = {
  layers: RfLayerMeta[];
  selection: RfSelection | null;
  pinned: boolean;
  setHover: (next: RfSelection | null) => void;
  pin: (next: RfSelection | null) => void;
  clear: () => void;
};

/** Click a multi-in conv kernel → show per-in slices on the previous layer. */
export type KernelExpandSelection = {
  layerId: string;
  filter: number;
  prevLayerId: string;
  /** One spatial kernel per input channel. */
  perIn2d?: number[][][];
  /** One 1-D kernel per input channel. */
  perIn1d?: number[][];
};

export type KernelExpandApi = {
  selection: KernelExpandSelection | null;
  /** Resolve the layer immediately before `layerId` in the pipeline. */
  prevLayerId: (layerId: string) => string | null;
  /** Select the conv layer in the inspector (right dock). */
  selectLayer: (layerId: string) => void;
  toggle: (next: KernelExpandSelection) => void;
  clear: () => void;
};

/** Ref to the latest per-layer feature-map snapshots for the inspected example. */
export const FeatureMapRefContext = createContext<RefObject<FeatureMapStore> | null>(null);
export const TrainingStatsRefContext = createContext<RefObject<CnnTrainingStats> | null>(null);
export const TrainingLiveRefContext = createContext<RefObject<boolean> | null>(null);
export const CnnPlayVizRefContext = createContext<RefObject<CnnPlayViz> | null>(null);
/** Incremented when node canvases should repaint from the feature-map ref. */
export const PaintGenerationContext = createContext(0);
export const ReceptiveFieldContext = createContext<ReceptiveFieldApi | null>(null);
export const KernelExpandContext = createContext<KernelExpandApi | null>(null);

export function useFeatureMaps(layerId: string): FeatureMapSnapshot | undefined {
  const ref = useContext(FeatureMapRefContext);
  return ref?.current?.[layerId];
}

export function useReceptiveField(): ReceptiveFieldApi | null {
  return useContext(ReceptiveFieldContext);
}

export function useKernelExpand(): KernelExpandApi | null {
  return useContext(KernelExpandContext);
}
