import { createContext, useContext, type RefObject } from "react";
import type { FeatureMapSnapshot } from "@ml-vis/core";

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

/** Ref to the latest per-layer feature-map snapshots for the inspected example. */
export const FeatureMapRefContext = createContext<RefObject<FeatureMapStore> | null>(null);
export const TrainingStatsRefContext = createContext<RefObject<CnnTrainingStats> | null>(null);
export const TrainingLiveRefContext = createContext<RefObject<boolean> | null>(null);
export const CnnPlayVizRefContext = createContext<RefObject<CnnPlayViz> | null>(null);
/** Incremented when node canvases should repaint from the feature-map ref. */
export const PaintGenerationContext = createContext(0);

export function useFeatureMaps(layerId: string): FeatureMapSnapshot | undefined {
  const ref = useContext(FeatureMapRefContext);
  return ref?.current?.[layerId];
}
