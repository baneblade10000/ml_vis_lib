import { createContext, useContext, type RefObject } from "react";

export type BoundaryStore = Record<string, number[][]>;

export type TrainingStats = {
  epoch: number;
  lossTrain: number;
  lossTest: number;
};

export const NetworkBoundaryRefContext = createContext<RefObject<BoundaryStore> | null>(null);

export const TrainingStatsRefContext = createContext<RefObject<TrainingStats> | null>(null);

export const TrainingLiveRefContext = createContext<RefObject<boolean> | null>(null);

/** Incremented when node heatmaps should repaint from boundaryRef. */
export const BoundaryPaintGenerationContext = createContext(0);

export function useBoundaryMatrix(nodeId: string): number[][] | undefined {
  const ref = useContext(NetworkBoundaryRefContext);
  return ref?.current?.[nodeId];
}
