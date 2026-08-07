import { createContext, useContext, type RefObject } from "react";
import type { NetworkDataMode, NetworkProblemType } from "@ml-vis/core";

export type BoundaryStore = Record<string, number[][]>;
export type CurveStore = Record<string, number[]>;

export type TrainingStats = {
  epoch: number;
  lossTrain: number;
  lossTest: number;
};

export type NetworkVizMode = {
  dataMode: NetworkDataMode;
  problemType: NetworkProblemType;
};

export const NetworkBoundaryRefContext = createContext<RefObject<BoundaryStore> | null>(null);

export const NetworkCurveRefContext = createContext<RefObject<CurveStore> | null>(null);

export const NetworkTargetCurveRefContext = createContext<RefObject<number[] | null> | null>(null);

export const NetworkVizModeContext = createContext<NetworkVizMode>({
  dataMode: "2d",
  problemType: "classification",
});

export const TrainingStatsRefContext = createContext<RefObject<TrainingStats> | null>(null);

export const TrainingLiveRefContext = createContext<RefObject<boolean> | null>(null);

/** Incremented when node heatmaps/curves should repaint from refs. */
export const BoundaryPaintGenerationContext = createContext(0);

export function useBoundaryMatrix(nodeId: string): number[][] | undefined {
  const ref = useContext(NetworkBoundaryRefContext);
  return ref?.current?.[nodeId];
}

export function useNodeCurve(nodeId: string): number[] | undefined {
  const ref = useContext(NetworkCurveRefContext);
  return ref?.current?.[nodeId];
}
