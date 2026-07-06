import type { ComponentType } from "react";
import { ComputationalGraphPlayground, MonteCarloPiPlayground, NeuralNetworkPlayground } from "@ml-vis/react";
import type { PlaygroundMessages } from "../i18n";

export type VisualizationId = "neural-network" | "computational-graph" | "monte-carlo-pi";

export interface VisualizationEntry {
  id: VisualizationId;
  path: string;
  titleKey: keyof PlaygroundMessages;
  descriptionKey: keyof PlaygroundMessages;
  component: ComponentType;
}

export const visualizations: VisualizationEntry[] = [
  {
    id: "computational-graph",
    path: "/viz/computational-graph",
    titleKey: "vizComputationalGraphTitle",
    descriptionKey: "vizComputationalGraphDescription",
    component: ComputationalGraphPlayground,
  },
  {
    id: "neural-network",
    path: "/viz/neural-network",
    titleKey: "vizNeuralNetworkTitle",
    descriptionKey: "vizNeuralNetworkDescription",
    component: NeuralNetworkPlayground,
  },
  {
    id: "monte-carlo-pi",
    path: "/viz/monte-carlo-pi",
    titleKey: "vizMonteCarloPiTitle",
    descriptionKey: "vizMonteCarloPiDescription",
    component: MonteCarloPiPlayground,
  },
];

export function getVisualizationById(id: string): VisualizationEntry | undefined {
  return visualizations.find((viz) => viz.id === id);
}
