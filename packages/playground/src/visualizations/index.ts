import type { ComponentType } from "react";
import { MonteCarloPiPlayground, NeuralNetworkPlayground } from "@ml-vis/react";
import type { PlaygroundMessages } from "../i18n";

export type VisualizationId = "neural-network" | "monte-carlo-pi";

export interface VisualizationEntry {
  id: VisualizationId;
  path: string;
  titleKey: keyof PlaygroundMessages;
  descriptionKey: keyof PlaygroundMessages;
  component: ComponentType;
}

export const visualizations: VisualizationEntry[] = [
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
