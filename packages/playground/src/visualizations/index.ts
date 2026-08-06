import type { ComponentType } from "react";
import {
  ComputationalGraphPlayground,
  ConvolutionalNetworkPlayground,
  NeuralNetworkPlayground,
} from "@ml-vis/react";
import type { PlaygroundMessages } from "../i18n";

export type VisualizationId = "neural-network" | "computational-graph" | "convolutional-network";

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
    id: "convolutional-network",
    path: "/viz/convolutional-network",
    titleKey: "vizConvolutionalNetworkTitle",
    descriptionKey: "vizConvolutionalNetworkDescription",
    component: ConvolutionalNetworkPlayground,
  },
];

export function getVisualizationById(id: string): VisualizationEntry | undefined {
  return visualizations.find((viz) => viz.id === id);
}
