import { createElement, type ComponentType } from "react";
import {
  ComputationalGraphPlayground,
  ConvolutionalNetworkPlayground,
  NeuralNetworkPlayground,
  // SignalPlayground, // hidden from the catalog for now — uncomment to re-enable
} from "@ml-vis/react";
import { createBurnCnnTrainWorker } from "../burn/createBurnCnnTrainWorker";
import type { PlaygroundMessages } from "../i18n";

export type VisualizationId =
  | "neural-network"
  | "computational-graph"
  | "convolutional-network"
  | "signal-lab";

export interface VisualizationEntry {
  id: VisualizationId;
  path: string;
  titleKey: keyof PlaygroundMessages;
  descriptionKey: keyof PlaygroundMessages;
  component: ComponentType;
}

function ConvolutionalNetworkWithBurn() {
  return createElement(ConvolutionalNetworkPlayground, {
    createWorker: createBurnCnnTrainWorker,
  });
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
    component: ConvolutionalNetworkWithBurn,
  },
  // {
  //   id: "signal-lab",
  //   path: "/viz/signal-lab",
  //   titleKey: "vizSignalLabTitle",
  //   descriptionKey: "vizSignalLabDescription",
  //   component: SignalPlayground,
  // },
];

export function getVisualizationById(id: string): VisualizationEntry | undefined {
  return visualizations.find((viz) => viz.id === id);
}
