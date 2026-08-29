import { createElement, type ComponentType } from "react";
import {
  ComputationalGraphPlayground,
  ConvolutionalNetworkPlayground,
  NeuralNetworkPlayground,
  TransformerPlayground,
  // SignalPlayground, // hidden from the catalog for now — uncomment to re-enable
} from "@ml-vis/react";
import { createCnnTrainWorker } from "../wasm/cnn/createCnnTrainWorker";
import { createTransformerTrainWorker } from "../wasm/transformer/createTransformerTrainWorker";
import type { PlaygroundMessages } from "../i18n";

export type VisualizationId =
  | "neural-network"
  | "computational-graph"
  | "convolutional-network"
  | "transformer"
  | "signal-lab";

export interface VisualizationEntry {
  id: VisualizationId;
  path: string;
  titleKey: keyof PlaygroundMessages;
  descriptionKey: keyof PlaygroundMessages;
  component: ComponentType;
}

function ConvolutionalNetworkWithWasm() {
  return createElement(ConvolutionalNetworkPlayground, {
    createWorker: createCnnTrainWorker,
  });
}

function TransformerWithWasm() {
  return createElement(TransformerPlayground, {
    createWorker: createTransformerTrainWorker,
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
    id: "transformer",
    path: "/viz/transformer",
    titleKey: "vizTransformerTitle",
    descriptionKey: "vizTransformerDescription",
    component: TransformerWithWasm,
  },
  {
    id: "convolutional-network",
    path: "/viz/convolutional-network",
    titleKey: "vizConvolutionalNetworkTitle",
    descriptionKey: "vizConvolutionalNetworkDescription",
    component: ConvolutionalNetworkWithWasm,
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
