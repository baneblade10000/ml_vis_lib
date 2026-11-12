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
  | "transformer-flow"
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

// TransformerFlow ships as a self-contained page in public/ (own runtime and
// styles), so the catalog entry embeds it rather than mounting a component.
function TransformerFlowFrame() {
  return createElement("iframe", {
    className: "transformer-flow-frame",
    src: `${import.meta.env.BASE_URL}transformer-constructor.html`,
    title: "TransformerFlow",
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
    id: "transformer-flow",
    path: "/viz/transformer-flow",
    titleKey: "vizTransformerFlowTitle",
    descriptionKey: "vizTransformerFlowDescription",
    component: TransformerFlowFrame,
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
