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
  | "ensemble-playground"
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

// TransformerFlow and Ensemble Playground ship as self-contained pages in
// public/ (own runtime and styles), so the catalog embeds them as full pages
// in an iframe rather than mounting a component.
function publicPage(src: string, title: string) {
  return createElement("iframe", {
    className: "public-page-frame",
    src: `${import.meta.env.BASE_URL}${src}`,
    title,
  });
}

function TransformerFlowFrame() {
  return publicPage("transformer-constructor.html", "TransformerFlow");
}

function EnsemblePlaygroundFrame() {
  return publicPage("ensemble.html", "Ensemble Playground");
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
    id: "ensemble-playground",
    path: "/viz/ensemble-playground",
    titleKey: "vizEnsemblePlaygroundTitle",
    descriptionKey: "vizEnsemblePlaygroundDescription",
    component: EnsemblePlaygroundFrame,
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
