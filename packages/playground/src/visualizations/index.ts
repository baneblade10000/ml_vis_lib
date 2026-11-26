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

interface VisualizationBase {
  id: VisualizationId;
  path: string;
  titleKey: keyof PlaygroundMessages;
  descriptionKey: keyof PlaygroundMessages;
}

/** A visualization mounted by the viz page as a React component. */
export interface ComponentVisualization extends VisualizationBase {
  component: ComponentType;
  embeddedPage?: never;
}

/** A tab of an embedded page, shown in the catalog header. */
export interface EmbeddedPageTab {
  /** What the page receives in the tab message. */
  id: string;
  labelKey: keyof PlaygroundMessages;
}

/** A button of an embedded page, shown in the catalog header beside the tabs. */
export interface EmbeddedPageAction {
  /** What the page receives in the action message. */
  id: string;
  labelKey: keyof PlaygroundMessages;
}

/**
 * A self-contained page in public/ (own runtime and styles). The catalog embeds
 * it as a full page in an iframe instead of mounting a component.
 */
export interface EmbeddedPage {
  /** Path under public/, relative to the app base. */
  src: string;
  title: string;
  tabs?: EmbeddedPageTab[];
  actions?: EmbeddedPageAction[];
  defaultTab?: string;
}

/** Posted to an embedded page when one of its header tabs is picked. */
export interface EmbeddedPageTabMessage {
  type: "ml-vis:tab";
  id: string;
}

/** Posted to an embedded page when one of its header actions is picked. */
export interface EmbeddedPageActionMessage {
  type: "ml-vis:action";
  id: string;
}

/**
 * Posted by an embedded page: the tab it switched to on its own, and what its
 * header shows beyond the tabs declared here.
 */
export interface EmbeddedPageHeaderMessage {
  type: "ml-vis:header";
  tab?: string;
  title?: string;
  badge?: string;
}

/** A visualization the viz page embeds as a full page in an iframe. */
export interface EmbeddedVisualization extends VisualizationBase {
  component?: never;
  embeddedPage: EmbeddedPage;
}

export type VisualizationEntry = ComponentVisualization | EmbeddedVisualization;

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
    id: "transformer-flow",
    path: "/viz/transformer-flow",
    titleKey: "vizTransformerFlowTitle",
    descriptionKey: "vizTransformerFlowDescription",
    // #embedded: the catalog carries the page's header — its tabs, the model
    // badge and the open/save actions; without the hash the page shows its own.
    embeddedPage: {
      src: "transformer-constructor.html#embedded",
      title: "TransformerFlow",
      tabs: [
        { id: "training", labelKey: "transformerFlowTabTraining" },
        { id: "builder", labelKey: "transformerFlowTabBuilder" },
        { id: "data", labelKey: "transformerFlowTabData" },
      ],
      actions: [
        { id: "import", labelKey: "transformerFlowActionOpen" },
        { id: "export", labelKey: "transformerFlowActionSave" },
      ],
    },
  },
  {
    id: "ensemble-playground",
    path: "/viz/ensemble-playground",
    titleKey: "vizEnsemblePlaygroundTitle",
    descriptionKey: "vizEnsemblePlaygroundDescription",
    // #embedded tells the page it is framed by the catalog, which carries
    // its tabs; without the hash the page shows its own header.
    embeddedPage: {
      src: "ensemble.html#embedded",
      title: "Ensemble Playground",
      defaultTab: "viz",
      tabs: [
        { id: "data", labelKey: "ensembleTabData" },
        { id: "viz", labelKey: "ensembleTabViz" },
        { id: "edit", labelKey: "ensembleTabTableEditor" },
        { id: "tableForest", labelKey: "ensembleTabTableForest" },
        { id: "tableBoost", labelKey: "ensembleTabTableBoost" },
      ],
    },
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
