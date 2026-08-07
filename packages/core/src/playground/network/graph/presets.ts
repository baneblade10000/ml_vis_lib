import type { ActivationFunction } from "../nn";
import { buildNetwork } from "../nn";
import { constructInputIds } from "../inputs";
import { INPUTS } from "../inputs";
import { Activations, createGraphLink, GraphNode } from "./runtime";
import { ComputationalGraph, resetGraphIdCounter, syncGraphIdCounterFromGraph } from "./computational-graph";
import { layoutMlpFromLayers, normalizeGraphLayout } from "./mlp-layout";
import type { ArchitecturePresetId } from "./types";

function syncPositionsFromLayers(
  graph: ComputationalGraph,
  layers: GraphNode[][],
  inputIds: string[],
): void {
  layoutMlpFromLayers(graph, layers, inputIds);
}

/** Build MLP graph from layer shape (same as legacy buildNetwork). */
export function buildMlpGraph(
  networkShape: number[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
  inputIds: string[],
): ComputationalGraph {
  resetGraphIdCounter(1);
  const layered = buildNetwork(networkShape, activation, outputActivation, null, inputIds);
  const graph = new ComputationalGraph();
  graph.inputIds = [...inputIds];
  graph.outputId = layered[layered.length - 1][0].id;

  for (const layer of layered) {
    for (const node of layer) {
      const kind =
        inputIds.includes(node.id) ? "input"
        : node.id === graph.outputId ? "output"
        : "dense";
      const gNode = new GraphNode(
        node.id,
        kind,
        node.activation,
        false,
        kind === "input" ? INPUTS[node.id]?.label : undefined,
      );
      gNode.bias = node.bias;
      graph.nodes.set(node.id, gNode);
    }
  }

  for (let layerIdx = 1; layerIdx < layered.length; layerIdx++) {
    for (const node of layered[layerIdx]) {
      for (const link of node.inputLinks) {
        const source = graph.nodes.get(link.source.id)!;
        const dest = graph.nodes.get(link.dest.id)!;
        const gLink = createGraphLink(source, dest, null);
        gLink.weight = link.weight;
        gLink.id = link.id;
      }
    }
  }

  syncPositionsFromLayers(graph, layered as GraphNode[][], inputIds);
  graph.recomputeTopoOrder();
  syncGraphIdCounterFromGraph(graph);
  return graph;
}

/**
 * Residual block: stem → [dense(relu) → dense(linear)] + skip → sum(relu) → out
 *   inputs ──► stem ──► h1 ──► h2 ──┐
 *              └────────────────────► add ──► output
 */
export function buildResBlockGraph(
  inputIds: string[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
): ComputationalGraph {
  resetGraphIdCounter(1);
  const graph = new ComputationalGraph();

  for (const id of inputIds) {
    graph.addInputNode(id, INPUTS[id]?.label);
  }

  const stem = graph.addNode("dense", activation, { label: "Stem" });
  for (const id of inputIds) {
    graph.connect(id, stem.id);
  }

  const h1 = graph.addNode("dense", Activations.RELU, { label: "Conv1" });
  const h2 = graph.addNode("dense", Activations.LINEAR, { label: "Conv2" });
  graph.connect(stem.id, h1.id);
  graph.connect(h1.id, h2.id);

  const add = graph.addNode("sum", Activations.RELU, { label: "Add" });
  graph.connect(h2.id, add.id);
  graph.connect(stem.id, add.id, 1);

  const output = graph.addNode("output", outputActivation, { label: "Out" });
  graph.connect(add.id, output.id);

  graph.recomputeTopoOrder();
  normalizeGraphLayout(graph, inputIds);
  return graph;
}

/** Two residual blocks in sequence. */
export function buildMiniResNetGraph(
  inputIds: string[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
): ComputationalGraph {
  resetGraphIdCounter(1);
  const graph = new ComputationalGraph();

  for (const id of inputIds) {
    graph.addInputNode(id, INPUTS[id]?.label);
  }

  const stem = graph.addNode("dense", activation, { label: "Stem" });
  for (const id of inputIds) {
    graph.connect(id, stem.id);
  }

  let prev = stem.id;

  for (let block = 0; block < 2; block++) {
    const h1 = graph.addNode("dense", Activations.RELU, { label: `B${block + 1}a` });
    const h2 = graph.addNode("dense", Activations.LINEAR, { label: `B${block + 1}b` });
    const add = graph.addNode("sum", Activations.RELU, { label: `Add${block + 1}` });

    graph.connect(prev, h1.id);
    graph.connect(h1.id, h2.id);
    graph.connect(h2.id, add.id);
    graph.connect(prev, add.id, 1);
    prev = add.id;
  }

  const output = graph.addNode("output", outputActivation, { label: "Out" });
  graph.connect(prev, output.id);

  graph.recomputeTopoOrder();
  normalizeGraphLayout(graph, inputIds);
  return graph;
}

export function applyArchitecturePreset(
  preset: ArchitecturePresetId,
  options: {
    networkShape: number[];
    numHiddenLayers: number;
    activation: ActivationFunction;
    outputActivation: ActivationFunction;
    enabledFeatures: Record<string, boolean>;
  },
): ComputationalGraph {
  const inputIds = constructInputIds(options.enabledFeatures);

  if (preset === "resblock") {
    return buildResBlockGraph(inputIds, options.activation, options.outputActivation);
  }
  if (preset === "mini-resnet") {
    return buildMiniResNetGraph(inputIds, options.activation, options.outputActivation);
  }

  const shape = [inputIds.length, ...options.networkShape.slice(0, options.numHiddenLayers), 1];
  return buildMlpGraph(shape, options.activation, options.outputActivation, inputIds);
}

export function presetLabel(preset: ArchitecturePresetId): string {
  switch (preset) {
    case "mlp":
      return "MLP";
    case "resblock":
      return "ResBlock";
    case "mini-resnet":
      return "Mini ResNet";
    case "custom":
      return "Custom";
  }
}
