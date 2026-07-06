import {
  Activations,
  Link,
  Node,
  RegularizationFunction,
  type ActivationFunction,
  type RegularizationFunction as RegFn,
} from "../nn";
import type { GraphNodeKind } from "./types";

/** Runtime node extended with graph topology kind. */
export class GraphNode extends Node {
  kind: GraphNodeKind;
  label?: string;

  constructor(
    id: string,
    kind: GraphNodeKind,
    activation: ActivationFunction,
    initZero?: boolean,
    label?: string,
  ) {
    super(id, activation, initZero);
    this.kind = kind;
    this.label = label;
    if (kind === "sum") {
      this.bias = 0;
    }
  }

  override updateOutput(): number {
    if (this.kind === "input") {
      return this.output;
    }
    if (this.kind === "sum") {
      this.totalInput = this.bias;
      for (const link of this.inputLinks) {
        if (link.isDead) continue;
        this.totalInput += link.weight * link.source.output;
      }
      this.output = this.activation.output(this.totalInput);
      return this.output;
    }
    return super.updateOutput();
  }
}

export function createGraphLink(
  source: GraphNode,
  dest: GraphNode,
  regularization: RegFn | null,
  initZero?: boolean,
  weight?: number,
): Link {
  const link = new Link(source, dest, regularization, initZero);
  if (weight !== undefined) {
    link.weight = weight;
  }
  source.outputs.push(link);
  dest.inputLinks.push(link);
  return link;
}

export function resetGraphDerivatives(nodes: Iterable<GraphNode>): void {
  for (const node of nodes) {
    node.outputDer = 0;
    node.inputDer = 0;
    node.accInputDer = 0;
    node.numAccumulatedDers = 0;
    for (const link of node.inputLinks) {
      link.errorDer = 0;
      link.accErrorDer = 0;
      link.numAccumulatedDers = 0;
    }
    for (const link of node.outputs) {
      link.errorDer = 0;
      link.accErrorDer = 0;
      link.numAccumulatedDers = 0;
    }
  }
}

export { Activations, RegularizationFunction };
