export type AutogradOp =
  | "input"
  | "const"
  | "add"
  | "mul"
  | "sub"
  | "div"
  | "neg"
  | "exp"
  | "tanh"
  | "relu";

export interface AutogradPosition {
  x: number;
  y: number;
}

export interface AutogradNodeDef {
  id: string;
  op: AutogradOp;
  label?: string;
  /** Stored scalar for leaf nodes (input / const). Ignored for operations. */
  value: number;
}

export interface AutogradEdgeDef {
  id: string;
  source: string;
  target: string;
}

export interface AutogradSnapshot {
  nodes: AutogradNodeDef[];
  edges: AutogradEdgeDef[];
  outputId: string;
  positions: Record<string, AutogradPosition>;
}

export interface OpSpec {
  symbol: string;
  label: string;
  /** Number of inputs the op accepts. `Infinity` = variadic (n-ary). */
  arity: number;
  /** Whether input order matters (sub, div). Commutative ops (add, mul) do not care. */
  ordered: boolean;
  /** Leaf nodes carry an editable scalar and take no inputs. */
  leaf: boolean;
}

export const OP_SPECS: Record<AutogradOp, OpSpec> = {
  input: { symbol: "x", label: "Input", arity: 0, ordered: false, leaf: true },
  const: { symbol: "c", label: "Const", arity: 0, ordered: false, leaf: true },
  add: { symbol: "+", label: "Add", arity: Infinity, ordered: false, leaf: false },
  mul: { symbol: "×", label: "Multiply", arity: Infinity, ordered: false, leaf: false },
  sub: { symbol: "−", label: "Subtract", arity: 2, ordered: true, leaf: false },
  div: { symbol: "÷", label: "Divide", arity: 2, ordered: true, leaf: false },
  neg: { symbol: "±", label: "Negate", arity: 1, ordered: false, leaf: false },
  exp: { symbol: "eˣ", label: "Exp", arity: 1, ordered: false, leaf: false },
  tanh: { symbol: "tanh", label: "Tanh", arity: 1, ordered: false, leaf: false },
  relu: { symbol: "relu", label: "ReLU", arity: 1, ordered: false, leaf: false },
};

/** Ops offered in the build palette (excludes the output node, which is implicit). */
export const AUTOGRAD_PALETTE_OPS: AutogradOp[] = [
  "input",
  "const",
  "add",
  "mul",
  "sub",
  "div",
  "neg",
  "exp",
  "tanh",
  "relu",
];
