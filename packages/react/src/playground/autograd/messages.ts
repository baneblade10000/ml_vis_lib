import { Locale } from "@ml-vis/core/i18n";
import { useLocale } from "../../i18n";

export type AutogradMessages = {
  forward: string;
  step: string;
  stepTitle: string;
  stepTitleBackward: string;
  stepDone: string;
  stepNextForward: string;
  stepNextBackward: string;
  stepActiveForward: string;
  stepActiveBackward: string;
  backward: string;
  reset: string;
  preset: string;
  presetExpr: string;
  presetNeuron: string;
  presetSigmoid: string;
  presetHinge: string;
  presetTwoLayer: string;
  presetMlp: string;
  presetAttention: string;
  blocks: string;
  paletteHint: string;
  value: string;
  grad: string;
  inspectorEmpty: string;
  inspectorNode: string;
  inspectorEdge: string;
  inspectorOp: string;
  inspectorValue: string;
  inspectorLocalDer: string;
  inspectorFrom: string;
  inspectorTo: string;
  setAsOutput: string;
  removeNode: string;
  removeEdge: string;
  opLabels: Record<string, string>;
};

const OP_LABELS_EN: Record<string, string> = {
  input: "Input",
  const: "Const",
  add: "Add",
  mul: "Multiply",
  sub: "Subtract",
  div: "Divide",
  neg: "Negate",
  exp: "Exp",
  tanh: "Tanh",
  relu: "ReLU",
};

const OP_LABELS_RU: Record<string, string> = {
  input: "Вход",
  const: "Константа",
  add: "Сложение",
  mul: "Умножение",
  sub: "Вычитание",
  div: "Деление",
  neg: "Отрицание",
  exp: "Exp",
  tanh: "Tanh",
  relu: "ReLU",
};

export const autogradMessages: Record<Locale, AutogradMessages> = {
  en: {
    forward: "Forward",
    step: "Step",
    stepTitle: "Compute all the nodes that are ready",
    stepTitleBackward: "Push the gradient through all the nodes that are ready",
    stepDone: "Forward and backward passes complete",
    stepNextForward: "To be computed by the next step",
    stepNextBackward: "The next step pushes a gradient through this node",
    stepActiveForward: "Computed by this step",
    stepActiveBackward: "This step pushed a gradient through this node",
    backward: "Backward",
    reset: "Reset",
    preset: "Preset",
    presetExpr: "(a + b) · c",
    presetNeuron: "Neuron tanh(w·x + b)",
    presetSigmoid: "Sigmoid σ(w·x + b)",
    presetHinge: "Hinge loss max(0, 1 − y·ŷ)",
    presetTwoLayer: "Two-layer network",
    presetMlp: "MLP 2 → 3 → 1",
    presetAttention: "Attention (2 positions)",
    blocks: "Operations",
    paletteHint: "Click or drag onto canvas · Connect handles · Del to remove",
    value: "value",
    grad: "grad",
    inspectorEmpty: "Select a node or edge",
    inspectorNode: "Node",
    inspectorEdge: "Edge",
    inspectorOp: "Operation",
    inspectorValue: "Value",
    inspectorLocalDer: "Local ∂",
    inspectorFrom: "From",
    inspectorTo: "To",
    setAsOutput: "Set as output",
    removeNode: "Remove node",
    removeEdge: "Remove edge",
    opLabels: OP_LABELS_EN,
  },
  ru: {
    forward: "Прямой ход",
    step: "Шаг",
    stepTitle: "Вычислить все готовые узлы",
    stepTitleBackward: "Распространить градиент через все готовые узлы",
    stepDone: "Прямой и обратный ход завершены",
    stepNextForward: "Будет вычислен на следующем шаге",
    stepNextBackward: "Следующий шаг распространит градиент через этот узел",
    stepActiveForward: "Вычислен на этом шаге",
    stepActiveBackward: "На этом шаге градиент распространён через этот узел",
    backward: "Обратный ход",
    reset: "Сброс",
    preset: "Пример",
    presetExpr: "(a + b) · c",
    presetNeuron: "Нейрон tanh(w·x + b)",
    presetSigmoid: "Сигмоид σ(w·x + b)",
    presetHinge: "Hinge loss max(0, 1 − y·ŷ)",
    presetTwoLayer: "Двухслойная сеть",
    presetMlp: "MLP 2 → 3 → 1",
    presetAttention: "Attention (2 позиции)",
    blocks: "Операции",
    paletteHint: "Клик или перетаскивание на холст · Соединяйте порты · Del — удалить",
    value: "значение",
    grad: "градиент",
    inspectorEmpty: "Выберите узел или ребро",
    inspectorNode: "Узел",
    inspectorEdge: "Ребро",
    inspectorOp: "Операция",
    inspectorValue: "Значение",
    inspectorLocalDer: "Локальная ∂",
    inspectorFrom: "Откуда",
    inspectorTo: "Куда",
    setAsOutput: "Сделать выходом",
    removeNode: "Удалить узел",
    removeEdge: "Удалить ребро",
    opLabels: OP_LABELS_RU,
  },
};

export function useAutogradMessages(): AutogradMessages {
  const locale = useLocale();
  return autogradMessages[locale];
}
