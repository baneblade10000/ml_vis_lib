import { Locale } from "@ml-vis/core/i18n";
import { useLocale } from "../../i18n";

export type AutogradMessages = {
  forward: string;
  backward: string;
  reset: string;
  preset: string;
  presetExpr: string;
  presetNeuron: string;
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
    backward: "Backward",
    reset: "Reset",
    preset: "Preset",
    presetExpr: "(a + b) · c",
    presetNeuron: "Neuron tanh(w·x + b)",
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
    backward: "Обратный ход",
    reset: "Сброс",
    preset: "Пример",
    presetExpr: "(a + b) · c",
    presetNeuron: "Нейрон tanh(w·x + b)",
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
