import type { Locale, TfDatasetId } from "@ml-vis/core";
import { useLocale } from "../../i18n";

export type NetworkMessages = {
  reset: string;
  resetWeights: string;
  play: string;
  pause: string;
  step: string;
  epoch: string;
  testLoss: string;
  trainLoss: string;
  blocks: string;
  paletteDense: string;
  paletteSum: string;
  paletteOutput: string;
  paletteDenseHint: string;
  paletteSumHint: string;
  paletteOutputHint: string;
  paletteHint: string;
  arrangeLayout: string;
  network: string;
  hiddenLayers: string;
  layer: string;
  noHiddenLayers: string;
  addHiddenLayer: string;
  removeHiddenLayer: string;
  addNeuron: string;
  removeNeuron: string;
  dataset: string;
  datasetLabels: Record<TfDatasetId, string>;
  training: string;
  architecture: string;
  learningRate: string;
  activation: string;
  discretize: string;
  noise: string;
  trainRatio: string;
  batchSize: string;
  regenerateData: string;
  inspectorEmpty: string;
  inspectorEdge: string;
  inspectorFrom: string;
  inspectorTo: string;
  inspectorWeight: string;
  inspectorKind: string;
  inspectorOutput: string;
  inspectorBias: string;
  removeEdge: string;
  removeNode: string;
};

export const networkMessages: Record<Locale, NetworkMessages> = {
  en: {
    reset: "Reset",
    resetWeights: "Reset weights",
    play: "Play",
    pause: "Pause",
    step: "Step",
    epoch: "Epoch",
    testLoss: "Test loss",
    trainLoss: "Train loss",
    blocks: "Blocks",
    paletteDense: "Dense",
    paletteSum: "Add",
    paletteOutput: "Output",
    paletteDenseHint: "Weighted sum + activation",
    paletteSumHint: "Sum inputs (residual merge)",
    paletteOutputHint: "Network output node",
    paletteHint: "Drag onto canvas · Connect handles · Del to remove",
    arrangeLayout: "Arrange layout",
    network: "Network",
    hiddenLayers: "Hidden layers",
    layer: "Layer",
    noHiddenLayers: "No hidden layers — output connects directly to inputs.",
    addHiddenLayer: "Add hidden layer",
    removeHiddenLayer: "Remove hidden layer",
    addNeuron: "Add neuron to layer",
    removeNeuron: "Remove neuron from layer",
    dataset: "Dataset",
    datasetLabels: {
      circle: "Circle",
      xor: "XOR",
      gauss: "Gaussian",
      spiral: "Spiral",
    },
    training: "Training",
    architecture: "Architecture",
    learningRate: "Learning rate",
    activation: "Activation",
    discretize: "Discretize boundary",
    noise: "Noise",
    trainRatio: "Train ratio",
    batchSize: "Batch size",
    regenerateData: "Regenerate data",
    inspectorEmpty: "Select a node or edge to inspect",
    inspectorEdge: "Edge",
    inspectorFrom: "From",
    inspectorTo: "To",
    inspectorWeight: "Weight",
    inspectorKind: "Kind",
    inspectorOutput: "Output",
    inspectorBias: "Bias",
    removeEdge: "Remove edge",
    removeNode: "Remove node",
  },
  ru: {
    reset: "Сброс",
    resetWeights: "Сброс весов",
    play: "Пуск",
    pause: "Пауза",
    step: "Шаг",
    epoch: "Эпоха",
    testLoss: "Loss (тест)",
    trainLoss: "Loss (обучение)",
    blocks: "Блоки",
    paletteDense: "Dense",
    paletteSum: "Сумма",
    paletteOutput: "Выход",
    paletteDenseHint: "Взвешенная сумма + активация",
    paletteSumHint: "Сумма входов (residual-связь)",
    paletteOutputHint: "Выходной узел сети",
    paletteHint: "Перетащите на холст · Соединяйте порты · Del — удалить",
    arrangeLayout: "Выровнять схему",
    network: "Сеть",
    hiddenLayers: "Скрытые слои",
    layer: "Слой",
    noHiddenLayers: "Нет скрытых слоёв — выход подключён напрямую ко входам.",
    addHiddenLayer: "Добавить скрытый слой",
    removeHiddenLayer: "Удалить скрытый слой",
    addNeuron: "Добавить нейрон в слой",
    removeNeuron: "Удалить нейрон из слоя",
    dataset: "Данные",
    datasetLabels: {
      circle: "Круг",
      xor: "XOR",
      gauss: "Гауссианы",
      spiral: "Спираль",
    },
    training: "Обучение",
    architecture: "Архитектура",
    learningRate: "Скорость обучения",
    activation: "Активация",
    discretize: "Дискретная граница",
    noise: "Шум",
    trainRatio: "Доля обучения",
    batchSize: "Размер батча",
    regenerateData: "Пересоздать данные",
    inspectorEmpty: "Выберите узел или ребро",
    inspectorEdge: "Ребро",
    inspectorFrom: "Откуда",
    inspectorTo: "Куда",
    inspectorWeight: "Вес",
    inspectorKind: "Тип",
    inspectorOutput: "Выход",
    inspectorBias: "Смещение",
    removeEdge: "Удалить ребро",
    removeNode: "Удалить узел",
  },
};

/** Messages for the current locale (subscribes to the core locale store). */
export function useNetworkMessages(): NetworkMessages {
  const locale = useLocale();
  return networkMessages[locale];
}
