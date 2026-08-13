import { Locale } from "@ml-vis/core/i18n";
import { Dataset1DId, DatasetId as NetworkDatasetId, NetworkProblemType, NetworkRegularizationId, WeightInitId } from "@ml-vis/core/network";
import { useLocale } from "../../i18n";

export type NetworkMessages = {
  reset: string;
  resetWeights: string;
  play: string;
  pause: string;
  starting: string;
  startingHint: string;
  step: string;
  epoch: string;
  testLoss: string;
  trainLoss: string;
  blocks: string;
  paletteDense: string;
  paletteDenseHint: string;
  paletteHint: string;
  arrangeLayout: string;
  layoutViz: string;
  layoutVizGraph: string;
  layoutVizMatrix: string;
  heatmapViz: string;
  edgeViz: string;
  edgeVizWeight: string;
  edgeVizGradient: string;
  network: string;
  hiddenLayers: string;
  layer: string;
  noHiddenLayers: string;
  addHiddenLayer: string;
  removeHiddenLayer: string;
  addNeuron: string;
  removeNeuron: string;
  mode: string;
  mode2D: string;
  mode1D: string;
  problemType: string;
  problemTypeLabels: Record<NetworkProblemType, string>;
  dataset: string;
  datasetLabels: Record<NetworkDatasetId, string>;
  dataset1dLabels: Record<Dataset1DId, string>;
  training: string;
  learningCurve: string;
  learningRate: string;
  optimizer: string;
  activation: string;
  weightInit: string;
  weightInitLabels: Record<WeightInitId, string>;
  regularization: string;
  regularizationLabels: Record<NetworkRegularizationId, string>;
  regularizationRate: string;
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
  inspectorGradient: string;
  inspectorKind: string;
  inspectorOutput: string;
  inspectorBias: string;
  removeEdge: string;
  removeNode: string;
  inspiredBy: string;
  inspiredBySource: string;
};

export const networkMessages: Record<Locale, NetworkMessages> = {
  en: {
    reset: "Reset",
    resetWeights: "Reset weights",
    play: "Play",
    pause: "Pause",
    starting: "Starting…",
    startingHint: "Preparing model…",
    step: "Step",
    epoch: "Epoch",
    testLoss: "Test loss",
    trainLoss: "Train loss",
    blocks: "Blocks",
    paletteDense: "Dense",
    paletteDenseHint: "Weighted sum + activation",
    paletteHint: "Drag onto canvas · Connect handles · Del to remove",
    arrangeLayout: "Arrange layout",
    layoutViz: "Weight display:",
    layoutVizGraph: "Edges",
    layoutVizMatrix: "Matrices",
    heatmapViz: "Resolution:",
    edgeViz: "Connections",
    edgeVizWeight: "Weights",
    edgeVizGradient: "Gradients",
    network: "Network",
    hiddenLayers: "Hidden layers",
    layer: "Layer",
    noHiddenLayers: "No hidden layers — output connects directly to inputs.",
    addHiddenLayer: "Add hidden layer",
    removeHiddenLayer: "Remove hidden layer",
    addNeuron: "Add neuron to layer",
    removeNeuron: "Remove neuron from layer",
    mode: "Data",
    mode2D: "2D",
    mode1D: "1D",
    problemType: "Problem",
    problemTypeLabels: {
      classification: "Classification",
      regression: "Regression",
    },
    dataset: "Dataset",
    datasetLabels: {
      circle: "Circle",
      xor: "XOR",
      gauss: "Gaussian",
      spiral: "Spiral",
      sinSin: "sin(x)·sin(y)",
    },
    dataset1dLabels: {
      gauss1d: "Two Gaussians",
      threshold: "Threshold",
      twoClusters: "Four clusters",
      sine: "Sine",
      linear: "Linear",
      cubic: "Cubic",
      step: "Step",
    },
    training: "Training",
    learningCurve: "Learning curve",
    learningRate: "Learning rate",
    optimizer: "Optimizer",
    activation: "Activation",
    weightInit: "Weight init",
    weightInitLabels: {
      uniform: "Uniform [-0.5, 0.5]",
      xavier: "Xavier (Glorot)",
      he: "He",
      normal: "Normal (σ=0.1)",
      zeros: "Zeros",
    },
    regularization: "Regularization",
    regularizationLabels: {
      none: "None",
      L1: "L1",
      L2: "L2",
    },
    regularizationRate: "Regularization rate",
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
    inspectorGradient: "∂E/∂w",
    inspectorKind: "Kind",
    inspectorOutput: "Output",
    inspectorBias: "Bias",
    removeEdge: "Remove edge",
    removeNode: "Remove node",
    inspiredBy: "Inspired by",
    inspiredBySource: "TensorFlow Playground",
  },
  ru: {
    reset: "Сброс",
    resetWeights: "Сброс весов",
    play: "Пуск",
    pause: "Пауза",
    starting: "Запуск…",
    startingHint: "Подготовка модели…",
    step: "Шаг",
    epoch: "Эпоха",
    testLoss: "Loss (тест)",
    trainLoss: "Loss (обучение)",
    blocks: "Блоки",
    paletteDense: "Dense",
    paletteDenseHint: "Взвешенная сумма + активация",
    paletteHint: "Перетащите на холст · Соединяйте порты · Del — удалить",
    arrangeLayout: "Выровнять схему",
    layoutViz: "Отображение весов:",
    layoutVizGraph: "Рёбра",
    layoutVizMatrix: "Матрицы",
    heatmapViz: "Разрешение:",
    edgeViz: "Связи",
    edgeVizWeight: "Веса",
    edgeVizGradient: "Градиенты",
    network: "Сеть",
    hiddenLayers: "Скрытые слои",
    layer: "Слой",
    noHiddenLayers: "Нет скрытых слоёв — выход подключён напрямую ко входам.",
    addHiddenLayer: "Добавить скрытый слой",
    removeHiddenLayer: "Удалить скрытый слой",
    addNeuron: "Добавить нейрон в слой",
    removeNeuron: "Удалить нейрон из слоя",
    mode: "Данные",
    mode2D: "2D",
    mode1D: "1D",
    problemType: "Задача",
    problemTypeLabels: {
      classification: "Классификация",
      regression: "Регрессия",
    },
    dataset: "Датасет",
    datasetLabels: {
      circle: "Круг",
      xor: "XOR",
      gauss: "Гауссианы",
      spiral: "Спираль",
      sinSin: "sin(x)·sin(y)",
    },
    dataset1dLabels: {
      gauss1d: "Две гауссианы",
      threshold: "Порог",
      twoClusters: "Четыре кластера",
      sine: "Синус",
      linear: "Линейная",
      cubic: "Кубическая",
      step: "Ступенька",
    },
    training: "Обучение",
    learningCurve: "Кривая обучения",
    learningRate: "Коэффициент обучения",
    optimizer: "Оптимизатор",
    activation: "Активация",
    weightInit: "Инициализация весов",
    weightInitLabels: {
      uniform: "Равномерная [-0.5, 0.5]",
      xavier: "Xavier (Glorot)",
      he: "He",
      normal: "Нормальная (σ=0.1)",
      zeros: "Нули",
    },
    regularization: "Регуляризация",
    regularizationLabels: {
      none: "Нет",
      L1: "L1",
      L2: "L2",
    },
    regularizationRate: "Коэффициент регуляризации",
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
    inspectorGradient: "∂E/∂w",
    inspectorKind: "Тип",
    inspectorOutput: "Выход",
    inspectorBias: "Смещение",
    removeEdge: "Удалить ребро",
    removeNode: "Удалить узел",
    inspiredBy: "Вдохновлено",
    inspiredBySource: "TensorFlow Playground",
  },
};

/** Messages for the current locale (subscribes to the core locale store). */
export function useNetworkMessages(): NetworkMessages {
  const locale = useLocale();
  return networkMessages[locale];
}
