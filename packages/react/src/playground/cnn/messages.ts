import type { Locale } from "@ml-vis/core";
import { useLocale } from "../../i18n";

/**
 * i18n strings for the convolutional-network playground, mirroring the
 * `network/messages.ts` pattern (typed bundle per locale + `useCnnMessages`).
 */
export interface CnnMessages {
  reset: string;
  resetWeights: string;
  play: string;
  pause: string;
  step: string;
  epoch: string;
  testLoss: string;
  trainLoss: string;
  testAcc: string;
  trainAcc: string;
  mode: string;
  mode2D: string;
  mode1D: string;
  blocks: string;
  paletteConv: string;
  paletteConvHint: string;
  palettePool: string;
  palettePoolHint: string;
  paletteGap: string;
  paletteGapHint: string;
  paletteDense: string;
  paletteDenseHint: string;
  paletteHint: string;
  flatten: string;
  input: string;
  output: string;
  network: string;
  layer: string;
  filters: string;
  kernelSize: string;
  units: string;
  poolKind: string;
  poolMax: string;
  poolAvg: string;
  removeLayer: string;
  addLayer: string;
  dataset: string;
  datasetLabels2D: Record<string, string>;
  datasetLabels1D: Record<string, string>;
  training: string;
  learningRate: string;
  optimizer: string;
  activation: string;
  noise: string;
  trainRatio: string;
  batchSize: string;
  regenerateData: string;
  inspectorEmpty: string;
  inspectorKind: string;
  inspectorInputShape: string;
  inspectorOutputShape: string;
  inspectorParams: string;
  inspectorFilters: string;
  inspectorKernels: string;
  inspectorBiases: string;
  inspectorBias: string;
  gallery: string;
  galleryHint: string;
  labelCorrect: string;
  labelWrong: string;
  arrangeLayout: string;
  weightsLegend: string;
  weightsLegendAria: string;
  readoutProb: string;
  class0: string;
  class1: string;
  lossTestTrain: string;
  denseWeightsEmpty: string;
  learningCurve: string;
  regularization: string;
  regularizationLabels: Record<"none" | "L1" | "L2", string>;
  regularizationRate: string;
}

export const cnnMessages: Record<Locale, CnnMessages> = {
  en: {
    reset: "Reset",
    resetWeights: "Reset weights",
    play: "Play",
    pause: "Pause",
    step: "Step",
    epoch: "Epoch",
    testLoss: "Test loss",
    trainLoss: "Train loss",
    testAcc: "Test acc",
    trainAcc: "Train acc",
    mode: "Data",
    mode2D: "2D",
    mode1D: "1D",
    blocks: "Blocks",
    paletteConv: "Conv",
    paletteConvHint: "Convolution (learnable kernel)",
    palettePool: "Pool",
    palettePoolHint: "2×2 downsampling (max/avg)",
    paletteGap: "GAP",
    paletteGapHint: "Global average pooling (per channel)",
    paletteDense: "Dense",
    paletteDenseHint: "Fully-connected layer",
    paletteHint: "Click or drag onto canvas to add a layer",
    flatten: "Flatten",
    input: "Input",
    output: "Output",
    network: "Network",
    layer: "Layer",
    filters: "Filters",
    kernelSize: "Kernel",
    units: "Units",
    poolKind: "Pooling",
    poolMax: "Max",
    poolAvg: "Avg",
    removeLayer: "Remove layer",
    addLayer: "Add layer",
    dataset: "Dataset",
    datasetLabels2D: {
      digits: "Digits 0 / 1",
      "circles-squares": "Circles / Squares",
      "two-three-loops": "2 loops / 3 loops",
      "three-four-loops": "3 loops / 4 loops",
    },
    datasetLabels1D: {
      heartbeat: "Heartbeat",
      tones: "High / Low tone",
      pulses: "One / Two pulses",
    },
    training: "Training",
    learningRate: "Learning rate",
    optimizer: "Optimizer",
    activation: "Activation",
    noise: "Noise",
    trainRatio: "Train ratio",
    batchSize: "Batch size",
    regenerateData: "Regenerate data",
    inspectorEmpty: "Select a layer to inspect",
    inspectorKind: "Kind",
    inspectorInputShape: "Input",
    inspectorOutputShape: "Output",
    inspectorParams: "Params",
    inspectorFilters: "Filters",
    inspectorKernels: "Kernels",
    inspectorBiases: "Biases",
    inspectorBias: "Bias",
    gallery: "Inputs",
    galleryHint: "Click an example to inspect it",
    labelCorrect: "correct",
    labelWrong: "wrong",
    arrangeLayout: "Arrange layout",
    weightsLegend: "Weights",
    weightsLegendAria: "Weight color scale from −1 (deep blue) to +1 (sky cyan)",
    readoutProb: "Predicted class probabilities",
    class0: "Class 0",
    class1: "Class 1",
    lossTestTrain: "test / train",
    denseWeightsEmpty: "Weights initialize on first forward",
    learningCurve: "Learning curve",
    regularization: "Regularization",
    regularizationLabels: {
      none: "None",
      L1: "L1",
      L2: "L2",
    },
    regularizationRate: "Regularization rate",
  },
  ru: {
    reset: "Сброс",
    resetWeights: "Сбросить веса",
    play: "Пуск",
    pause: "Пауза",
    step: "Шаг",
    epoch: "Эпоха",
    testLoss: "Loss (тест)",
    trainLoss: "Loss (обучение)",
    testAcc: "Точность (тест)",
    trainAcc: "Точность (обучение)",
    mode: "Данные",
    mode2D: "2D",
    mode1D: "1D",
    blocks: "Блоки",
    paletteConv: "Свёртка",
    paletteConvHint: "Свёртка (обучаемое ядро)",
    palettePool: "Пулинг",
    palettePoolHint: "Понижение 2×2 (max/avg)",
    paletteGap: "GAP",
    paletteGapHint: "Global average pooling (по каналам)",
    paletteDense: "Полносвязный",
    paletteDenseHint: "Полносвязный слой",
    paletteHint: "Клик или перетащите на холст, чтобы добавить слой",
    flatten: "Плоский",
    input: "Вход",
    output: "Выход",
    network: "Сеть",
    layer: "Слой",
    filters: "Фильтры",
    kernelSize: "Ядро",
    units: "Нейроны",
    poolKind: "Пулинг",
    poolMax: "Max",
    poolAvg: "Avg",
    removeLayer: "Удалить слой",
    addLayer: "Добавить слой",
    dataset: "Данные",
    datasetLabels2D: {
      digits: "Цифры 0 / 1",
      "circles-squares": "Круги / Квадраты",
      "two-three-loops": "2 петли / 3 петли",
      "three-four-loops": "3 петли / 4 петли",
    },
    datasetLabels1D: {
      heartbeat: "Пульс",
      tones: "Высокий / Низкий тон",
      pulses: "Один / Два импульса",
    },
    training: "Обучение",
    learningRate: "Коэффициент обучения",
    optimizer: "Оптимизатор",
    activation: "Активация",
    noise: "Шум",
    trainRatio: "Доля обучения",
    batchSize: "Размер батча",
    regenerateData: "Пересоздать данные",
    inspectorEmpty: "Выберите слой для просмотра",
    inspectorKind: "Тип",
    inspectorInputShape: "Вход",
    inspectorOutputShape: "Выход",
    inspectorParams: "Параметры",
    inspectorFilters: "Фильтры",
    inspectorKernels: "Ядра",
    inspectorBiases: "Смещения",
    inspectorBias: "Bias",
    gallery: "Входы",
    galleryHint: "Кликните по примеру для просмотра",
    labelCorrect: "верно",
    labelWrong: "ошибка",
    arrangeLayout: "Выровнять схему",
    weightsLegend: "Веса",
    weightsLegendAria: "Цвет весов от −1 (тёмно-синий) до +1 (голубой)",
    readoutProb: "Вероятности классов",
    class0: "Класс 0",
    class1: "Класс 1",
    lossTestTrain: "тест / обуч.",
    denseWeightsEmpty: "Веса появятся после первого прохода",
    learningCurve: "Кривая обучения",
    regularization: "Регуляризация",
    regularizationLabels: {
      none: "Нет",
      L1: "L1",
      L2: "L2",
    },
    regularizationRate: "Коэффициент регуляризации",
  },
};

export function useCnnMessages(): CnnMessages {
  const locale = useLocale();
  return cnnMessages[locale];
}
