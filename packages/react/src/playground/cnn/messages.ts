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
  paletteDense: string;
  paletteDenseHint: string;
  paletteHint: string;
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
  gallery: string;
  galleryHint: string;
  labelCorrect: string;
  labelWrong: string;
  arrangeLayout: string;
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
    mode: "Mode",
    mode2D: "2D — images",
    mode1D: "1D — signals",
    blocks: "Blocks",
    paletteConv: "Conv",
    paletteConvHint: "Convolution (learnable kernel)",
    palettePool: "Pool",
    palettePoolHint: "2×2 downsampling (max/avg)",
    paletteDense: "Dense",
    paletteDenseHint: "Fully-connected layer",
    paletteHint: "Drag onto canvas · select a layer to edit it",
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
      "disc-ring": "Disc / Ring",
      "cross-square": "Cross / Square",
      "blob-gradient": "Blob / Gradient",
    },
    datasetLabels1D: {
      "sine-pulse": "Sine pulse",
      step: "Step / Square",
      ramp: "Ramp / Plateau",
    },
    training: "Training",
    learningRate: "Learning rate",
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
    gallery: "Inputs",
    galleryHint: "Click an example to inspect it",
    labelCorrect: "correct",
    labelWrong: "wrong",
    arrangeLayout: "Arrange layout",
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
    mode: "Режим",
    mode2D: "2D — изображения",
    mode1D: "1D — сигналы",
    blocks: "Блоки",
    paletteConv: "Свёртка",
    paletteConvHint: "Свёртка (обучаемое ядро)",
    palettePool: "Пулинг",
    palettePoolHint: "Понижение 2×2 (max/avg)",
    paletteDense: "Полносвязный",
    paletteDenseHint: "Полносвязный слой",
    paletteHint: "Перетащите на холст · выберите слой для редактирования",
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
      "disc-ring": "Диск / Кольцо",
      "cross-square": "Крест / Квадрат",
      "blob-gradient": "Блоб / Градиент",
    },
    datasetLabels1D: {
      "sine-pulse": "Синус-импульс",
      step: "Ступень / Меандр",
      ramp: "Наклон / Плато",
    },
    training: "Обучение",
    learningRate: "Скорость обучения",
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
    gallery: "Входы",
    galleryHint: "Кликните по примеру для просмотра",
    labelCorrect: "верно",
    labelWrong: "ошибка",
    arrangeLayout: "Выровнять схему",
  },
};

export function useCnnMessages(): CnnMessages {
  const locale = useLocale();
  return cnnMessages[locale];
}
