import type { Locale } from "@ml-vis/core";

export type PlaygroundMessages = {
  language: string;
  catalogTitle: string;
  catalogDescription: string;
  openVisualization: string;
  backToCatalog: string;
  vizComputationalGraphTitle: string;
  vizComputationalGraphDescription: string;
  vizNeuralNetworkTitle: string;
  vizNeuralNetworkDescription: string;
  vizConvolutionalNetworkTitle: string;
  vizConvolutionalNetworkDescription: string;
};

export const playgroundMessages: Record<Locale, PlaygroundMessages> = {
  en: {
    language: "Language",
    catalogTitle: "Visualizations",
    catalogDescription:
      "Interactive visualizations for machine learning.",
    openVisualization: "Open",
    backToCatalog: "Back to catalog",
    vizComputationalGraphTitle: "Computational Graph",
    vizComputationalGraphDescription:
      "Assembly of a computational graph from elementary operations. Forward pass evaluates node values; backward pass propagates gradients via reverse-mode automatic differentiation—the foundation of backpropagation.",
    vizNeuralNetworkTitle: "Neural Network",
    vizNeuralNetworkDescription:
      "Train a feedforward network on two-dimensional classification tasks; visualize the decision boundary, loss, and layer activations during optimization.",
    vizConvolutionalNetworkTitle: "Convolutional Network",
    vizConvolutionalNetworkDescription:
      "Train a CNN end-to-end (learnable kernels, pooling, dense head) on small image and 1-D signal datasets. Watch feature maps form layer-by-layer as loss falls and accuracy climbs.",
  },
  ru: {
    language: "Язык",
    catalogTitle: "Визуализации",
    catalogDescription: "Интерактивные визуализации для изучения методов машинного обучения.",
    openVisualization: "Открыть",
    backToCatalog: "Назад к каталогу",
    vizComputationalGraphTitle: "Вычислительный граф",
    vizComputationalGraphDescription:
      "Построение вычислительного графа из элементарных операций. Прямой проход вычисляет значения узлов, обратный — распространяет градиенты (автоматическое дифференцирование, reverse mode). Основа алгоритма обратного распространения ошибки.",
    vizNeuralNetworkTitle: "Нейронная сеть",
    vizNeuralNetworkDescription:
      "Обучение полносвязной сети на двумерных задачах классификации; визуализация границы решений, функции потерь и активаций в процессе оптимизации.",
    vizConvolutionalNetworkTitle: "Свёрточная сеть",
    vizConvolutionalNetworkDescription:
      "Полное обучение CNN (обучаемые ядра, пулинг, полносвязная голова) на маленьких изображениях и 1-D сигналах. Наблюдайте, как послойно формируются карты признаков, а loss падает и точность растёт.",
  },
};

export function usePlaygroundT(locale: Locale): (key: keyof PlaygroundMessages) => string {
  return (key) => playgroundMessages[locale][key];
}
