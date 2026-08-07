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
  mobileStubService: string;
  mobileStubTitle: string;
  mobileStubBody: string;
  mobileStubNoticeLabel: string;
  mobileStubNotice: string;
  mobileStubFooter: string;
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
    mobileStubService: "Interactive visualizations service",
    mobileStubTitle: "Access from mobile devices is not supported",
    mobileStubBody:
      "This service is intended for use on a personal computer or laptop. Interactive visualizations require a screen of sufficient size and a pointing device.",
    mobileStubNoticeLabel: "How to proceed",
    mobileStubNotice:
      "Open this page in a desktop browser (current version of Chrome, Firefox, Edge, or Safari).",
    mobileStubFooter:
      "If you believe this message was shown in error, widen the browser window or connect an external display.",
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
    mobileStubService: "Сервис интерактивных визуализаций",
    mobileStubTitle: "Доступ с мобильных устройств ограничен",
    mobileStubBody:
      "Сервис предназначен для работы на персональном компьютере или ноутбуке. Интерактивные визуализации требуют экрана достаточного размера и устройства ввода типа «мышь».",
    mobileStubNoticeLabel: "Что необходимо сделать",
    mobileStubNotice:
      "Откройте эту страницу в браузере на персональном компьютере (актуальная версия Chrome, Firefox, Edge или Safari).",
    mobileStubFooter:
      "Если данное сообщение отображается ошибочно, увеличьте ширину окна браузера или подключите внешний монитор.",
  },
};

export function usePlaygroundT(locale: Locale): (key: keyof PlaygroundMessages) => string {
  return (key) => playgroundMessages[locale][key];
}
