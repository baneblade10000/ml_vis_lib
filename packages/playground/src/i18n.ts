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
  vizSignalLabTitle: string;
  vizSignalLabDescription: string;
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
      "Forward and reverse-mode autodiff on a graph of elementary ops.",
    vizNeuralNetworkTitle: "Neural Network",
    vizNeuralNetworkDescription:
      "MLP training: decision boundary, loss, activations.",
    vizConvolutionalNetworkTitle: "Convolutional Network",
    vizConvolutionalNetworkDescription:
      "CNN training on images and 1-D signals: kernels, feature maps, metrics.",
    vizSignalLabTitle: "Signals & Transforms",
    vizSignalLabDescription:
      "Convolution, cross-correlation, Fourier transform, convolution theorem.",
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
      "Прямой и обратный проход (reverse-mode autodiff) на графе операций.",
    vizNeuralNetworkTitle: "Нейронная сеть",
    vizNeuralNetworkDescription:
      "Обучение MLP: граница решений, loss, активации.",
    vizConvolutionalNetworkTitle: "Свёрточная сеть",
    vizConvolutionalNetworkDescription:
      "Обучение CNN на изображениях и 1-D сигналах: ядра, карты признаков, метрики.",
    vizSignalLabTitle: "Сигналы и преобразования",
    vizSignalLabDescription:
      "Свёртка, кросс-корреляция, Фурье, теорема о свёртке.",
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
