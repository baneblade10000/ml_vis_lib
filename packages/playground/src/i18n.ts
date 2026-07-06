import type { Locale } from "@ml-vis/core";

export type PlaygroundMessages = {
  language: string;
  catalogTitle: string;
  catalogDescription: string;
  openVisualization: string;
  backToCatalog: string;
  vizNeuralNetworkTitle: string;
  vizNeuralNetworkDescription: string;
  vizMonteCarloPiTitle: string;
  vizMonteCarloPiDescription: string;
};

export const playgroundMessages: Record<Locale, PlaygroundMessages> = {
  en: {
    language: "Language",
    catalogTitle: "Visualizations",
    catalogDescription: "Interactive demos for exploring machine learning concepts.",
    openVisualization: "Open",
    backToCatalog: "Back to catalog",
    vizNeuralNetworkTitle: "Neural Network Playground",
    vizNeuralNetworkDescription:
      "Train a small neural network on 2D datasets and watch the decision boundary, loss, and activations update live.",
    vizMonteCarloPiTitle: "Monte Carlo π",
    vizMonteCarloPiDescription:
      "Estimate π by throwing random points into a square — watch the quarter circle fill up and the estimate converge.",
  },
  ru: {
    language: "Язык",
    catalogTitle: "Визуализации",
    catalogDescription: "Интерактивные демо для изучения концепций машинного обучения.",
    openVisualization: "Открыть",
    backToCatalog: "Назад к каталогу",
    vizNeuralNetworkTitle: "Neural Network Playground",
    vizNeuralNetworkDescription:
      "Обучайте небольшую нейросеть на 2D-данных и наблюдайте за границей решений, loss и активациями в реальном времени.",
    vizMonteCarloPiTitle: "Monte Carlo π",
    vizMonteCarloPiDescription:
      "Оценивайте π случайными точками в квадрате — наблюдайте, как заполняется четверть круга и сходится оценка.",
  },
};

export function usePlaygroundT(locale: Locale): (key: keyof PlaygroundMessages) => string {
  return (key) => playgroundMessages[locale][key];
}
