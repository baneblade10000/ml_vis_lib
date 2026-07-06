import type { Locale } from "@ml-vis/core";
import { useLocale } from "../i18n";

export type MonteCarloMessages = {
  reset: string;
  play: string;
  pause: string;
  step: string;
  samples: string;
  batchSize: string;
  speed: string;
  inside: string;
  outside: string;
  truePi: string;
  error: string;
  matchingDigits: string;
  convergence: string;
  estimate: string;
  explainer: string;
  formulaEmpty: string;
};

export const monteCarloMessages: Record<Locale, MonteCarloMessages> = {
  en: {
    reset: "Reset",
    play: "Play",
    pause: "Pause",
    step: "Step",
    samples: "Samples",
    batchSize: "Batch size",
    speed: "Speed",
    inside: "Inside",
    outside: "Outside",
    truePi: "True π",
    error: "Error",
    matchingDigits: "Matching digits",
    convergence: "Convergence",
    estimate: "Estimate",
    explainer: "Random points in a 1×1 square. Fraction inside the quarter circle ≈ π/4.",
    formulaEmpty: "4 × (inside ÷ total)",
  },
  ru: {
    reset: "Сброс",
    play: "Старт",
    pause: "Пауза",
    step: "Шаг",
    samples: "Точек",
    batchSize: "Размер пакета",
    speed: "Скорость",
    inside: "Внутри",
    outside: "Снаружи",
    truePi: "Истинное π",
    error: "Погрешность",
    matchingDigits: "Совпадающие цифры",
    convergence: "Сходимость",
    estimate: "Оценка",
    explainer: "Случайные точки в квадрате 1×1. Доля внутри четверти круга ≈ π/4.",
    formulaEmpty: "4 × (внутри ÷ всего)",
  },
};

export function useMonteCarloT(): (key: keyof MonteCarloMessages) => string {
  const locale = useLocale();
  return (key) => monteCarloMessages[locale][key];
}
