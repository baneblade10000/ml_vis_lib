import { Locale } from "@ml-vis/core/i18n";
import { useLocale } from "../../i18n";
import type { SignalPresetId } from "./config";

export type SignalTabId = "convolution" | "correlation" | "fourier" | "theorem";

export type SignalMessages = {
  signal: string;
  kernel: string;
  play: string;
  pause: string;
  step: string;
  reset: string;
  position: string;
  tabConvolution: string;
  tabCorrelation: string;
  tabFourier: string;
  tabTheorem: string;
  legendF: string;
  legendKernel: string;
  legendConvolution: string;
  legendCorrelation: string;
  legendDirect: string;
  legendFft: string;
  legendSpectrum: string;
  legendHarmonic: string;
  legendResult: string;
  convHint: string;
  corrHint: string;
  fourierHint: string;
  theoremHint: string;
  harmonic: string;
  showAsConvolution: string;
  recipe: string;
  maxDeviation: string;
  presetLabels: Record<SignalPresetId, string>;
  paramLabels: {
    width: string;
    sigma: string;
    tau: string;
    frequency: string;
  };
};

const PRESET_LABELS_EN: Record<SignalPresetId, string> = {
  delta: "Delta (impulse)",
  box: "Box",
  gaussian: "Gaussian",
  triangle: "Triangle",
  expDecay: "Exp. decay",
  sinc: "Sinc",
  cosine: "Cosine",
  sine: "Sine",
};

const PRESET_LABELS_RU: Record<SignalPresetId, string> = {
  delta: "Дельта (импульс)",
  box: "Прямоугольник",
  gaussian: "Гауссиана",
  triangle: "Треугольник",
  expDecay: "Эксп. спад",
  sinc: "Sinc",
  cosine: "Косинус",
  sine: "Синус",
};

export const signalMessages: Record<Locale, SignalMessages> = {
  en: {
    signal: "Signal f",
    kernel: "Kernel g",
    play: "Play",
    pause: "Pause",
    step: "Step",
    reset: "Reset",
    position: "Position",
    tabConvolution: "Convolution",
    tabCorrelation: "Cross-correlation",
    tabFourier: "Fourier transform",
    tabTheorem: "Convolution theorem",
    legendF: "f",
    legendKernel: "g",
    legendConvolution: "f ∗ g",
    legendCorrelation: "f ⋆ g",
    legendDirect: "direct f ∗ g",
    legendFft: "ifft(F·G)",
    legendSpectrum: "|F[k]|",
    legendHarmonic: "harmonic k",
    legendResult: "result",
    convHint: "Convolution flips the kernel: y[n] = Σ f[m]·g[n−m].",
    corrHint: "Cross-correlation does NOT flip the kernel — that is how CNNs “scan” a signal.",
    fourierHint: "Any signal is a sum of sinusoids. The spectrum |F[k]| shows each one's weight.",
    theoremHint: "Convolution in time equals pointwise multiplication in frequency.",
    harmonic: "Harmonic",
    showAsConvolution: "Overlay f ∗ g for comparison",
    recipe: "F = fft(f),  G = fft(g),  ifft(F · G)  ⟺  f ∗ g",
    maxDeviation: "max deviation",
    presetLabels: PRESET_LABELS_EN,
    paramLabels: { width: "width", sigma: "σ", tau: "τ", frequency: "freq." },
  },
  ru: {
    signal: "Сигнал f",
    kernel: "Ядро g",
    play: "Пуск",
    pause: "Пауза",
    step: "Шаг",
    reset: "Сброс",
    position: "Позиция",
    tabConvolution: "Свёртка",
    tabCorrelation: "Кросс-корреляция",
    tabFourier: "Преобразование Фурье",
    tabTheorem: "Теорема о свёртке",
    legendF: "f",
    legendKernel: "g",
    legendConvolution: "f ∗ g",
    legendCorrelation: "f ⋆ g",
    legendDirect: "прямая f ∗ g",
    legendFft: "ifft(F·G)",
    legendSpectrum: "|F[k]|",
    legendHarmonic: "гармоника k",
    legendResult: "результат",
    convHint: "Свёртка переворачивает ядро: y[n] = Σ f[m]·g[n−m].",
    corrHint: "Кросс-корреляция ядро НЕ переворачивает — так CNN «сканирует» сигнал.",
    fourierHint: "Любой сигнал — сумма синусоид. Спектр |F[k]| показывает вес каждой.",
    theoremHint: "Свёртка во времени равна перемножению в частотной области.",
    harmonic: "Гармоника",
    showAsConvolution: "Наложить f ∗ g для сравнения",
    recipe: "F = fft(f),  G = fft(g),  ifft(F · G)  ⟺  f ∗ g",
    maxDeviation: "макс. отклонение",
    presetLabels: PRESET_LABELS_RU,
    paramLabels: { width: "ширина", sigma: "σ", tau: "τ", frequency: "част." },
  },
};

export function useSignalMessages(): SignalMessages {
  const locale = useLocale();
  return signalMessages[locale];
}
