export {
  addSignals,
  buildSignal,
  defaultSignalParams,
  getSignalPresetSpec,
  makeSignal,
  SIGNAL_PRESET_IDS,
  SIGNAL_PRESETS,
  zeros,
} from "./signal";
export type {
  Signal,
  SignalParamId,
  SignalParamSpec,
  SignalParams,
  SignalPresetId,
  SignalPresetSpec,
} from "./signal";

export { convolve, correlate } from "./convolve";

export {
  convolveViaFft,
  dftMagnitude,
  fft,
  fftRadix2,
  ifft,
  makeComplex,
  nextPow2,
  toComplex,
} from "./fft";
export type { ComplexArray, ConvolutionViaFftResult, Spectrum } from "./fft";
