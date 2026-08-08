export { ChartBox, useCanvasChart } from "./useCanvasChart";

export {
  DecisionBoundaryChart,
  DecisionBoundaryPlayground,
  PlaygroundControls,
} from "./DecisionBoundaryPlayground";
export type { DecisionBoundaryChartProps, PlaygroundControlsProps } from "./DecisionBoundaryPlayground";

export { ReactFlowNetworkGraph } from "./playground/network/ReactFlowNetworkGraph";
export type { ReactFlowNetworkGraphProps } from "./playground/network/ReactFlowNetworkGraph";
export { NetworkInspector } from "./playground/network/NetworkInspector";
export { NeuralNetworkPlayground } from "./playground/NeuralNetworkPlayground";
export type { NeuralNetworkPlaygroundProps } from "./playground/NeuralNetworkPlayground";
export { ComputationalGraphPlayground } from "./playground/autograd/ComputationalGraphPlayground";
export type { ComputationalGraphPlaygroundProps } from "./playground/autograd/ComputationalGraphPlayground";
export { ConvolutionalNetworkPlayground } from "./playground/cnn/ConvolutionalNetworkPlayground";
export type { ConvolutionalNetworkPlaygroundProps } from "./playground/cnn/ConvolutionalNetworkPlayground";
export { SignalPlayground } from "./playground/signal/SignalPlayground";
export type { SignalPlaygroundProps } from "./playground/signal/SignalPlayground";

export {
  I18nProvider,
  localeLabels,
  locales,
  reactMessages,
  setLocale,
  useI18n,
  useLocale,
} from "./i18n";
export type { I18nProviderProps, ReactMessages } from "./i18n";
export type { Locale } from "@ml-vis/core";

export {
  Section,
  SectionLayout,
  SectionNav,
  SectionProvider,
  useSections,
} from "./section";
export type {
  SectionLayoutProps,
  SectionNavProps,
  SectionProps,
  SectionProviderProps,
} from "./section";

export type { SectionDefinition, SectionSize } from "@ml-vis/core";
