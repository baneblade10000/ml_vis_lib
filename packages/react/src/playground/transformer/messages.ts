/**
 * Typed message bundle for the transformer playground.
 * `network/messages.ts` pattern (typed bundle per locale + `useTransformerMessages`).
 */

import { Locale } from "@ml-vis/core/i18n";
import { useLocale } from "../../i18n";

export interface TransformerMessages {
  reset: string;
  play: string;
  pause: string;
  step: string;
  stepCount: string;
  loss: string;
  accuracy: string;
  learningRate: string;
  task: string;
  taskTranslate: string;
  taskTranslateHint: string;
  taskReverse: string;
  taskReverseHint: string;
  encoder: string;
  decoder: string;
  selfAttention: string;
  maskedSelfAttention: string;
  crossAttention: string;
  feedForward: string;
  layer: string;
  head: string;
  headMean: string;
  inputSequence: string;
  targetSequence: string;
  prediction: string;
  memory: string;
  logits: string;
  encSelfTitle: string;
  encSelfHint: string;
  decSelfTitle: string;
  decSelfHint: string;
  crossTitle: string;
  crossHint: string;
  lossCurve: string;
  queryAxis: string;
  keyAxis: string;
  loadingWasm: string;
  wasmError: string;
  inspiredBy: string;
  inspiredBySource: string;
}

const en: TransformerMessages = {
  reset: "Reset",
  play: "Play",
  pause: "Pause",
  step: "Step",
  stepCount: "Step",
  loss: "Loss (EMA)",
  accuracy: "Accuracy",
  learningRate: "Learning rate",
  task: "Task",
  taskTranslate: "Translate RU→EN",
  taskTranslateHint:
    "Translate a Russian phrase word-by-word; cross-attention learns the word alignment (adjective and noun swap places)",
  taskReverse: "Reverse",
  taskReverseHint: "The decoder must emit the input words in reverse order",
  encoder: "Encoder",
  decoder: "Decoder",
  selfAttention: "Self-attention",
  maskedSelfAttention: "Masked self-attention",
  crossAttention: "Cross-attention",
  feedForward: "Feed-forward",
  layer: "Layer",
  head: "Head",
  headMean: "mean",
  inputSequence: "Input",
  targetSequence: "Target",
  prediction: "Prediction",
  memory: "Memory",
  logits: "Logits",
  encSelfTitle: "Encoder self-attention",
  encSelfHint: "Rows = queries, columns = keys; every word looks at the whole Russian phrase",
  decSelfTitle: "Decoder self-attention",
  decSelfHint: "Causal mask: a token can only attend to itself and earlier outputs",
  crossTitle: "Cross-attention",
  crossHint: "English queries attend to Russian keys — the learned word alignment between the languages",
  lossCurve: "Loss",
  queryAxis: "queries",
  keyAxis: "keys",
  loadingWasm: "Loading Rust engine…",
  wasmError: "WASM engine error",
  inspiredBy: "Inspired by",
  inspiredBySource: "Attention Is All You Need",
};

const ru: TransformerMessages = {
  reset: "Сброс",
  play: "Запуск",
  pause: "Пауза",
  step: "Шаг",
  stepCount: "Шаг",
  loss: "Loss (EMA)",
  accuracy: "Точность",
  learningRate: "Скорость обучения",
  task: "Задача",
  taskTranslate: "Перевод RU→EN",
  taskTranslateHint:
    "Пословный перевод русской фразы; cross-attention учит выравнивание слов (прилагательное и существительное меняются местами)",
  taskReverse: "Реверс",
  taskReverseHint: "Декодер должен выдать входные слова в обратном порядке",
  encoder: "Энкодер",
  decoder: "Декодер",
  selfAttention: "Self-attention",
  maskedSelfAttention: "Masked self-attention",
  crossAttention: "Cross-attention",
  feedForward: "Feed-forward",
  layer: "Слой",
  head: "Голова",
  headMean: "средн.",
  inputSequence: "Вход",
  targetSequence: "Цель",
  prediction: "Предсказание",
  memory: "Память",
  logits: "Логиты",
  encSelfTitle: "Self-attention энкодера",
  encSelfHint: "Строки — запросы, столбцы — ключи; каждое слово смотрит на всю русскую фразу",
  decSelfTitle: "Self-attention декодера",
  decSelfHint: "Причинная маска: токен видит только себя и предыдущие выходы",
  crossTitle: "Cross-attention",
  crossHint: "Английские запросы смотрят на русские ключи — выученное выравнивание слов между языками",
  lossCurve: "Loss",
  queryAxis: "запросы",
  keyAxis: "ключи",
  loadingWasm: "Загрузка Rust-движка…",
  wasmError: "Ошибка WASM-движка",
  inspiredBy: "По мотивам",
  inspiredBySource: "Attention Is All You Need",
};

export const transformerMessages: Record<Locale, TransformerMessages> = { en, ru };

export function useTransformerMessages(): TransformerMessages {
  const locale = useLocale();
  return transformerMessages[locale];
}

export type { Locale };
