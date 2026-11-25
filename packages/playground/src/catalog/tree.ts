import type { VisualizationId } from "../visualizations";

export interface Bilingual {
  ru: string;
  en: string;
}

/**
 * Node of the ML knowledge tree. Sections (depth 1) carry an accent color
 * that their subtree inherits; leaves may attach visualization cards.
 * Nodes without visualizations are planned branches shown as ghosts.
 */
export interface CatalogTreeNode {
  id: string;
  label: Bilingual;
  accent?: string;
  children?: CatalogTreeNode[];
  vizIds?: VisualizationId[];
}

export const SECTION_ACCENTS = {
  basics: "#3457a3",
  classical: "#b45309",
  neuralNetworks: "#6d28d9",
  computerVision: "#dc263a",
  nlp: "#0f766e",
  generative: "#be185d",
  reinforcement: "#0c4a6e",
  signals: "#64748b",
} as const;

export const catalogKnowledgeTree: CatalogTreeNode = {
  id: "ml",
  label: { ru: "Машинное обучение", en: "Machine Learning" },
  children: [
    {
      id: "basics",
      label: { ru: "Основы", en: "Foundations" },
      accent: SECTION_ACCENTS.basics,
      children: [
        {
          id: "basics-linalg",
          label: { ru: "Линейная алгебра и матрицы", en: "Linear algebra & matrices" },
        },
        {
          id: "basics-optim",
          label: { ru: "Градиентный спуск и оптимизация", en: "Gradient descent & optimization" },
        },
        {
          id: "basics-autodiff",
          label: { ru: "Вычислительные графы и автодифференцирование", en: "Computational graphs & autodiff" },
          vizIds: ["computational-graph"],
        },
        {
          id: "basics-overfit",
          label: { ru: "Переобучение и bias–variance", en: "Overfitting & bias–variance" },
        },
        {
          id: "basics-metrics",
          label: { ru: "Метрики и валидация", en: "Metrics & validation" },
        },
      ],
    },
    {
      id: "classical",
      label: { ru: "Классическое ML", en: "Classical ML" },
      accent: SECTION_ACCENTS.classical,
      children: [
        {
          id: "classical-linear",
          label: { ru: "Линейная и логистическая регрессия", en: "Linear & logistic regression" },
        },
        {
          id: "classical-trees",
          label: { ru: "Деревья решений и ансамбли", en: "Decision trees & ensembles" },
          vizIds: ["ensemble-playground"],
        },
        {
          id: "classical-svm",
          label: { ru: "SVM и ядерные методы", en: "SVM & kernel methods" },
        },
        {
          id: "classical-clustering",
          label: { ru: "Кластеризация", en: "Clustering" },
        },
        {
          id: "classical-pca",
          label: { ru: "Снижение размерности (PCA)", en: "Dimensionality reduction (PCA)" },
        },
      ],
    },
    {
      id: "nn",
      label: { ru: "Нейронные сети", en: "Neural networks" },
      accent: SECTION_ACCENTS.neuralNetworks,
      children: [
        {
          id: "nn-mlp",
          label: { ru: "Перцептрон и MLP", en: "Perceptron & MLP" },
          vizIds: ["neural-network"],
        },
        {
          id: "nn-activations",
          label: { ru: "Функции активации", en: "Activation functions" },
        },
        {
          id: "nn-backprop",
          label: { ru: "Backpropagation", en: "Backpropagation" },
        },
        {
          id: "nn-regularization",
          label: { ru: "Регуляризация (dropout, L2)", en: "Regularization (dropout, L2)" },
        },
        {
          id: "nn-normalization",
          label: { ru: "Нормализация слоёв", en: "Layer normalization" },
        },
      ],
    },
    {
      id: "cv",
      label: { ru: "Компьютерное зрение", en: "Computer vision" },
      accent: SECTION_ACCENTS.computerVision,
      children: [
        {
          id: "cv-conv",
          label: { ru: "Свёртка, ядра и карты признаков", en: "Convolution, kernels & feature maps" },
          vizIds: ["convolutional-network"],
        },
        {
          id: "cv-architectures",
          label: { ru: "Архитектуры CNN (ResNet и др.)", en: "CNN architectures (ResNet etc.)" },
        },
        {
          id: "cv-detection",
          label: { ru: "Детектирование и сегментация", en: "Detection & segmentation" },
        },
      ],
    },
    {
      id: "nlp",
      label: { ru: "Последовательности и NLP", en: "Sequences & NLP" },
      accent: SECTION_ACCENTS.nlp,
      children: [
        {
          id: "nlp-embeddings",
          label: { ru: "Эмбеддинги слов", en: "Word embeddings" },
        },
        {
          id: "nlp-rnn",
          label: { ru: "Рекуррентные сети (RNN, LSTM)", en: "Recurrent networks (RNN, LSTM)" },
        },
        {
          id: "nlp-attention",
          label: { ru: "Механизм внимания", en: "Attention mechanism" },
        },
        {
          id: "nlp-transformers",
          label: { ru: "Трансформеры", en: "Transformers" },
          vizIds: ["transformer", "transformer-flow"],
        },
        {
          id: "nlp-llm",
          label: { ru: "Языковые модели (LLM)", en: "Language models (LLM)" },
        },
      ],
    },
    {
      id: "generative",
      label: { ru: "Генеративные модели", en: "Generative models" },
      accent: SECTION_ACCENTS.generative,
      children: [
        {
          id: "generative-ae",
          label: { ru: "Автоэнкодеры и VAE", en: "Autoencoders & VAE" },
        },
        {
          id: "generative-gan",
          label: { ru: "Генеративно-состязательные сети (GAN)", en: "Generative adversarial networks (GAN)" },
        },
        {
          id: "generative-diffusion",
          label: { ru: "Диффузионные модели", en: "Diffusion models" },
        },
      ],
    },
    {
      id: "rl",
      label: { ru: "Обучение с подкреплением", en: "Reinforcement learning" },
      accent: SECTION_ACCENTS.reinforcement,
      children: [
        {
          id: "rl-mdp",
          label: { ru: "Среда, стратегия и награда (MDP)", en: "Environments, policies & rewards (MDP)" },
        },
        {
          id: "rl-q",
          label: { ru: "Q-learning и Value-методы", en: "Q-learning & value methods" },
        },
      ],
    },
    {
      id: "signals",
      label: { ru: "Сигналы и спектры", en: "Signals & spectra" },
      accent: SECTION_ACCENTS.signals,
      children: [
        {
          id: "signals-conv",
          label: { ru: "Свёртка и корреляция", en: "Convolution & correlation" },
        },
        {
          id: "signals-fourier",
          label: { ru: "Преобразование Фурье", en: "Fourier transform" },
        },
      ],
    },
  ],
};

export function findTreeNode(id: string, node: CatalogTreeNode = catalogKnowledgeTree): CatalogTreeNode | undefined {
  if (node.id === id) return node;
  for (const child of node.children ?? []) {
    const found = findTreeNode(id, child);
    if (found) return found;
  }
  return undefined;
}
