import { describe, expect, it } from "vitest";
import { makeDataset } from "./datasets";
import { predictProbabilities, trainBatch, accuracy } from "./mlp";
import { advanceLiveTraining, createLiveTrainingState, DEFAULT_CONFIG } from "./index";

describe("makeDataset", () => {
  it("generates reproducible samples", () => {
    const config = { ...DEFAULT_CONFIG, seed: 42 };
    const first = makeDataset(config);
    const second = makeDataset(config);
    expect(first).toEqual(second);
    expect(first.length).toBeGreaterThanOrEqual(16);
  });
});

describe("mlp", () => {
  it("improves accuracy after training", () => {
    const config = { ...DEFAULT_CONFIG, sampleCount: 64, gridSize: 20, epochs: 20 };
    const state = createLiveTrainingState(config);
    const before = accuracy(state.mlp, state.split.trainFeatures, state.split.trainLabels);
    for (let epoch = 0; epoch < 10; epoch++) {
      const batch = state.split.trainFeatures;
      const labels = state.split.trainLabels;
      trainBatch(state.mlp, batch, labels);
    }
    const after = accuracy(state.mlp, state.split.trainFeatures, state.split.trainLabels);
    expect(after).toBeGreaterThanOrEqual(before);
    const probs = predictProbabilities(state.mlp, state.split.trainFeatures.slice(0, 3));
    expect(probs.every((p) => p >= 0 && p <= 1)).toBe(true);
  });
});

describe("live training", () => {
  it("advances epochs and records history", () => {
    const state = createLiveTrainingState({ ...DEFAULT_CONFIG, epochs: 5 });
    const next = advanceLiveTraining(state, 3);
    expect(next.epoch).toBe(3);
    expect(next.history.length).toBeGreaterThan(1);
  });
});
