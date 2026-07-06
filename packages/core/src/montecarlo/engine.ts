import { createRng } from "../playground/rng";
import {
  DEFAULT_MONTE_CARLO_CONFIG,
  PI_TRUE,
  type MonteCarloConfig,
  type MonteCarloHistoryPoint,
  type MonteCarloPoint,
} from "./types";

export function isInsideUnitQuarterCircle(x: number, y: number): boolean {
  return x * x + y * y <= 1;
}

export function estimatePi(insideCount: number, totalSamples: number): number {
  if (totalSamples <= 0) return 0;
  return (4 * insideCount) / totalSamples;
}

export function estimateError(estimate: number): number {
  return Math.abs(estimate - PI_TRUE);
}

export class MonteCarloPiEngine {
  points: MonteCarloPoint[] = [];
  totalSamples = 0;
  insideCount = 0;
  history: MonteCarloHistoryPoint[] = [];
  config: MonteCarloConfig;
  private rng: () => number;

  constructor(config: Partial<MonteCarloConfig> = {}) {
    this.config = { ...DEFAULT_MONTE_CARLO_CONFIG, ...config };
    this.rng = createRng(this.config.seed);
    this.recordHistory();
  }

  get piEstimate(): number {
    return estimatePi(this.insideCount, this.totalSamples);
  }

  get error(): number {
    return estimateError(this.piEstimate);
  }

  get insideRatio(): number {
    return this.totalSamples === 0 ? 0 : this.insideCount / this.totalSamples;
  }

  addBatch(count = this.config.batchSize): void {
    const batch = Math.max(1, count);
    for (let i = 0; i < batch; i++) {
      const x = this.rng();
      const y = this.rng();
      const inside = isInsideUnitQuarterCircle(x, y);
      this.points.push({ x, y, inside });
      if (inside) this.insideCount += 1;
      this.totalSamples += 1;
    }

    this.recordHistory();
  }

  reset(seed = this.config.seed): void {
    this.config = { ...this.config, seed };
    this.rng = createRng(seed);
    this.points = [];
    this.totalSamples = 0;
    this.insideCount = 0;
    this.history = [];
    this.recordHistory();
  }

  updateConfig(patch: Partial<MonteCarloConfig>): void {
    this.config = { ...this.config, ...patch };
    if (patch.seed != null) {
      this.reset(patch.seed);
    }
  }

  private recordHistory(): void {
    const estimate = this.piEstimate;
    const last = this.history[this.history.length - 1];
    const minGap = Math.max(1, Math.floor(this.totalSamples / 400));
    if (last && this.totalSamples > 0 && this.totalSamples - last.samples < minGap) {
      this.history[this.history.length - 1] = {
        samples: this.totalSamples,
        estimate,
        error: estimateError(estimate),
      };
      return;
    }

    this.history.push({
      samples: this.totalSamples,
      estimate,
      error: estimateError(estimate),
    });
  }
}
