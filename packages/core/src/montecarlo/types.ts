export interface MonteCarloPoint {
  x: number;
  y: number;
  inside: boolean;
}

export interface MonteCarloHistoryPoint {
  samples: number;
  estimate: number;
  error: number;
}

export interface MonteCarloConfig {
  batchSize: number;
  maxVisiblePoints: number;
  seed: number;
}

export const DEFAULT_MONTE_CARLO_CONFIG: MonteCarloConfig = {
  batchSize: 1,
  maxVisiblePoints: 12_000,
  seed: 42,
};

export const PI_TRUE = Math.PI;
