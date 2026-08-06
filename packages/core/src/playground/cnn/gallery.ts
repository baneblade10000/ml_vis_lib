import { zeros2D, type Map2D } from "./tensor";

/** A labeled 2-D image example (single grayscale channel). */
export interface ImageExample {
  pixels: Map2D;
  label: 0 | 1;
}

/** A labeled 1-D signal example. */
export interface SignalExample {
  values: number[];
  label: 0 | 1;
}

export type CnnDatasetId2D = "disc-ring" | "cross-square" | "blob-gradient";
export type CnnDatasetId1D = "sine-pulse" | "step" | "ramp";
export type CnnDatasetId = CnnDatasetId2D | CnnDatasetId1D;

export const CNN_DATASET_IDS_2D: CnnDatasetId2D[] = ["disc-ring", "cross-square", "blob-gradient"];
export const CNN_DATASET_IDS_1D: CnnDatasetId1D[] = ["sine-pulse", "step", "ramp"];

export const IMAGE_SIZE = 12;
export const SIGNAL_LENGTH = 48;
export const NUM_EXAMPLES = 120;
/** Train/test split fraction. */
export const DEFAULT_TRAIN_RATIO = 0.5;

// ─── RNG helpers ──────────────────────────────────────────────────────────────

function uniform(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/** Additive Gaussian noise per pixel; `level` ∈ [0,1] scales σ. */
function addNoise(value: number, level: number): number {
  if (level <= 0) return value;
  const u1 = Math.random() || 1e-9;
  const u2 = Math.random();
  const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return value + n * level;
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ─── 2-D image generators ─────────────────────────────────────────────────────

const CENTER = (IMAGE_SIZE - 1) / 2;

/**
 * Class 1: a filled disc (bright centre). Class 0: a hollow ring (dark centre,
 * bright rim). The brightness profile is a smooth bump so convolutions have
 * real spatial structure to exploit.
 */
function generateDiscRing(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const inner = IMAGE_SIZE * 0.22;
  const outer = IMAGE_SIZE * 0.46;
  for (let r = 0; r < IMAGE_SIZE; r++) {
    for (let c = 0; c < IMAGE_SIZE; c++) {
      const dx = c - CENTER;
      const dy = r - CENTER;
      const d = Math.sqrt(dx * dx + dy * dy);
      let v: number;
      if (label === 1) {
        v = d < outer ? Math.exp(-(d * d) / (2 * inner * inner)) : 0;
      } else {
        v = Math.exp(-((d - outer) ** 2) / (2 * inner * inner));
      }
      img[r][c] = addNoise(v, noise);
    }
  }
  return img;
}

/** Class 1: a plus/cross. Class 0: a square frame. */
function generateCrossSquare(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const arm = Math.floor(IMAGE_SIZE * 0.18);
  const frame = Math.floor(IMAGE_SIZE * 0.12);
  for (let r = 0; r < IMAGE_SIZE; r++) {
    for (let c = 0; c < IMAGE_SIZE; c++) {
      let v = 0;
      if (label === 1) {
        if (Math.abs(r - CENTER) <= arm || Math.abs(c - CENTER) <= arm) v = 1;
      } else {
        const onEdge =
          Math.min(Math.abs(r - 0), Math.abs(r - (IMAGE_SIZE - 1))) < frame ||
          Math.min(Math.abs(c - 0), Math.abs(c - (IMAGE_SIZE - 1))) < frame;
        const inside = r > frame && r < IMAGE_SIZE - 1 - frame && c > frame && c < IMAGE_SIZE - 1 - frame;
        if (onEdge && inside) v = 1;
      }
      img[r][c] = addNoise(v, noise);
    }
  }
  return img;
}

/** Class 1: a bright Gaussian blob in the centre. Class 0: a smooth diagonal gradient. */
function generateBlobGradient(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const sigma = IMAGE_SIZE * 0.28;
  for (let r = 0; r < IMAGE_SIZE; r++) {
    for (let c = 0; c < IMAGE_SIZE; c++) {
      let v: number;
      if (label === 1) {
        const dx = c - CENTER;
        const dy = r - CENTER;
        v = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      } else {
        v = (r + c) / (2 * (IMAGE_SIZE - 1));
      }
      img[r][c] = addNoise(v, noise);
    }
  }
  return img;
}

const IMAGE_GENERATORS: Record<CnnDatasetId2D, (label: 0 | 1, noise: number) => Map2D> = {
  "disc-ring": generateDiscRing,
  "cross-square": generateCrossSquare,
  "blob-gradient": generateBlobGradient,
};

// ─── 1-D signal generators ────────────────────────────────────────────────────

/**
 * Class 1: a localized sine bump centred at a random position. Class 0: a flat
 * baseline with no structured pulse. A 1-D conv can learn to detect the bump.
 */
function generateSinePulse(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH);
  const center = uniform(SIGNAL_LENGTH * 0.25, SIGNAL_LENGTH * 0.75);
  const width = SIGNAL_LENGTH * 0.12;
  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    let v: number;
    if (label === 1) {
      const d = (i - center) / width;
      v = Math.exp(-0.5 * d * d) * Math.sin(2 * Math.PI * (i - center) / (width * 1.5));
    } else {
      v = 0.1 * Math.sin(i * 0.5);
    }
    values[i] = addNoise(v, noise);
  }
  return values;
}

/** Class 1: a single rectangular pulse. Class 0: an alternating ± square wave. */
function generateStep(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH);
  const start = Math.floor(uniform(SIGNAL_LENGTH * 0.2, SIGNAL_LENGTH * 0.4));
  const width = Math.floor(uniform(SIGNAL_LENGTH * 0.15, SIGNAL_LENGTH * 0.3));
  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    let v: number;
    if (label === 1) {
      v = i >= start && i < start + width ? 1 : -0.4;
    } else {
      v = Math.floor(i / (SIGNAL_LENGTH * 0.16)) % 2 === 0 ? 0.6 : -0.6;
    }
    values[i] = addNoise(v, noise);
  }
  return values;
}

/** Class 1: a rising ramp. Class 0: a flat plateau. */
function generateRamp(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH);
  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    let v: number;
    if (label === 1) {
      v = i / (SIGNAL_LENGTH - 1);
    } else {
      v = 0.5;
    }
    values[i] = addNoise(v, noise);
  }
  return values;
}

const SIGNAL_GENERATORS: Record<CnnDatasetId1D, (label: 0 | 1, noise: number) => number[]> = {
  "sine-pulse": generateSinePulse,
  step: generateStep,
  ramp: generateRamp,
};

// ─── Public dataset API ───────────────────────────────────────────────────────

/** Build a 2-D dataset, half class 0 and half class 1, shuffled. */
export function makeImageDataset(
  id: CnnDatasetId2D,
  count = NUM_EXAMPLES,
  noiseLevel = 0,
): ImageExample[] {
  const gen = IMAGE_GENERATORS[id];
  const sigma = noiseLevel * 0.2;
  const examples: ImageExample[] = [];
  const perClass = Math.floor(count / 2);
  for (let i = 0; i < perClass; i++) {
    examples.push({ pixels: gen(0, sigma), label: 0 });
    examples.push({ pixels: gen(1, sigma), label: 1 });
  }
  shuffleInPlace(examples);
  return examples;
}

/** Build a 1-D dataset, half class 0 and half class 1, shuffled. */
export function makeSignalDataset(
  id: CnnDatasetId1D,
  count = NUM_EXAMPLES,
  noiseLevel = 0,
): SignalExample[] {
  const gen = SIGNAL_GENERATORS[id];
  const sigma = noiseLevel * 0.25;
  const examples: SignalExample[] = [];
  const perClass = Math.floor(count / 2);
  for (let i = 0; i < perClass; i++) {
    examples.push({ values: gen(0, sigma), label: 0 });
    examples.push({ values: gen(1, sigma), label: 1 });
  }
  shuffleInPlace(examples);
  return examples;
}

/** Wrap a single image (h×w) as a one-channel {@link import("./tensor").Volume}. */
export function imageToVolume(pixels: Map2D): Map2D[] {
  return [pixels.map((row) => row.slice())];
}

/** Wrap a signal as a one-channel {@link import("./tensor").Signal}. */
export function signalToInput(values: number[]): number[][] {
  return [values.slice()];
}
