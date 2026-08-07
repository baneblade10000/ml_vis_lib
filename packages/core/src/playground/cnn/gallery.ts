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

/**
 * Educational CNN datasets — binary classification, tiny resolution for
 * interactive browser training. Prefer recognizable shapes over abstract blobs.
 */
export type CnnDatasetId2D = "digits" | "circles-squares" | "bars";
export type CnnDatasetId1D = "heartbeat" | "tones" | "pulses";
export type CnnDatasetId = CnnDatasetId2D | CnnDatasetId1D;

export const CNN_DATASET_IDS_2D: CnnDatasetId2D[] = ["digits", "circles-squares", "bars"];
export const CNN_DATASET_IDS_1D: CnnDatasetId1D[] = ["heartbeat", "tones", "pulses"];

/** Slightly larger than before so digits stay readable after 2× pooling. */
export const IMAGE_SIZE = 16;
export const SIGNAL_LENGTH = 48;
export const NUM_EXAMPLES = 160;
/** Train/test split fraction. */
export const DEFAULT_TRAIN_RATIO = 0.5;

// ─── RNG helpers ──────────────────────────────────────────────────────────────

function uniform(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function randInt(min: number, maxInclusive: number): number {
  return Math.floor(uniform(min, maxInclusive + 1));
}

/** Additive Gaussian noise per pixel; `level` ∈ [0,1] scales σ. */
function addNoise(value: number, level: number): number {
  if (level <= 0) return value;
  const u1 = Math.random() || 1e-9;
  const u2 = Math.random();
  const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return value + n * level;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function paint(img: Map2D, r: number, c: number, v: number): void {
  const rr = Math.round(r);
  const cc = Math.round(c);
  if (rr < 0 || rr >= IMAGE_SIZE || cc < 0 || cc >= IMAGE_SIZE) return;
  img[rr][cc] = Math.max(img[rr][cc], clamp01(v));
}

/** Soft circular brush — looks more “handwritten” than hard pixels. */
function brush(img: Map2D, r: number, c: number, radius: number, ink = 1): void {
  const rad = Math.max(0.6, radius);
  const r0 = Math.floor(r - rad - 1);
  const r1 = Math.ceil(r + rad + 1);
  const c0 = Math.floor(c - rad - 1);
  const c1 = Math.ceil(c + rad + 1);
  for (let y = r0; y <= r1; y++) {
    for (let x = c0; x <= c1; x++) {
      const d = Math.hypot(y - r, x - c);
      if (d > rad + 0.5) continue;
      const falloff = Math.exp(-(d * d) / (2 * (rad * 0.55) ** 2));
      paint(img, y, x, ink * falloff);
    }
  }
}

function drawLine(
  img: Map2D,
  r0: number,
  c0: number,
  r1: number,
  c1: number,
  thickness: number,
): void {
  const steps = Math.max(2, Math.ceil(Math.hypot(r1 - r0, c1 - c0) * 2));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    brush(img, r0 + (r1 - r0) * t, c0 + (c1 - c0) * t, thickness);
  }
}

function finishImage(img: Map2D, noise: number): Map2D {
  for (let r = 0; r < IMAGE_SIZE; r++) {
    for (let c = 0; c < IMAGE_SIZE; c++) {
      img[r][c] = clamp01(addNoise(img[r][c], noise));
    }
  }
  return img;
}

// ─── 2-D image generators ─────────────────────────────────────────────────────

/**
 * MNIST-style binary digit task: class 1 = “1”, class 0 = “0”.
 * Thick strokes + jitter so each sample varies like real handwriting.
 */
function generateDigits(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const ox = uniform(-1.2, 1.2);
  const oy = uniform(-1.2, 1.2);
  const thick = uniform(0.85, 1.35);

  if (label === 1) {
    const top = 2 + oy;
    const bot = IMAGE_SIZE - 3 + oy;
    const x = IMAGE_SIZE * 0.5 + ox + uniform(-0.8, 0.8);
    const lean = uniform(-0.9, 0.9);
    drawLine(img, top, x - lean, bot, x + lean, thick);
    // optional top serif (like handwritten 1)
    if (Math.random() < 0.65) {
      drawLine(img, top, x - lean - 1.6, top + 0.4, x - lean + 0.2, thick * 0.85);
    }
  } else {
    const cx = IMAGE_SIZE * 0.5 + ox;
    const cy = IMAGE_SIZE * 0.5 + oy;
    const rx = uniform(3.6, 5.2);
    const ry = uniform(4.2, 5.8);
    const steps = 48;
    for (let i = 0; i <= steps; i++) {
      const a = (i / steps) * Math.PI * 2;
      brush(img, cy + Math.sin(a) * ry, cx + Math.cos(a) * rx, thick * 0.95);
    }
  }
  return finishImage(img, noise);
}

/** Class 1: filled circle. Class 0: filled square. Classic shape classification. */
function generateCirclesSquares(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const cx = IMAGE_SIZE * 0.5 + uniform(-1.5, 1.5);
  const cy = IMAGE_SIZE * 0.5 + uniform(-1.5, 1.5);

  if (label === 1) {
    const radius = uniform(3.5, 5.5);
    for (let r = 0; r < IMAGE_SIZE; r++) {
      for (let c = 0; c < IMAGE_SIZE; c++) {
        const d = Math.hypot(r - cy, c - cx);
        img[r][c] = d <= radius ? 1 : d <= radius + 0.7 ? 1 - (d - radius) / 0.7 : 0;
      }
    }
  } else {
    const half = uniform(3.2, 5.0);
    for (let r = 0; r < IMAGE_SIZE; r++) {
      for (let c = 0; c < IMAGE_SIZE; c++) {
        const inside = Math.abs(r - cy) <= half && Math.abs(c - cx) <= half;
        img[r][c] = inside ? 1 : 0;
      }
    }
  }
  return finishImage(img, noise);
}

/** Class 1: horizontal bar. Class 0: vertical bar — edge-orientation toy task. */
function generateBars(label: 0 | 1, noise: number): Map2D {
  const img = zeros2D(IMAGE_SIZE, IMAGE_SIZE);
  const thick = uniform(1.2, 2.4);
  const pos = uniform(4, IMAGE_SIZE - 5);
  const len0 = uniform(2, 4);
  const len1 = IMAGE_SIZE - 1 - uniform(2, 4);

  if (label === 1) {
    drawLine(img, pos, len0, pos + uniform(-0.6, 0.6), len1, thick);
  } else {
    drawLine(img, len0, pos, len1, pos + uniform(-0.6, 0.6), thick);
  }
  return finishImage(img, noise);
}

const IMAGE_GENERATORS: Record<CnnDatasetId2D, (label: 0 | 1, noise: number) => Map2D> = {
  digits: generateDigits,
  "circles-squares": generateCirclesSquares,
  bars: generateBars,
};

// ─── 1-D signal generators ────────────────────────────────────────────────────

/** Class 1: QRS-like heartbeat spike. Class 0: quiet baseline + noise. */
function generateHeartbeat(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH).fill(0);
  const center = uniform(SIGNAL_LENGTH * 0.3, SIGNAL_LENGTH * 0.7);
  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    let v = 0.05 * Math.sin(i * 0.35);
    if (label === 1) {
      const d = i - center;
      // small Q dip, tall R peak, mild S
      v += -0.25 * Math.exp(-0.5 * ((d + 2.2) / 1.1) ** 2);
      v += 1.15 * Math.exp(-0.5 * (d / 1.35) ** 2);
      v += -0.35 * Math.exp(-0.5 * ((d - 2.8) / 1.4) ** 2);
    }
    values[i] = addNoise(v, noise);
  }
  return values;
}

/** Class 1: high-frequency tone. Class 0: low-frequency tone. */
function generateTones(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH);
  const freq = label === 1 ? uniform(0.35, 0.55) : uniform(0.06, 0.12);
  const phase = uniform(0, Math.PI * 2);
  const amp = uniform(0.7, 1);
  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    values[i] = addNoise(amp * Math.sin(2 * Math.PI * freq * i + phase), noise);
  }
  return values;
}

/** Class 1: single rectangular pulse. Class 0: two separated pulses. */
function generatePulses(label: 0 | 1, noise: number): number[] {
  const values = new Array<number>(SIGNAL_LENGTH).fill(-0.25);
  const width = randInt(3, 6);

  const paintPulse = (start: number) => {
    for (let i = start; i < start + width && i < SIGNAL_LENGTH; i++) values[i] = 1;
  };

  if (label === 1) {
    paintPulse(randInt(8, SIGNAL_LENGTH - width - 8));
  } else {
    const gap = randInt(6, 12);
    const first = randInt(4, Math.max(5, SIGNAL_LENGTH - width * 2 - gap - 4));
    paintPulse(first);
    paintPulse(first + width + gap);
  }

  for (let i = 0; i < SIGNAL_LENGTH; i++) values[i] = addNoise(values[i], noise);
  return values;
}

const SIGNAL_GENERATORS: Record<CnnDatasetId1D, (label: 0 | 1, noise: number) => number[]> = {
  heartbeat: generateHeartbeat,
  tones: generateTones,
  pulses: generatePulses,
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

/**
 * Wrap a single image (h×w) as a one-channel volume.
 * Returns a view (no row copy) — layers must not mutate the example pixels.
 */
export function imageToVolume(pixels: Map2D): Map2D[] {
  return [pixels];
}

/**
 * Wrap a signal as a one-channel input.
 * Returns a view (no copy) — layers must not mutate the example values.
 */
export function signalToInput(values: number[]): number[][] {
  return [values];
}
