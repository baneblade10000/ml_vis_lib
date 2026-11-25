// ============================================================
// palette.js — дизайн-токены для <canvas>
// ============================================================
// Страница — отдельный документ, поэтому токены --nn-* объявлены на :root
// в style.css (зеркало network-playground.css). DOM берёт их напрямую,
// canvas ходит сюда, чтобы всё жило на одной палитре.

let rootStyle = null;

function root() {
  if (!rootStyle) rootStyle = getComputedStyle(document.documentElement);
  return rootStyle;
}

const cache = new Map();

/** Значение custom-свойства: token('--nn-accent') → '#3457a3'. */
export function token(name) {
  let value = cache.get(name);
  if (value === undefined) {
    value = root().getPropertyValue(name).trim();
    // Пустое — токен ещё не применён; перечитать в следующий раз.
    if (value) cache.set(name, value);
  }
  return value;
}

/** Токен вида '#rrggbb' как { r, g, b }. */
export function tokenRgb(name) {
  const hex = token(name).slice(1);
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
  };
}

export function rgba({ r, g, b }, alpha) {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Строка font для canvas в шрифте плейграунда. */
export function font(size, weight = '') {
  return `${weight} ${size}px ${token('--nn-font')}`.trim();
}

// Шкала вероятности класса: класс 0 → белый → класс 1, как у границы
// решений в network-playground (colors.ts: valueToRgb, gamma 1.6).
const SCALE_GAMMA = 1.6;
let scale = null;

function scaleEndpoints() {
  if (!scale) {
    scale = {
      negative: tokenRgb('--scale-negative'),
      zero: tokenRgb('--scale-zero'),
      positive: tokenRgb('--scale-positive'),
    };
  }
  return scale;
}

const mix = (a, b, t) => ({
  r: Math.round(a.r + (b.r - a.r) * t),
  g: Math.round(a.g + (b.g - a.g) * t),
  b: Math.round(a.b + (b.b - a.b) * t),
});

/** p = 0 даёт цвет класса 0, p = 1 — класса 1, p = 0.5 — белый. */
export function classRgb(probability) {
  const v = Math.min(1, Math.max(0, probability)) * 2 - 1;
  const shaped = Math.sign(v) * Math.abs(v) ** SCALE_GAMMA;
  const { negative, zero, positive } = scaleEndpoints();
  return shaped < 0 ? mix(zero, negative, -shaped) : mix(zero, positive, shaped);
}

export function classColor(probability, alpha = 1) {
  const { r, g, b } = classRgb(probability);
  return alpha === 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const linear = (channel) => {
  const s = channel / 255;
  return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};
const luminance = ({ r, g, b }) =>
  0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);

const INK_DARK_LUMINANCE = luminance({ r: 15, g: 23, b: 42 }); // --nn-text

/** Цвет текста на шкале: тёмный, пока он читается лучше белого. */
export function classInk(probability) {
  const lum = luminance(classRgb(probability));
  const darkContrast = (lum + 0.05) / (INK_DARK_LUMINANCE + 0.05);
  const whiteContrast = 1.05 / (lum + 0.05);
  return darkContrast >= whiteContrast ? token('--nn-text') : token('--nn-surface');
}
