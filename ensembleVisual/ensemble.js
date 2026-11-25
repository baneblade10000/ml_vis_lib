// ============================================================
// ensemble.js — алгоритмы + утилиты визуализации
// ============================================================

export function makeData(n = 250, noise = 0.15, seed = 42) {
  let s = seed;
  const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const x1 = rnd() * 2 - 1;
    const x2 = rnd() * 2 - 1;
    const label = (x1 * x1 + x2 * x2 < 0.5) ? 1 : 0;
    X.push([x1, x2]);
    y.push(rnd() < noise ? 1 - label : label);
  }
  return { X, y };
}

function gini(y) {
  const n = y.length; if (!n) return 0;
  let c1 = 0; for (const v of y) c1 += v;
  const p1 = c1 / n; return 1 - p1 * p1 - (1 - p1) ** 2;
}

function bestSplitClf(X, y, featIdx, minLeaf) {
  const vals = [...new Set(X.map(r => r[featIdx]))].sort((a, b) => a - b);
  let best = { gain: 0, thresh: null };
  const parent = gini(y), n = y.length;
  for (let i = 0; i < vals.length - 1; i++) {
    const t = (vals[i] + vals[i + 1]) / 2;
    const yL = [], yR = [];
    for (let j = 0; j < n; j++) (X[j][featIdx] <= t ? yL : yR).push(y[j]);
    if (yL.length < minLeaf || yR.length < minLeaf) continue;
    const w = (yL.length * gini(yL) + yR.length * gini(yR)) / n;
    if (parent - w > best.gain) best = { gain: parent - w, thresh: t };
  }
  return best;
}

export function buildTreeClf(X, y, opts = {}) {
  const { maxDepth = 4, nFeatures = 1, minLeaf = 3, depth = 0 } = opts;
  const node = { pred: y.reduce((a, b) => a + b, 0) / (y.length || 1), n: y.length };
  if (depth >= maxDepth || y.length < 2 * minLeaf || gini(y) < 1e-6) return node;

  const feats = [];
  const all = X[0] ? X[0].map((_, i) => i) : [0];
  while (feats.length < Math.min(nFeatures, all.length)) {
    const f = all[Math.floor(Math.random() * all.length)];
    if (!feats.includes(f)) feats.push(f);
  }
  let best = { gain: 0, thresh: null, feat: null };
  for (const f of feats) {
    const s = bestSplitClf(X, y, f, minLeaf);
    if (s.gain > best.gain) best = { ...s, feat: f };
  }
  if (best.feat === null || best.gain <= 1e-6) return node;

  const XL = [], yL = [], XR = [], yR = [];
  for (let i = 0; i < X.length; i++) {
    if (X[i][best.feat] <= best.thresh) { XL.push(X[i]); yL.push(y[i]); }
    else { XR.push(X[i]); yR.push(y[i]); }
  }
  node.feat = best.feat; node.thresh = best.thresh; node.gain = best.gain;
  node.left = buildTreeClf(XL, yL, { ...opts, depth: depth + 1 });
  node.right = buildTreeClf(XR, yR, { ...opts, depth: depth + 1 });
  return node;
}

export function predictTree(node, x) {
  while (node.left) node = (x[node.feat] <= node.thresh) ? node.left : node.right;
  return node.pred;
}

export function bootstrap(X, y, frac = 1.0) {
  const n = Math.max(2, Math.floor(X.length * frac));
  const Xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const k = Math.floor(Math.random() * X.length);
    Xs.push(X[k]); ys.push(y[k]);
  }
  return { X: Xs, y: ys };
}

export function buildForest(X, y, opts = {}) {
  const { nTrees = 10, maxDepth = 4, nFeatures = 1, minLeaf = 3, sampleFrac = 1.0 } = opts;
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const s = bootstrap(X, y, sampleFrac);
    trees.push(buildTreeClf(s.X, s.y, { maxDepth, nFeatures, minLeaf }));
  }
  return trees;
}

export function forestPredict(trees, x) {
  let s = 0; for (const t of trees) s += predictTree(t, x);
  return s / trees.length;
}

function buildTreeReg(X, y, opts = {}) {
  const { maxDepth = 3, minLeaf = 3, depth = 0 } = opts;
  const mean = y.reduce((a, b) => a + b, 0) / (y.length || 1);
  const node = { pred: mean, n: y.length };
  if (depth >= maxDepth || y.length < 2 * minLeaf) return node;

  const parentVar = y.reduce((a, v) => a + (v - mean) ** 2, 0);
  let best = { gain: 0, thresh: null, feat: null };
  const nFeats = X.length ? X.length : 0;

  for (let f = 0; f < nFeats; f++) {
    const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const t = (vals[i] + vals[i + 1]) / 2;
      const yL = [], yR = [];
      for (let j = 0; j < X.length; j++) (X[j][f] <= t ? yL : yR).push(y[j]);
      if (yL.length < minLeaf || yR.length < minLeaf) continue;
      const mL = yL.reduce((a, b) => a + b, 0) / yL.length;
      const mR = yR.reduce((a, b) => a + b, 0) / yR.length;
      const sse = yL.reduce((a, v) => a + (v - mL) ** 2, 0)
                + yR.reduce((a, v) => a + (v - mR) ** 2, 0);
      if (parentVar - sse > best.gain) best = { gain: parentVar - sse, thresh: t, feat: f };
    }
  }
  if (best.feat === null || best.gain <= 1e-9) return node;

  const XL = [], yL = [], XR = [], yR = [];
  for (let i = 0; i < X.length; i++) {
    if (X[i][best.feat] <= best.thresh) { XL.push(X[i]); yL.push(y[i]); }
    else { XR.push(X[i]); yR.push(y[i]); }
  }
  node.feat = best.feat; node.thresh = best.thresh; node.gain = best.gain;
  node.left = buildTreeReg(XL, yL, { ...opts, depth: depth + 1 });
  node.right = buildTreeReg(XR, yR, { ...opts, depth: depth + 1 });
  return node;
}

export const sigmoid = z => 1 / (1 + Math.exp(-z));

export function initBoost(X, y, lr = 0.3) {
  const p = Math.min(0.99, Math.max(0.01, y.reduce((a, b) => a + b, 0) / y.length));
  const F0 = Math.log(p / (1 - p));
  return {
    F: X.map(() => F0), F0, trees: [], lr,
    history: [{ loss: logLoss(y, X.map(() => F0)), acc: accuracy(X.map(() => F0), y) }]
  };
}

export function boostStep(state, X, y, treeOpts = {}) {
  const { lr } = state;
  const residuals = X.map((_, i) => y[i] - sigmoid(state.F[i]));
  const tree = buildTreeReg(X, residuals, treeOpts);
  const newF = X.map((_, i) => state.F[i] + lr * predictTree(tree, X[i]));
  return {
    ...state,
    F: newF,
    trees: [...state.trees, tree],
    history: [...state.history, { loss: logLoss(y, newF), acc: accuracy(newF, y) }]
  };
}

export function boostPredict(state, x) {
  let f = state.F0;
  for (const t of state.trees) f += state.lr * predictTree(t, x);
  return sigmoid(f);
}

export function logLoss(y, F) {
  let s = 0;
  for (let i = 0; i < y.length; i++) {
    const p = Math.min(0.999, Math.max(0.001, sigmoid(F[i])));
    s += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
  }
  return s / y.length;
}

export function accuracy(F, y) {
  let c = 0;
  for (let i = 0; i < y.length; i++) if ((F[i] > 0 ? 1 : 0) === y[i]) c++;
  return c / y.length;
}

export function drawTreeNode(ctx, node, size, opts = {}) {
  const { color = '#60a5fa' } = opts;
  if (!node.left) return;
  const line = (n) => {
    if (!n.left) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.5;
    ctx.beginPath();
    if (n.feat === 0) {
      const px = ((n.thresh + 1) / 2) * size;
      ctx.moveTo(px, 0); ctx.lineTo(px, size);
    } else {
      const py = (1 - (n.thresh + 1) / 2) * size;
      ctx.moveTo(0, py); ctx.lineTo(size, py);
    }
    ctx.stroke(); ctx.globalAlpha = 1;
    line(n.left); line(n.right);
  };
  line(node);
}

export function drawLossChart(ctx, w, h, values, color = '#f59e0b', opts = {}) {
  const {
    xLabel = 'итерация',
    yLabel = 'log-loss',
    title = '',
  } = opts;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fafbfd';
  ctx.fillRect(0, 0, w, h);

  const padL = 44, padR = 10, padT = title ? 22 : 12, padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL + 0.5, padT + 0.5, plotW, plotH);

  if (!values || values.length === 0) {
    ctx.fillStyle = '#8a99ac';
    ctx.font = '11px Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('нет данных', padL + plotW / 2, padT + plotH / 2);
    return;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (!isFinite(min) || !isFinite(max)) return;
  if (max - min < 1e-6) { max = min + 0.1; min = Math.max(0, min - 0.1); }
  const span = max - min;
  min = Math.max(0, min - span * 0.1);
  max = max + span * 0.1;

  const n = values.length;
  const xAt = i => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = v => padT + plotH - ((v - min) / (max - min)) * plotH;

  ctx.font = '10px Roboto, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 3; k++) {
    const v = min + (max - min) * (k / 3);
    const y = yAt(v);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    ctx.fillStyle = '#8a99ac';
    ctx.fillText(v.toFixed(2), padL - 6, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xTicks = Math.min(6, n - 1);
  if (xTicks >= 1) {
    for (let k = 0; k <= xTicks; k++) {
      const i = Math.round((k / xTicks) * (n - 1));
      const x = xAt(i);
      ctx.strokeStyle = '#eef2f7';
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#8a99ac';
      ctx.fillText(String(i + 1), x, padT + plotH + 4);
    }
  } else {
    ctx.fillStyle = '#8a99ac';
    ctx.fillText('1', xAt(0), padT + plotH + 4);
  }

  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xAt(i), y = yAt(values[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (n <= 40) {
    ctx.fillStyle = color;
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(xAt(i), yAt(values[i]), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = '#5b6b7f';
  ctx.font = '10px Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(xLabel, padL + plotW / 2, h - 10);

  ctx.save();
  ctx.translate(10, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  if (title) {
    const last = values[n - 1];
    ctx.fillStyle = '#3b4a5c';
    ctx.font = '600 11px Roboto, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`${title}: ${last.toFixed(3)}`, padL, 4);
  }
}
// ============================================================
// Bias-Variance Decomposition (BVD) для Brier Score
// ============================================================

export function drawBvdChart(ctx, w, h, series, opts = {}) {
  const {
    xLabel = 'шаг',
    yLabel = 'ошибка',
    title = 'Bias² и Variance',
  } = opts;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fafbfd';
  ctx.fillRect(0, 0, w, h);

  const padL = 44, padR = 10, padT = 24, padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL + 0.5, padT + 0.5, plotW, plotH);

  if (!series || !series.length || !series[0].values.length) {
    ctx.fillStyle = '#8a99ac';
    ctx.font = '11px Roboto, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Нажмите «Оценить Bias² / Variance»', padL + plotW / 2, padT + plotH / 2);
    return;
  }

  const allVals = series.flatMap(s => s.values).filter(isFinite);
  let min = 0;
  let max = Math.max(...allVals);
  if (max < 1e-4) max = 0.1;
  max = max * 1.12;

  const n = series[0].values.length;
  const xAt = i => padL + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = v => padT + plotH - ((v - min) / (max - min)) * plotH;

  // Сетка по Y
  ctx.font = '10px Roboto, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 3; k++) {
    const v = min + (max - min) * (k / 3);
    const y = yAt(v);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    ctx.fillStyle = '#8a99ac';
    ctx.fillText(v.toFixed(3), padL - 5, y);
  }

  // Сетка по X
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xTicks = Math.min(6, n - 1);
  if (xTicks >= 1) {
    for (let k = 0; k <= xTicks; k++) {
      const i = Math.round((k / xTicks) * (n - 1));
      const x = xAt(i);
      ctx.strokeStyle = '#eef2f7';
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#8a99ac';
      ctx.fillText(String(i + 1), x, padT + plotH + 4);
    }
  } else {
    ctx.fillStyle = '#8a99ac';
    ctx.fillText('1', xAt(0), padT + plotH + 4);
  }

  // Оси
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // Кривые
  for (const s of series) {
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dashed ? 1.5 : 2.2;
    if (s.dashed) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xAt(i), y = yAt(s.values[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Заголовок и цветная легенда вверху
  ctx.font = '600 10px Roboto, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  let curX = padL;
  ctx.fillStyle = '#3b4a5c';
  ctx.fillText(title + ':', curX, 5);
  curX += ctx.measureText(title + ': ').width + 4;

  for (const s of series) {
    const last = s.values[n - 1];
    const txt = `${s.label}=${last.toFixed(3)}`;
    ctx.fillStyle = s.color;
    ctx.fillText(txt, curX, 5);
    curX += ctx.measureText(txt).width + 8;
  }

  ctx.fillStyle = '#5b6b7f';
  ctx.font = '10px Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(xLabel, padL + plotW / 2, h - 10);
}
