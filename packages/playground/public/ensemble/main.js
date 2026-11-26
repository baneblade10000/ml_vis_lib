import {
  buildForest, predictTree,
  initBoost, boostStep, boostPredict,
  drawTreeNode as drawTreeOverlay,
  drawLossChart, drawBvdChart, accuracy, logLoss, sigmoid
} from './ensemble.js';


import {
  createView, worldToScreen, screenToWorld,
  drawAxes, drawProbField, drawPointsOnPlot
} from './plot2d.js';

import { initDrawer, openTree, updateDrawer, setTitle } from './tree-drawer.js';
import { token, tokenRgb, rgba, classColor } from './palette.js';

const $ = id => document.getElementById(id);
const getCtx = id => $(id)?.getContext('2d') || null;
const SIZE = 380;

// Цвета серий графиков: у каждой модели свой акцент, Test-линия темнее Train.
const SERIES = {
  forest: token('--nn-accent'),
  forestTest: token('--nn-accent-hover'),
  forestVar: token('--scale-positive'),
  boost: token('--ens-boost'),
  boostTest: token('--ens-boost-deep'),
  boostVar: token('--nn-crimson'),
  ok: token('--ens-ok'),
  okDeep: token('--ens-ok-deep'),
};

const state = {
  X: [], y: [], isTest: [],
  forest: null, boost: null,
  playing: false, timer: null,
  forestStep: 0,
  selected: null,
  shape: 'circle',
  view: createView(),
  params: {
    n: 250, noise: 0.15, testSplit: 0.25,
    nTrees: 12, maxDepthF: 5, nFeaturesF: 2, sampleFracF: 1.0,
    nIters: 25, maxDepthB: 3, lr: 0.3,
  }
};

// Плоскость рисуется в 2×: на широких карточках она показывается крупнее 380 px.
const RES = 2;
const cvF = $('cvForest'), cvB = $('cvBoost');
for (const cv of [cvF, cvB]) cv.width = cv.height = SIZE * RES;
const ctxF = cvF.getContext('2d'), ctxB = cvB.getContext('2d');
ctxF.scale(RES, RES);
ctxB.scale(RES, RES);
const ctxChartFTrain = $('chartForestTrain').getContext('2d');
const ctxChartFTest  = $('chartForestTest').getContext('2d');
const ctxChartBTrain = $('chartBoostTrain').getContext('2d');
const ctxChartBTest  = $('chartBoostTest').getContext('2d');
const ctxChartFBias = getCtx('chartForestBias');
const ctxChartFVar  = getCtx('chartForestVar');
const ctxChartBBias = getCtx('chartBoostBias');
const ctxChartBVar  = getCtx('chartBoostVar');

initDrawer();
function evalMetrics(probs, y) {
  if (!y.length) return { acc: 0, loss: 0 };
  const F = probs.map(p => Math.log(Math.max(1e-9, p) / Math.max(1e-9, 1 - p)));
  return { acc: accuracy(F, y), loss: logLoss(y, F) };
}
function getActiveTableForest() {
  if (!TABLE.forest) return [];
  const step = TABLE.forestStep != null ? TABLE.forestStep : TABLE.forest.length;
  return TABLE.forest.slice(0, step);
}

function getSplit() {
  const Xtr = [], ytr = [], Xte = [], yte = [];
  for (let i = 0; i < state.X.length; i++) {
    if (state.isTest[i]) { Xte.push(state.X[i]); yte.push(state.y[i]); }
    else { Xtr.push(state.X[i]); ytr.push(state.y[i]); }
  }
  // Защита на случай, если testSplit = 0 или 1
  if (!Xtr.length) return { Xtr: state.X, ytr: state.y, Xte: [], yte: [] };
  return { Xtr, ytr, Xte, yte };
}

function splitTrainTest() {
  const n = state.X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  // Перемешиваем индексы (Фишер-Йейтс)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.round(n * state.params.testSplit);
  state.isTest = new Array(n).fill(false);
  for (let k = 0; k < nTest; k++) state.isTest[idx[k]] = true;
}

// ---------- генерация данных ----------
function makeShaped(shape, n, noise) {
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    let x1, x2, label;
    if (shape === 'circle') {
      x1 = Math.random() * 2 - 1; x2 = Math.random() * 2 - 1;
      label = x1 * x1 + x2 * x2 < 0.5 ? 1 : 0;
    } else if (shape === 'xor') {
      x1 = Math.random() * 2 - 1; x2 = Math.random() * 2 - 1;
      label = (x1 * x2 > 0) ? 1 : 0;
    } else if (shape === 'spiral') {
      const t = Math.random() * 3 * Math.PI;
      const r = 0.1 + t / (3 * Math.PI) * 0.9;
      const cls = Math.random() < 0.5 ? 1 : 0;
      const ang = t + (cls ? Math.PI : 0);
      x1 = r * Math.cos(ang) + (Math.random() - 0.5) * 0.05;
      x2 = r * Math.sin(ang) + (Math.random() - 0.5) * 0.05;
      label = cls;
    } else {
      const t = Math.random() * Math.PI;
      if (Math.random() < 0.5) {
        x1 = Math.cos(t) - 0.5; x2 = Math.sin(t) - 0.2; label = 0;
      } else {
        x1 = Math.cos(t + Math.PI) + 0.5; x2 = Math.sin(t + Math.PI) + 0.2; label = 1;
      }
      x1 += (Math.random() - 0.5) * 0.1; x2 += (Math.random() - 0.5) * 0.1;
    }
    X.push([x1, x2]);
    y.push(Math.random() < noise ? 1 - label : label);
  }
  return { X, y };
}

function regen() {
  const d = makeShaped(state.shape, state.params.n, state.params.noise);
  state.X = d.X; state.y = d.y;
  splitTrainTest();
  state.view = createView();
  reset();
}


// ---------- хелперы ----------
function fmtCell(v) {
  if (v == null || isNaN(v)) return '';
  return String(Math.round(v * 1000) / 1000);
}

// парсинг числа с поддержкой запятой и промежуточных состояний
function parseNum(raw) {
  if (raw == null) return { v: NaN, pending: true };
  const s = String(raw).replace(',', '.').trim();
  if (s === '' || s === '-' || s === '.' || s === '-.' ||
      s === '0.' || s === '-0.' || s === '+.' || s === '+') {
    return { v: NaN, pending: true };
  }
  const v = parseFloat(s);
  return { v, pending: false };
}

function forestPreds(trees, X) {
  return X.map(x => {
    if (!trees.length) return 0.5;
    let s = 0; for (const t of trees) s += predictTree(t, x);
    return s / trees.length;
  });
}

// ---------- рендер леса ----------
// ---------- рендер леса ----------
function renderForest() {
  const trees = state.forest ? state.forest.slice(0, state.forestStep) : [];
  const view = state.view;
  drawAxes(ctxF, SIZE, view);

  if (trees.length > 0) {
    drawProbField(ctxF, SIZE, view, x => {
      let s = 0; for (const t of trees) s += predictTree(t, x);
      return s / trees.length;
    });
    drawAxesOverlay(ctxF, SIZE, view);
  }

  drawPointsOnPlot(ctxF, SIZE, view, state.X, state.y, {
    selected: state.selected != null ? state.selected : -1,
    isTest: state.isTest
  });

  const { Xtr, ytr, Xte, yte } = getSplit();
  const tr = evalMetrics(forestPreds(trees, Xtr), ytr);
  const te = evalMetrics(forestPreds(trees, Xte), yte);

  $('mForestTrees').textContent = `${trees.length}/${state.forest?.length || 0}`;
  $('mForestAcc').textContent = trees.length
    ? `${(tr.acc * 100).toFixed(0)}% / ${(te.acc * 100).toFixed(0)}%` : '—';
  $('mForestLoss').textContent = trees.length
    ? `${tr.loss.toFixed(2)} / ${te.loss.toFixed(2)}` : '—';
  $('statForest').textContent = trees.length
    ? `test ${(te.acc * 100).toFixed(1)}%` : '—';
}


// ---------- рендер бустинга ----------
function renderBoost() {
  const view = state.view;
  drawAxes(ctxB, SIZE, view);

  if (state.boost) {
    const b = state.boost;
    drawProbField(ctxB, SIZE, view, x => boostPredict(b, x));
    drawAxesOverlay(ctxB, SIZE, view);

    // Веса остатков считаем только для Train, для Test = 0
    const residuals = state.X.map((x, i) =>
      state.isTest[i] ? 0 : Math.abs(state.y[i] - boostPredict(b, x))
    );
    drawPointsOnPlot(ctxB, SIZE, view, state.X, state.y, {
      weights: residuals,
      selected: state.selected != null ? state.selected : -1,
      isTest: state.isTest
    });

    const { Xtr, ytr, Xte, yte } = getSplit();
    const tr = evalMetrics(Xtr.map(x => boostPredict(b, x)), ytr);
    const te = evalMetrics(Xte.map(x => boostPredict(b, x)), yte);

    $('mBoostIters').textContent = b.trees.length;
    $('mBoostAcc').textContent = `${(tr.acc * 100).toFixed(0)}% / ${(te.acc * 100).toFixed(0)}%`;
    $('mBoostLoss').textContent = `${tr.loss.toFixed(2)} / ${te.loss.toFixed(2)}`;
    $('statBoost').textContent = `test ${(te.acc * 100).toFixed(1)}%`;
  } else {
    drawPointsOnPlot(ctxB, SIZE, view, state.X, state.y, {
      selected: state.selected != null ? state.selected : -1,
      isTest: state.isTest
    });
    $('mBoostIters').textContent = '0';
    $('mBoostAcc').textContent = '—';
    $('mBoostLoss').textContent = '—';
    $('statBoost').textContent = '—';
  }
}

function drawAxesOverlay(ctx, size, view) {
  ctx.save();
  ctx.strokeStyle = rgba(tokenRgb('--nn-text-secondary'), 0.6);
  ctx.lineWidth = 1.2;
  const [ox, oy] = worldToScreen(view, size, 0, 0);
  if (oy >= 0 && oy <= size) {
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(size, oy); ctx.stroke();
  }
  if (ox >= 0 && ox <= size) {
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, size); ctx.stroke();
  }
  ctx.restore();
}

// ---------- миниатюры ----------
function renderMinis(container, trees, source) {
  container.innerHTML = '';
  trees.forEach((tree, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'mini';
    wrap.title = 'Открыть диаграмму дерева';
    wrap.onclick = () => {
      setTitle(source === 'forest' ? '🌲 Дерево леса' : '🚀 Итерация бустинга');
      openTree(trees, idx, state.selected != null ? state.X[state.selected] : null);
    };
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    // Линии разбивки — тёмными: акцент сливается с полями цвета класса.
    const miniGrid = { major: token('--nn-border'), minor: token('--nn-surface-muted') };
    drawAxes(g, 64, { cx: 0, cy: 0, scale: 30 }, miniGrid);
    for (let px = 0; px < 64; px += 3) {
      for (let py = 0; py < 64; py += 3) {
        const [x, y2] = screenToWorld({ cx: 0, cy: 0, scale: 30 }, 64, px + 1.5, py + 1.5);
        const v = predictTree(tree, [x, y2]);
        g.fillStyle = classColor(v, 0.55);
        g.fillRect(px, py, 3, 3);
      }
    }
    drawPointsOnPlot(g, 64, { cx: 0, cy: 0, scale: 30 }, state.X, state.y, { scale: 0.55 });
    drawTreeOverlay(g, tree, 64, { color: token('--nn-text') });
    wrap.appendChild(c);
    const num = document.createElement('span');
    num.className = 'num'; num.textContent = idx + 1;
    wrap.appendChild(num);
    container.appendChild(wrap);
  });
}

// ---------- графики ----------
// ---------- графики ----------
function renderCharts() {
  if (!ctxChartFTrain || !ctxChartFTest || !ctxChartBTrain || !ctxChartBTest) return;
  const { Xtr, ytr, Xte, yte } = getSplit();

  // 1. Loss для Random Forest
  if (state.forest && state.forestStep > 0) {
    const fTrain = [], fTest = [];
    for (let k = 1; k <= state.forestStep; k++) {
      const trees = state.forest.slice(0, k);
      fTrain.push(evalMetrics(forestPreds(trees, Xtr), ytr).loss);
      if (Xte.length) fTest.push(evalMetrics(forestPreds(trees, Xte), yte).loss);
    }
    drawLossChart(ctxChartFTrain, 380, 120, fTrain, SERIES.forest, {
      xLabel: 'число деревьев', yLabel: 'loss', title: 'Лес · Train Loss'
    });
    drawLossChart(ctxChartFTest, 380, 120, fTest, SERIES.forestTest, {
      xLabel: 'число деревьев', yLabel: 'loss', title: 'Лес · Test Loss'
    });
  } else {
    drawLossChart(ctxChartFTrain, 380, 120, [], SERIES.forest, { title: 'Лес · Train Loss' });
    drawLossChart(ctxChartFTest, 380, 120, [], SERIES.forestTest, { title: 'Лес · Test Loss' });
  }

  // 2. Loss для Gradient Boosting
  if (state.boost) {
    const b = state.boost;
    const bTrain = b.history.map(h => h.loss);
    const bTest = [];

    if (Xte.length) {
      let f = Xte.map(() => b.F0);
      bTest.push(logLoss(yte, f));
      for (let k = 0; k < b.trees.length; k++) {
        f = f.map((val, idx) => val + b.lr * predictTree(b.trees[k], Xte[idx]));
        bTest.push(logLoss(yte, f));
      }
    }

    drawLossChart(ctxChartBTrain, 380, 120, bTrain, SERIES.boost, {
      xLabel: 'итерация', yLabel: 'loss', title: 'Бустинг · Train Loss'
    });
    drawLossChart(ctxChartBTest, 380, 120, bTest, SERIES.boostTest, {
      xLabel: 'итерация', yLabel: 'loss', title: 'Бустинг · Test Loss'
    });
  } else {
    drawLossChart(ctxChartBTrain, 380, 120, [], SERIES.boost, { title: 'Бустинг · Train Loss' });
    drawLossChart(ctxChartBTest, 380, 120, [], SERIES.boostTest, { title: 'Бустинг · Test Loss' });
  }

  // 3. Пошаговая отрисовка Bias² и Variance (срез до текущего шага)
  if (ctxChartFBias) {
    const fStep = state.forest ? state.forestStep : 0;
    const bStep = state.boost ? state.boost.trees.length : 0;

    const fBias = (state.bvdData && fStep > 0) ? state.bvdData.fBias.slice(0, fStep) : [];
    const fVar  = (state.bvdData && fStep > 0) ? state.bvdData.fVar.slice(0, fStep)  : [];
    const bBias = (state.bvdData && bStep > 0) ? state.bvdData.bBias.slice(0, bStep) : [];
    const bVar  = (state.bvdData && bStep > 0) ? state.bvdData.bVar.slice(0, bStep)  : [];

    drawBvdChart(ctxChartFBias, 380, 115, fBias.length ? [{ label: 'Bias²', values: fBias, color: SERIES.forest }] : [], { xLabel: 'число деревьев', title: 'Лес · Bias²' });
    drawBvdChart(ctxChartFVar,  380, 115, fVar.length  ? [{ label: 'Var',   values: fVar,  color: SERIES.forestVar }] : [], { xLabel: 'число деревьев', title: 'Лес · Variance' });
    drawBvdChart(ctxChartBBias, 380, 115, bBias.length ? [{ label: 'Bias²', values: bBias, color: SERIES.boost }] : [], { xLabel: 'итерация', title: 'Бустинг · Bias²' });
    drawBvdChart(ctxChartBVar,  380, 115, bVar.length  ? [{ label: 'Var',   values: bVar,  color: SERIES.boostVar }] : [], { xLabel: 'итерация', title: 'Бустинг · Variance' });
  }
}
function computeTabularBvd(mode) {
  const d = imputeTable();
  if (d.Xtr.length < 6) return null;

  const evalX = d.Xte.length >= 4 ? d.Xte : d.Xtr;
  const evalY = d.Xte.length >= 4 ? d.yte : d.ytr;
  const G = evalX.length;
  const M = 15;
  const steps = mode === 'forest' ? TABLE.params.nTrees : TABLE.params.nIters;
  const predsByStep = Array.from({ length: steps }, () => []);

  for (let m = 0; m < M; m++) {
    const Xb = [], yb = [];
    for (let i = 0; i < d.Xtr.length; i++) {
      const k = Math.floor(Math.random() * d.Xtr.length);
      Xb.push(d.Xtr[k]); yb.push(d.ytr[k]);
    }
    if (mode === 'forest') {
      const nFeat = Math.min(TABLE.feats, Math.max(1, TABLE.params.nFeaturesF));
      const forest = buildForest(Xb, yb, {
        nTrees: steps, maxDepth: TABLE.params.maxDepthF,
        nFeatures: nFeat, sampleFrac: TABLE.params.sampleFracF,
      });
      const sumG = new Float64Array(G);
      for (let t = 0; t < steps; t++) {
        for (let g = 0; g < G; g++) sumG[g] += predictTree(forest[t], evalX[g]);
        const avg = new Float64Array(G);
        for (let g = 0; g < G; g++) avg[g] = sumG[g] / (t + 1);
        predsByStep[t].push(avg);
      }
    } else {
      let bst = initBoost(Xb, yb, TABLE.params.lr);
      for (let it = 0; it < steps; it++) {
        bst = boostStep(bst, Xb, yb, { maxDepth: TABLE.params.maxDepthB });
        const cur = new Float64Array(G);
        for (let g = 0; g < G; g++) cur[g] = boostPredict(bst, evalX[g]);
        predsByStep[it].push(cur);
      }
    }
  }

  const bias2Arr = [], varArr = [];
  for (let k = 0; k < steps; k++) {
    let sumB2 = 0, sumVar = 0;
    for (let g = 0; g < G; g++) {
      let meanP = 0;
      for (let m = 0; m < M; m++) meanP += predsByStep[k][m][g];
      meanP /= M;
      sumB2 += (meanP - evalY[g]) ** 2;
      let v = 0;
      for (let m = 0; m < M; m++) v += (predsByStep[k][m][g] - meanP) ** 2;
      sumVar += v / M;
    }
    bias2Arr.push(sumB2 / G);
    varArr.push(sumVar / G);
  }
  return { bias2Arr, varArr };
}
function compute2dBvdData() {
  const M = 15;
  const grid = [], eta = [];
  const step = 0.12;
  for (let x1 = -0.95; x1 <= 0.95; x1 += step) {
    for (let x2 = -0.95; x2 <= 0.95; x2 += step) {
      grid.push([x1, x2]);
      eta.push(trueProb(state.shape, x1, x2, state.params.noise));
    }
  }
  const G = grid.length;
  const nTrees = state.params.nTrees;
  const nIters = state.params.nIters;

  const fPreds = Array.from({ length: nTrees }, () => []);
  const bPreds = Array.from({ length: nIters }, () => []);

  for (let m = 0; m < M; m++) {
    const d = makeShaped(state.shape, state.params.n, state.params.noise);
    const nTr = Math.max(10, Math.round(d.X.length * (1 - state.params.testSplit)));
    const Xtr = d.X.slice(0, nTr);
    const ytr = d.y.slice(0, nTr);

    const forest = buildForest(Xtr, ytr, {
      nTrees, maxDepth: state.params.maxDepthF,
      nFeatures: state.params.nFeaturesF, sampleFrac: state.params.sampleFracF
    });
    const sumGrid = new Float64Array(G);
    for (let t = 0; t < nTrees; t++) {
      for (let g = 0; g < G; g++) sumGrid[g] += predictTree(forest[t], grid[g]);
      const avg = new Float64Array(G);
      for (let g = 0; g < G; g++) avg[g] = sumGrid[g] / (t + 1);
      fPreds[t].push(avg);
    }

    let bst = initBoost(Xtr, ytr, state.params.lr);
    for (let it = 0; it < nIters; it++) {
      bst = boostStep(bst, Xtr, ytr, { maxDepth: state.params.maxDepthB });
      const cur = new Float64Array(G);
      for (let g = 0; g < G; g++) cur[g] = boostPredict(bst, grid[g]);
      bPreds[it].push(cur);
    }
  }

  const calcCurve = (predsByStep) => {
    const bias2Arr = [], varArr = [];
    for (let k = 0; k < predsByStep.length; k++) {
      let sumB2 = 0, sumVar = 0;
      for (let g = 0; g < G; g++) {
        let meanP = 0;
        for (let m = 0; m < M; m++) meanP += predsByStep[k][m][g];
        meanP /= M;
        sumB2 += (meanP - eta[g]) ** 2;
        let v = 0;
        for (let m = 0; m < M; m++) v += (predsByStep[k][m][g] - meanP) ** 2;
        sumVar += v / M;
      }
      bias2Arr.push(sumB2 / G);
      varArr.push(sumVar / G);
    }
    return { bias2Arr, varArr };
  };

  const resF = calcCurve(fPreds);
  const resB = calcCurve(bPreds);
  state.bvdData = {
    fBias: resF.bias2Arr,
    fVar: resF.varArr,
    bBias: resB.bias2Arr,
    bVar: resB.varArr
  };
}




// ---------- взаимодействие с canvas ----------
function setupInteraction(canvas, which) {
  if (!canvas) return;
  let dragging = false;
  let dragStart = null;
  let dragMoved = false;

  canvas.addEventListener('mousedown', e => {
    dragging = true;
    dragMoved = false;
    dragStart = { x: e.clientX, y: e.clientY, cx: state.view.cx, cy: state.view.cy };
    canvas.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
    state.view.cx = dragStart.cx - dx / state.view.scale;
    state.view.cy = dragStart.cy + dy / state.view.scale;
    renderAll();
  });

  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    canvas.style.cursor = 'crosshair';
  });

  canvas.addEventListener('click', e => {
    if (dragMoved) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (SIZE / rect.width);
    const py = (e.clientY - rect.top) * (SIZE / rect.height);
    const [wx, wy] = screenToWorld(state.view, SIZE, px, py);

    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < state.X.length; i++) {
      const dx = state.X[i][0] - wx;
      const dy = state.X[i][1] - wy;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }

    const threshold = (16 / state.view.scale) ** 2;
    if (best === -1 || bestD > threshold) {
      selectPoint(null);
      return;
    }
    selectPoint(best, which);
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (SIZE / rect.width);
    const py = (e.clientY - rect.top) * (SIZE / rect.height);
    const [wx, wy] = screenToWorld(state.view, SIZE, px, py);
    const factor = Math.exp(-e.deltaY * 0.0015);
    const newScale = Math.max(20, Math.min(1000, state.view.scale * factor));
    state.view.scale = newScale;
    state.view.cx = wx - (px - SIZE / 2) / newScale;
    state.view.cy = wy + (py - SIZE / 2) / newScale;
    renderAll();
  }, { passive: false });
}


setupInteraction(cvF, 'forest');
setupInteraction(cvB, 'boost');

// ---------- выбор точки ----------
function selectPoint(idx, which) {
  state.selected = idx;
  window.__selectedPoint = idx != null ? state.X[idx] : null;
  const info = $('clickInfo');

  if (idx == null) {
    info.classList.add('empty');
    info.textContent = 'Кликните по точке на сетке или строке таблицы.';
    renderAll(); renderTable(); return;
  }
  info.classList.remove('empty');

  const splitTag = state.isTest[idx] ? ' [Test]' : ' [Train]';
  const ptStr = state.X[idx].map(v => v.toFixed(2)).join(', ');
  let html = `<b>Точка #${idx}${splitTag}</b> (${ptStr}) · истина: <code>${state.y[idx]}</code><br>`;

  const treesF = state.forest ? state.forest.slice(0, state.forestStep) : [];

  // Если кликнули по сетке Леса (или из таблицы, когда обучен только Лес)
  if (which === 'forest' || (which == null && treesF.length)) {
    if (treesF.length) {
      const votes = treesF.map(t => predictTree(t, state.X[idx]));
      const avg = votes.reduce((a, v) => a + v, 0) / votes.length;
      html += `🌲 <b>Лес:</b> p = <code>${avg.toFixed(3)}</code> · голоса: ${votes.map(v => `<code>${v.toFixed(2)}</code>`).join(' ')}<br>`;

      // Сразу открываем панель с деревьями Леса для этой точки
      if (which === 'forest') {
        setTitle('🌲 Дерево леса');
        openTree(treesF, 0, state.X[idx]);
      }
    } else if (which === 'forest') {
      html += `<span class="muted">Сначала постройте Random Forest, чтобы увидеть путь по деревьям.</span><br>`;
    }
  }

  // Если кликнули по сетке Бустинга (или из таблицы, когда обучен Бустинг)
  if (which === 'boost' || (which == null && state.boost)) {
    if (state.boost && state.boost.trees.length) {
      let f = state.boost.F0;
      const contribs = state.boost.trees.map(t => {
        const c = state.boost.lr * predictTree(t, state.X[idx]);
        f += c; return c;
      });
      html += `🚀 <b>Бустинг:</b> F0 = <code>${state.boost.F0.toFixed(2)}</code> + вклады [${contribs.map(c => `<code>${c.toFixed(2)}</code>`).join(', ')}] → p = <code>${sigmoid(f).toFixed(3)}</code>`;

      // Сразу открываем панель с деревьями Бустинга для этой точки
      if (which === 'boost') {
        setTitle('Итерация бустинга');
        openTree(state.boost.trees, 0, state.X[idx]);
      }
    } else if (which === 'boost') {
      html += `<span class="muted">Сначала сделайте хотя бы один шаг бустинга, чтобы увидеть путь по деревьям.</span>`;
    }
  }

  info.innerHTML = html;

  // Если панель уже была открыта (например, при клике из таблицы), просто обновляем путь
  if (which == null && $('treeDrawer').classList.contains('open')) {
    updateDrawer(state.X[idx]);
  }

  renderAll();
  renderTable();
}




// ---------- анимация ----------
function stopPlaying() {
  state.playing = false;
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  $('btnPlayF').textContent = '▶';
  $('btnPlayB').textContent = '▶';
}

function playForest() {
  if (state.playing) { stopPlaying(); return; }
  stopPlaying();
  state.playing = true;
  $('btnPlayF').textContent = '⏸';
  if (!state.forest) {
    const { Xtr, ytr } = getSplit();
    state.forest = buildForest(Xtr, ytr, {
      nTrees: state.params.nTrees, maxDepth: state.params.maxDepthF,
      nFeatures: state.params.nFeaturesF, sampleFrac: state.params.sampleFracF
    });
    state.forestStep = 0;
  }
  state.timer = setInterval(() => {
    if (state.forestStep >= state.forest.length) { stopPlaying(); return; }
    state.forestStep++;
    renderAll(); renderTable();
  }, 500);
}

function playBoost() {
  if (state.playing) { stopPlaying(); return; }
  stopPlaying();
  state.playing = true;
  $('btnPlayB').textContent = '⏸';
  const { Xtr, ytr } = getSplit();
  if (!state.boost) state.boost = initBoost(Xtr, ytr, state.params.lr);
  state.timer = setInterval(() => {
    if (state.boost.trees.length >= state.params.nIters) { stopPlaying(); return; }
    state.boost = boostStep(state.boost, Xtr, ytr, { maxDepth: state.params.maxDepthB });
    renderAll(); renderTable();
  }, 550);
}



function reset() {
  state.forest = null; state.boost = null;
  state.forestStep = 0; state.selected = null;
  state.bvdReady = false; // <-- ДОБАВИТЬ
  window.__selectedPoint = null;
  stopPlaying();
  renderAll();
  renderTable();
}


function renderTable() {
  const tbody = $('dataBody');
  if (!tbody) return;
  const trees = state.forest ? state.forest.slice(0, state.forestStep) : [];
  const fPreds = forestPreds(trees, state.X);
  const bPreds = state.boost ? state.X.map(x => boostPredict(state.boost, x)) : null;
  const filter = $('filterClass')?.value || 'all';
  const splitFilter = $('filterSplit')?.value || 'all';
  const q = ($('searchX')?.value || '').trim();

  tbody.innerHTML = '';
  let shown = 0;

  for (let i = 0; i < state.X.length; i++) {
    if (filter !== 'all' && String(state.y[i]) !== filter) continue;
    const isTe = state.isTest[i];
    if (splitFilter === 'train' && isTe) continue;
    if (splitFilter === 'test' && !isTe) continue;
    if (q) {
      const s = state.X[i].map(v => v.toFixed(3)).join(' ');
      if (!s.includes(q)) continue;
    }
    shown++;

    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    if (state.selected === i) tr.classList.add('selected');
    tr.onclick = (e) => {
      if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') selectPoint(i);
    };

    const tdIdx = document.createElement('td');
    tdIdx.className = 'muted';
    tdIdx.textContent = i;
    tr.appendChild(tdIdx);

    const tdSplit = document.createElement('td');
    tdSplit.innerHTML = isTe
      ? '<span class="split split--test">Test</span>'
      : '<span class="split split--train">Train</span>';
    tr.appendChild(tdSplit);

    const cols = [
      [0, state.X[i][0], 'x'],
      [1, state.X[i][1], 'x'],
      [-1, state.y[i], 'y']
    ];
    for (const [colIdx, val, kind] of cols) {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.className = 'cell-input';
      inp.type = 'text';
      inp.inputMode = 'decimal';
      inp.value = fmtCell(val);

      inp.oninput = () => {
        const { v, pending } = parseNum(inp.value);
        if (pending || (kind === 'y' && v !== 0 && v !== 1)) {
          inp.classList.add('invalid'); return;
        }
        inp.classList.remove('invalid');
        if (kind === 'x') state.X[i][colIdx] = v;
        else state.y[i] = v;
        onDataEdited();
      };

      inp.onblur = () => {
        const { v, pending } = parseNum(inp.value);
        if (pending || isNaN(v)) {
          inp.value = fmtCell(kind === 'x' ? state.X[i][colIdx] : state.y[i]);
        } else if (kind === 'y') {
          state.y[i] = v >= 0.5 ? 1 : 0;
          inp.value = state.y[i];
        } else {
          state.X[i][colIdx] = v;
          inp.value = fmtCell(v);
        }
        inp.classList.remove('invalid');
        onDataEdited();
      };
      td.appendChild(inp);
      tr.appendChild(td);
    }

    const fp = fPreds[i];
    const bp = bPreds ? bPreds[i] : null;
    const fOk = trees.length ? ((fp > 0.5 ? 1 : 0) === state.y[i]) : null;
    const bOk = bPreds ? ((bp > 0.5 ? 1 : 0) === state.y[i]) : null;

    const predCells = [
      { text: trees.length ? fp.toFixed(3) : '—', cls: 'muted' },
      { text: bPreds ? bp.toFixed(3) : '—', cls: 'muted' },
      { text: fOk == null ? '—' : (fOk ? '✓' : '✗'), cls: fOk == null ? 'muted' : (fOk ? 'ok' : 'bad') },
      { text: bOk == null ? '—' : (bOk ? '✓' : '✗'), cls: bOk == null ? 'muted' : (bOk ? 'ok' : 'bad') },
    ];
    for (const c of predCells) {
      const td = document.createElement('td');
      td.className = c.cls;
      td.textContent = c.text;
      tr.appendChild(td);
    }

    const tdDel = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'row-del';
    del.textContent = '×';
    del.title = 'Удалить строку';
    del.onclick = () => {
      state.X.splice(i, 1); state.y.splice(i, 1); state.isTest.splice(i, 1);
      if (state.selected === i) state.selected = null;
      reset();
    };
    tdDel.appendChild(del);
    tr.appendChild(tdDel);

    tbody.appendChild(tr);
  }

  if (!shown) {
    tbody.innerHTML = `<tr><td colspan="10" class="muted" style="text-align:center;padding:20px">Нет строк по фильтру</td></tr>`;
  }
  if ($('dataCount')) $('dataCount').textContent = `${shown} / ${state.X.length}`;
}


if ($('filterSplit')) $('filterSplit').onchange = renderTable;



function renderAll() {
  renderForest();
  renderBoost();
  renderCharts();
  renderMinis($('minisForest'), state.forest ? state.forest.slice(0, state.forestStep) : [], 'forest');
  renderMinis($('minisBoost'), state.boost ? state.boost.trees : [], 'boost');
  window.__selectedPoint = state.selected != null ? state.X[state.selected] : null;
  if ($('treeDrawer').classList.contains('open')) {
    updateDrawer(window.__selectedPoint);
  }
}


// ---------- вкладки ----------
const TAB_PANELS = {
  viz: 'tabViz', data: 'tabData', edit: 'tabEdit',
  tableForest: 'tabTableForest', tableBoost: 'tabTableBoost',
};

// При открытии из каталога (#embedded) вкладки живут в его шапке, и выбранная
// приходит сюда сообщением; своя строка вкладок — для отдельного открытия.
function selectTab(which) {
  const panel = TAB_PANELS[which];
  if (!panel) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === which));
  for (const id of Object.values(TAB_PANELS)) $(id).style.display = 'none';
  $(panel).style.display = 'grid';

  if (which === 'data') renderTable();
  if (which === 'edit') {
    if (!TABLE.X.length) genTable();
    else { renderTableTab(); updateTableStats(); }
  }
  if (which === 'tableForest') {
    if (!TABLE.X.length) genTable();
    renderTableForestResults();
  }
  if (which === 'tableBoost') {
    if (!TABLE.X.length) genTable();
    renderTableBoostResults();
  }
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => selectTab(tab.dataset.tab);
});

window.addEventListener('message', (e) => {
  if (e.origin === location.origin && e.data?.type === 'ml-vis:tab') selectTab(e.data.id);
});

// ---------- переключатель моделей ----------
// Обе модели считаются всегда — переключатель только показывает одну из них.
function selectModel(model) {
  const forest = model === 'forest';
  document.querySelectorAll('#modelSwitch .seg-btn').forEach(b => {
    const active = b.dataset.model === model;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', String(active));
  });
  $('cardForest').hidden = !forest;
  $('cardBoost').hidden = forest;
  $('paramsForest').hidden = !forest;
  $('paramsBoost').hidden = forest;
}

$('modelSwitch').addEventListener('click', (e) => {
  const btn = e.target.closest('.seg-btn');
  if (btn) selectModel(btn.dataset.model);
});


// Кнопка добавления строки в конструктор многомерных таблиц (с возможностью оставить пропуски)
if ($('btnAddTableRow')) {
  $('btnAddTableRow').onclick = () => {
    if (!TABLE.X.length) genTable();
    const row = Array.from({ length: TABLE.feats }, () => +(Math.random() * 2 - 1).toFixed(3));
    TABLE.X.push(row);
    TABLE.y.push(Math.random() < 0.5 ? 0 : 1);
    markTableDirty();
    renderTableTab();
    const last = $('tBody')?.lastElementChild;
    if (last) {
      last.classList.add('row-flash');
      last.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  };
}



$('btnForest').onclick = () => {
  const { Xtr, ytr } = getSplit();
  state.forest = buildForest(Xtr, ytr, {
    nTrees: state.params.nTrees, maxDepth: state.params.maxDepthF,
    nFeatures: state.params.nFeaturesF, sampleFrac: state.params.sampleFracF
  });
  state.forestStep = state.forest.length;
  renderAll(); renderTable();
};
$('btnBoost').onclick = () => {
  const { Xtr, ytr } = getSplit();
  if (!state.boost) state.boost = initBoost(Xtr, ytr, state.params.lr);
  if (state.boost.trees.length >= state.params.nIters) return;
  state.boost = boostStep(state.boost, Xtr, ytr, { maxDepth: state.params.maxDepthB });
  renderAll(); renderTable();
};

$('btnResetF').onclick = () => { state.forest = null; state.forestStep = 0; renderAll(); renderTable(); };
$('btnResetB').onclick = () => { state.boost = null; renderAll(); renderTable(); };
$('btnRegen').onclick = regen;
$('btnPlayF').onclick = playForest;
$('btnPlayB').onclick = playBoost;

function bindZoom(btnId, factor) {
  const b = $(btnId);
  if (!b) return;
  b.onclick = () => {
    state.view.scale = Math.max(20, Math.min(1000, state.view.scale * factor));
    renderAll();
  };
}
bindZoom('zoomIn', 1.3);
bindZoom('zoomOut', 1 / 1.3);
const btnZoomReset = $('zoomReset');
if (btnZoomReset) btnZoomReset.onclick = () => { state.view = createView(); renderAll(); };

$('shapeChips').addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  document.querySelectorAll('#shapeChips .chip').forEach(c => c.classList.remove('active'));
  chip.classList.add('active');
  state.shape = chip.dataset.shape;
  regen();
});

$('filterClass').onchange = renderTable;
$('searchX').oninput = renderTable;

// Доля заполненного трека слайдера (см. input[type=range] в style.css).
function paintRange(slider) {
  slider.style.setProperty('--range-progress', `${((slider.value - slider.min) / (slider.max - slider.min)) * 100}%`);
}

function bindRange(id, key, fmt = v => v, onChange = () => {}) {
  const slider = $(id);
  const num = $(id + 'Num');
  if (!slider) return;

  const min = +slider.min;
  const max = +slider.max;
  const step = +slider.step || 1;
  const parts = String(step).split('.');
  const dec = parts.length > 1 ? parts[1].length : 0;

  const clamp = v => Math.min(max, Math.max(min, v));
  const round = v => step > 0 ? Number((Math.round(v / step) * step).toFixed(dec)) : v;

  slider.value = state.params[key];
  paintRange(slider);
  if (num) num.value = fmt(state.params[key]);

  const apply = (v, updateText = true) => {
    const c = clamp(round(v));
    state.params[key] = c;
    slider.value = c;
    paintRange(slider);
    if (num && updateText) num.value = fmt(c);
  };

  slider.oninput = () => {
    apply(+slider.value, true);
    onChange();
  };

  if (num) {
    num.oninput = () => {
      const raw = num.value.replace(',', '.').trim();
      if (raw === '' || raw === '-' || raw.endsWith('.') || /\.\d*0+$/.test(raw)) return;
      const v = parseFloat(raw);
      if (!isNaN(v)) {
        apply(v, false);
        onChange();
      }
    };

    const commit = () => {
      const v = parseFloat(num.value.replace(',', '.'));
      apply(isNaN(v) ? state.params[key] : v, true);
      onChange();
    };

    num.onchange = commit;
    num.onblur = commit;
    num.onkeydown = (e) => {
      if (e.key === 'Enter') { e.preventDefault(); num.blur(); }
      if (e.key === 'ArrowUp') { e.preventDefault(); apply(state.params[key] + step, true); onChange(); }
      if (e.key === 'ArrowDown') { e.preventDefault(); apply(state.params[key] - step, true); onChange(); }
    };
  }
}


bindRange('n', 'n', v => v, regen);
bindRange('noise', 'noise', v => v.toFixed(2), regen);
bindRange('testSplit', 'testSplit', v => v.toFixed(2), () => { splitTrainTest(); reset(); });
bindRange('nTrees', 'nTrees', v => v, () => { if (state.forest) $('btnForest').click(); });
bindRange('depthF', 'maxDepthF', v => v, () => { if (state.forest) $('btnForest').click(); });
bindRange('nFeatF', 'nFeaturesF', v => v, () => { if (state.forest) $('btnForest').click(); });
bindRange('fracF', 'sampleFracF', v => v.toFixed(2), () => { if (state.forest) $('btnForest').click(); });
bindRange('nIters', 'nIters', v => v);
bindRange('depthB', 'maxDepthB', v => v);
bindRange('lr', 'lr', v => v.toFixed(2));

regen();
// Подписи на canvas рисуются до загрузки Inter — перерисовать после.
document.fonts.ready.then(() => renderAll());

// ============================================================
// РЕДАКТОР ДАННЫХ
// ============================================================
function renderEditTable() {
  const tbody = $('editBody');
  if (!tbody) return;
  tbody.innerHTML = '';
  for (let i = 0; i < state.X.length; i++) {
    const tr = document.createElement('tr');
    tr.dataset.idx = i;

    const tdIdx = document.createElement('td');
    tdIdx.className = 'muted';
    tdIdx.textContent = i;
    tr.appendChild(tdIdx);

    for (const [col, val, kind] of [
      ['x1', state.X[i][0], 'x'],
      ['x2', state.X[i][1], 'x'],
      ['y', state.y[i], 'y'],
    ]) {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.className = 'cell-input';
      inp.type = 'text';
      inp.inputMode = 'decimal';
      inp.autocomplete = 'off';
      inp.value = fmtCell(val);
      inp.dataset.col = col;
      inp.dataset.row = i;

      inp.addEventListener('input', () => {
        const { v, pending } = parseNum(inp.value);
        if (pending) { inp.classList.add('invalid'); return; }
        if (kind === 'y' && v !== 0 && v !== 1) { inp.classList.add('invalid'); return; }
        inp.classList.remove('invalid');
        if (kind === 'x') state.X[i][+inp.dataset.col.slice(1) - 1] = v;
        else state.y[i] = v;
        onDataEdited();
      });

      inp.addEventListener('blur', () => {
        const { v, pending } = parseNum(inp.value);
        if (pending || isNaN(v)) {
          const prev = kind === 'x'
            ? state.X[i][+inp.dataset.col.slice(1) - 1]
            : state.y[i];
          inp.value = fmtCell(prev);
          inp.classList.remove('invalid');
          return;
        }
        if (kind === 'y') {
          const yv = v >= 0.5 ? 1 : 0;
          state.y[i] = yv;
          inp.value = yv;
        } else {
          state.X[i][+inp.dataset.col.slice(1) - 1] = v;
          inp.value = fmtCell(v);
        }
        inp.classList.remove('invalid');
        onDataEdited();
      });

      td.appendChild(inp);
      tr.appendChild(td);
    }

    const tdDel = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'row-del';
    del.title = 'Удалить строку';
    del.textContent = '×';
    del.onclick = () => {
      state.X.splice(i, 1);
      state.y.splice(i, 1);
      if (state.selected === i) state.selected = null;
      else if (state.selected != null && state.selected > i) state.selected--;
      reset();
    };
    tdDel.appendChild(del);
    tr.appendChild(tdDel);

    tbody.appendChild(tr);
  }
  $('editCount').textContent = `${state.X.length} строк`;
}

let editDebounce = null;
function onDataEdited() {
  clearTimeout(editDebounce);
  editDebounce = setTimeout(() => {
    state.forest = null;
    state.boost = null;
    state.forestStep = 0;
    state.selected = null;
    window.__selectedPoint = null;
    stopPlaying();
    renderAll();
    renderTable();
  }, 250);
}

$('btnAddRow').onclick = () => {
  const x1 = Math.random() * 2 - 1;
  const x2 = Math.random() * 2 - 1;
  const y = Math.random() < 0.5 ? 0 : 1;
  state.X.push([x1, x2]);
  state.y.push(y);
  state.forest = null; state.boost = null; state.forestStep = 0;
  renderAll(); renderTable(); renderEditTable();
  const last = $('editBody').lastElementChild;
  if (last) {
    last.classList.add('row-flash');
    last.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
};

$('btnResetData').onclick = () => { regen(); renderEditTable(); };

$('btnExportCsv').onclick = () => {
  const lines = ['x1,x2,y'];
  for (let i = 0; i < state.X.length; i++) {
    lines.push(`${state.X[i][0]},${state.X[i][1]},${state.y[i]}`);
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'ensemble-data.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

$('csvInput').onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result).replace(/\r/g, '');
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    let start = 0;
    const head = lines[0].toLowerCase().split(/[,;\t]/);
    if (head.includes('x1') || head.includes('x2') || head.includes('y')) start = 1;

    const X = [], y = [];
    for (let i = start; i < lines.length; i++) {
      const parts = lines[i].split(/[,;\t]/).map(p => p.trim());
      if (parts.length < 3) continue;
      const x1 = parseFloat(parts[0].replace(',', '.'));
      const x2 = parseFloat(parts[1].replace(',', '.'));
      const yv = parseFloat(parts[2].replace(',', '.'));
      if (!isFinite(x1) || !isFinite(x2)) continue;
      X.push([x1, x2]);
      y.push(yv >= 0.5 ? 1 : 0);
    }
    if (!X.length) { alert('Не удалось распознать CSV. Ожидается: x1,x2,y'); return; }

    state.X = X; state.y = y;
    state.forest = null; state.boost = null; state.forestStep = 0;
    state.selected = null; state.view = createView();
    stopPlaying();
    renderAll(); renderTable(); renderEditTable();
  };
  reader.readAsText(file, 'utf-8');
  e.target.value = '';
};

// ============================================================
// ТАБЛИЧНЫЕ ДАННЫЕ
// ============================================================
const TABLE = {
  X: [], y: [], isTest: [], feats: 3,
  forest: null, boost: null,
  selectedForestRow: null,
  selectedBoostRow: null,
  params: {
    nTrees: 12, maxDepthF: 5, nFeaturesF: 2, sampleFracF: 1.0,
    nIters: 25, maxDepthB: 3, lr: 0.3,
  }
};

function bindTableRange(id, key, fmt = v => v, onChange = () => {}) {
  const slider = $(id), num = $(id + 'Num');
  if (!slider) return;
  const min = +slider.min, max = +slider.max, step = +slider.step || 1;
  const parts = String(step).split('.');
  const dec = parts.length > 1 ? parts[1].length : 0;
  const clamp = v => Math.min(max, Math.max(min, v));
  const round = v => step > 0 ? Number((Math.round(v / step) * step).toFixed(dec)) : v;

  slider.value = TABLE.params[key];
  paintRange(slider);
  if (num) num.value = fmt(TABLE.params[key]);

  const apply = (v, updateText = true) => {
    const c = clamp(round(v));
    TABLE.params[key] = c;
    slider.value = c;
    paintRange(slider);
    if (num && updateText) num.value = fmt(c);
  };

  slider.oninput = () => { apply(+slider.value, true); onChange(); };
  if (num) {
    num.oninput = () => {
      const raw = num.value.replace(',', '.').trim();
      if (raw === '' || raw === '-' || raw.endsWith('.') || /\.\d*0+$/.test(raw)) return;
      const v = parseFloat(raw);
      if (!isNaN(v)) { apply(v, false); onChange(); }
    };
    const commit = () => {
      const v = parseFloat(num.value.replace(',', '.'));
      apply(isNaN(v) ? TABLE.params[key] : v, true);
      onChange();
    };
    num.onchange = commit;
    num.onblur = commit;
    num.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); num.blur(); } };
  }
}

bindTableRange('tfTrees', 'nTrees', v => v, () => { if (TABLE.forest) trainForestOnTable(); });
bindTableRange('tfDepth', 'maxDepthF', v => v, () => { if (TABLE.forest) trainForestOnTable(); });
bindTableRange('tfFeat',  'nFeaturesF', v => v, () => { if (TABLE.forest) trainForestOnTable(); });
bindTableRange('tfFrac',  'sampleFracF', v => v.toFixed(2), () => { if (TABLE.forest) trainForestOnTable(); });
bindTableRange('tbIters', 'nIters', v => v);
bindTableRange('tbDepth', 'maxDepthB', v => v);
bindTableRange('tbLr',    'lr', v => v.toFixed(2));
function splitTableTrainTest() {
  const n = TABLE.X.length;
  const splitFrac = Math.max(0, Math.min(0.5, parseFloat(String($('tTestSplit')?.value || '0.25').replace(',', '.')) || 0.25));
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.round(n * splitFrac);
  TABLE.isTest = new Array(n).fill(false);
  for (let k = 0; k < nTest; k++) TABLE.isTest[idx[k]] = true;
}

function genTable() {
  const rows = Math.max(10, Math.min(2000, +$('tRows')?.value || 200));
  const feats = Math.max(2, Math.min(5, +$('tFeats')?.value || 3));
  const miss = Math.max(0, Math.min(0.5, parseFloat(String($('tMissing')?.value || '0').replace(',', '.')) || 0));
  const noise = Math.max(0, Math.min(0.5, parseFloat(String($('tNoise')?.value || '0').replace(',', '.')) || 0));
  const shape = $('tShape')?.value || 'linear';

  TABLE.feats = feats;
  TABLE.X = [];
  TABLE.y = [];

  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let f = 0; f < feats; f++) row.push(Math.random() * 2 - 1);
    let label;
    if (shape === 'linear') {
      let s = 0.2 + 1.4 * row[0] - 1.1 * row[1];
      for (let f = 2; f < feats; f++) s += 0.4 * row[f];
      label = s > 0 ? 1 : 0;
    } else if (shape === 'xor') {
      label = (row[0] * row[1] > 0) ? 1 : 0;
    } else {
      label = (row[0] * row[0] + row[1] * row[1] < 0.5) ? 1 : 0;
    }
    if (Math.random() < noise) label = 1 - label;
    for (let f = 0; f < feats; f++) if (Math.random() < miss) row[f] = NaN;
    TABLE.X.push(row);
    TABLE.y.push(label);
  }

  splitTableTrainTest();
  markTableDirty();
  renderTableTab();
}

function imputeTable() {
  const n = TABLE.X.length, f = TABLE.feats;
  if (!n) return { Xtr: [], ytr: [], Xte: [], yte: [], rows: 0, imputed: 0, dropped: 0, medians: [] };

  const medians = [];
  for (let j = 0; j < f; j++) {
    const vals = [];
    for (let i = 0; i < n; i++) {
      if (!TABLE.isTest[i] && !isNaN(TABLE.y[i]) && !isNaN(TABLE.X[i][j])) {
        vals.push(TABLE.X[i][j]);
      }
    }
    vals.sort((a, b) => a - b);
    medians.push(vals.length ? vals[Math.floor(vals.length / 2)] : 0);
  }

  const Xtr = [], ytr = [], Xte = [], yte = [];
  let imputed = 0, dropped = 0;

  for (let i = 0; i < n; i++) {
    if (isNaN(TABLE.y[i])) { dropped++; continue; }
    const row = [];
    for (let j = 0; j < f; j++) {
      if (isNaN(TABLE.X[i][j])) { row.push(medians[j]); imputed++; }
      else row.push(TABLE.X[i][j]);
    }
    if (TABLE.isTest[i]) { Xte.push(row); yte.push(TABLE.y[i]); }
    else { Xtr.push(row); ytr.push(TABLE.y[i]); }
  }

  if (!Xtr.length) return { Xtr: Xte, ytr: yte, Xte: [], yte: [], rows: Xte.length, imputed, dropped, medians };
  return { Xtr, ytr, Xte, yte, rows: Xtr.length + Xte.length, imputed, dropped, medians };
}

function renderTableTab() {
  const head = $('tHead'), body = $('tBody');
  if (!head || !body) return;
  const f = TABLE.feats;
  head.innerHTML = '<tr><th>#</th><th>Выборка</th>' +
    Array.from({ length: f }, (_, j) => `<th>x${j + 1}</th>`).join('') +
    '<th>y</th><th></th></tr>';
  body.innerHTML = '';

  for (let i = 0; i < TABLE.X.length; i++) {
    const tr = document.createElement('tr');
    const tdIdx = document.createElement('td');
    tdIdx.className = 'muted';
    tdIdx.textContent = i;
    tr.appendChild(tdIdx);

    const tdSplit = document.createElement('td');
    tdSplit.innerHTML = TABLE.isTest[i]
      ? '<span class="split split--test">Test</span>'
      : '<span class="split split--train">Train</span>';
    tr.appendChild(tdSplit);

    for (let j = 0; j < f; j++) {
      const td = document.createElement('td');
      const v = TABLE.X[i][j];
      const inp = document.createElement('input');
      inp.className = 'cell-input';
      inp.type = 'text';
      inp.inputMode = 'decimal';
      inp.value = isNaN(v) ? '' : fmtCell(v);
      if (isNaN(v)) { inp.placeholder = 'NaN'; inp.classList.add('invalid'); }

      inp.oninput = () => {
        const { v: vv, pending } = parseNum(inp.value);
        if (pending) { inp.classList.add('invalid'); return; }
        TABLE.X[i][j] = vv;
        inp.classList.remove('invalid');
        markTableDirty();
      };
      inp.onblur = () => {
        const { v: vv, pending } = parseNum(inp.value);
        if (pending || isNaN(vv)) {
          TABLE.X[i][j] = NaN;
          inp.value = '';
          inp.placeholder = 'NaN';
          inp.classList.add('invalid');
        } else {
          TABLE.X[i][j] = vv;
          inp.value = fmtCell(vv);
          inp.classList.remove('invalid');
        }
        markTableDirty();
      };
      td.appendChild(inp);
      tr.appendChild(td);
    }

    const tdY = document.createElement('td');
    const inpY = document.createElement('input');
    inpY.className = 'cell-input';
    inpY.type = 'text';
    inpY.inputMode = 'numeric';
    const yv = TABLE.y[i];
    inpY.value = isNaN(yv) ? '' : String(yv);
    if (isNaN(yv)) { inpY.placeholder = 'NaN'; inpY.classList.add('invalid'); }

    inpY.oninput = () => {
      const s = inpY.value.trim();
      if (s === '') { TABLE.y[i] = NaN; inpY.classList.add('invalid'); markTableDirty(); return; }
      if (s === '0' || s === '1') { TABLE.y[i] = +s; inpY.classList.remove('invalid'); markTableDirty(); }
      else inpY.classList.add('invalid');
    };
    inpY.onblur = () => {
      const s = inpY.value.trim();
      if (s === '' || s === '0' || s === '1') {
        TABLE.y[i] = s === '' ? NaN : +s;
        inpY.classList.toggle('invalid', s === '');
        if (s === '') inpY.placeholder = 'NaN';
      } else {
        inpY.value = isNaN(TABLE.y[i]) ? '' : String(TABLE.y[i]);
        inpY.classList.remove('invalid');
      }
      markTableDirty();
    };
    tdY.appendChild(inpY);
    tr.appendChild(tdY);

    const tdDel = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'row-del';
    del.textContent = '×';
    del.onclick = () => {
      TABLE.X.splice(i, 1); TABLE.y.splice(i, 1); TABLE.isTest.splice(i, 1);
      markTableDirty(); renderTableTab();
    };
    tdDel.appendChild(del);
    tr.appendChild(tdDel);

    body.appendChild(tr);
  }
  if ($('tCount')) $('tCount').textContent = `${TABLE.X.length} строк · ${f} признаков`;
}

function updateTableStats() {
  const d = imputeTable();
  const info = $('tDataInfo');
  if (info) {
    info.textContent = `Train: ${d.Xtr.length} · Test: ${d.Xte.length}${d.imputed ? ` · пропусков импутировано: ${d.imputed}` : ''}${d.dropped ? ` · отброшено: ${d.dropped}` : ''}`;
  }
  renderTableSummary();
}


function markTableDirty() {
  TABLE.forest = null; TABLE.boost = null;
  TABLE.selectedForestRow = null; TABLE.selectedBoostRow = null;
  updateTableStats();
  if ($('tfStatTrain')) $('tfStatTrain').textContent = '—';
  if ($('tfStatTest'))  $('tfStatTest').textContent = '—';
  if ($('tbStatTrain')) $('tbStatTrain').textContent = '—';
  if ($('tbStatTest'))  $('tbStatTest').textContent = '—';
  renderTableForestResults();
  renderTableBoostResults();
}
function renderTableCharts() {
  const d = imputeTable();

  // --- RANDOM FOREST CHARTS ---
  const activeTrees = getActiveTableForest();
  const cTfTrL = getCtx('chartTfTrain'), cTfTeL = getCtx('chartTfTest');
  const cTfTrA = getCtx('chartTfAccTrain'), cTfTeA = getCtx('chartTfAccTest');

  if (cTfTrL && cTfTeL) {
    if (activeTrees.length > 0) {
      const trLoss = [], teLoss = [], trAcc = [], teAcc = [];
      for (let k = 1; k <= activeTrees.length; k++) {
        const sub = activeTrees.slice(0, k);
        const mTr = evalMetrics(forestPreds(sub, d.Xtr), d.ytr);
        trLoss.push(mTr.loss);
        trAcc.push(mTr.acc);
        if (d.Xte.length) {
          const mTe = evalMetrics(forestPreds(sub, d.Xte), d.yte);
          teLoss.push(mTe.loss);
          teAcc.push(mTe.acc);
        }
      }
      drawLossChart(cTfTrL, 340, 115, trLoss, SERIES.forest, { xLabel: 'деревьев', yLabel: 'loss', title: 'Лес · Train Loss' });
      drawLossChart(cTfTeL, 340, 115, teLoss, SERIES.forestTest, { xLabel: 'деревьев', yLabel: 'loss', title: 'Лес · Test Loss' });
      if (cTfTrA) drawLossChart(cTfTrA, 340, 115, trAcc, SERIES.ok, { xLabel: 'деревьев', yLabel: 'acc', title: 'Лес · Train Acc' });
      if (cTfTeA) drawLossChart(cTfTeA, 340, 115, teAcc, SERIES.okDeep, { xLabel: 'деревьев', yLabel: 'acc', title: 'Лес · Test Acc' });
    } else {
      drawLossChart(cTfTrL, 340, 115, [], SERIES.forest, { title: 'Лес · Train Loss' });
      drawLossChart(cTfTeL, 340, 115, [], SERIES.forestTest, { title: 'Лес · Test Loss' });
      if (cTfTrA) drawLossChart(cTfTrA, 340, 115, [], SERIES.ok, { title: 'Лес · Train Acc' });
      if (cTfTeA) drawLossChart(cTfTeA, 340, 115, [], SERIES.okDeep, { title: 'Лес · Test Acc' });
    }
  }

  // --- GRADIENT BOOSTING CHARTS ---
  const cTbTrL = getCtx('chartTbTrain'), cTbTeL = getCtx('chartTbTest');
  const cTbTrA = getCtx('chartTbAccTrain'), cTbTeA = getCtx('chartTbAccTest');

  if (cTbTrL && cTbTeL) {
    if (TABLE.boost && TABLE.boost.trees.length > 0) {
      const b = TABLE.boost;
      const trLoss = b.history.map(h => h.loss);
      const trAcc  = b.history.map(h => h.acc);
      const teLoss = [], teAcc = [];

      if (d.Xte.length) {
        let f = d.Xte.map(() => b.F0);
        teLoss.push(logLoss(d.yte, f));
        teAcc.push(accuracy(f, d.yte));
        for (let k = 0; k < b.trees.length; k++) {
          f = f.map((val, idx) => val + b.lr * predictTree(b.trees[k], d.Xte[idx]));
          teLoss.push(logLoss(d.yte, f));
          teAcc.push(accuracy(f, d.yte));
        }
      }
      drawLossChart(cTbTrL, 340, 115, trLoss, SERIES.boost, { xLabel: 'итераций', yLabel: 'loss', title: 'Бустинг · Train Loss' });
      drawLossChart(cTbTeL, 340, 115, teLoss, SERIES.boostTest, { xLabel: 'итераций', yLabel: 'loss', title: 'Бустинг · Test Loss' });
      if (cTbTrA) drawLossChart(cTbTrA, 340, 115, trAcc, SERIES.ok, { xLabel: 'итераций', yLabel: 'acc', title: 'Бустинг · Train Acc' });
      if (cTbTeA) drawLossChart(cTbTeA, 340, 115, teAcc, SERIES.okDeep, { xLabel: 'итераций', yLabel: 'acc', title: 'Бустинг · Test Acc' });
    } else {
      drawLossChart(cTbTrL, 340, 115, [], SERIES.boost, { title: 'Бустинг · Train Loss' });
      drawLossChart(cTbTeL, 340, 115, [], SERIES.boostTest, { title: 'Бустинг · Test Loss' });
      if (cTbTrA) drawLossChart(cTbTrA, 340, 115, [], SERIES.ok, { title: 'Бустинг · Train Acc' });
      if (cTbTeA) drawLossChart(cTbTeA, 340, 115, [], SERIES.okDeep, { title: 'Бустинг · Test Acc' });
    }
  }
    drawActiveChartsDrawer(false);

}
function updateTableModelStats() {
  const d = imputeTable();

  // 1. Random Forest metrics (updated on every step)
  const activeTrees = getActiveTableForest();
  const totalTrees = TABLE.forest ? TABLE.forest.length : TABLE.params.nTrees;
  if ($('tfStepStat')) $('tfStepStat').textContent = `${activeTrees.length} / ${totalTrees}`;

  if (activeTrees.length > 0) {
    const tr = evalMetrics(forestPreds(activeTrees, d.Xtr), d.ytr);
    const te = evalMetrics(forestPreds(activeTrees, d.Xte), d.yte);
    if ($('tfStatTrain')) $('tfStatTrain').textContent = `${(tr.acc * 100).toFixed(1)}% / ${tr.loss.toFixed(3)}`;
    if ($('tfStatTest'))  $('tfStatTest').textContent  = d.Xte.length ? `${(te.acc * 100).toFixed(1)}% / ${te.loss.toFixed(3)}` : '—';
  } else {
    if ($('tfStatTrain')) $('tfStatTrain').textContent = '—';
    if ($('tfStatTest'))  $('tfStatTest').textContent  = '—';
  }

  // 2. Gradient Boosting metrics (updated on every step)
  const iters = TABLE.boost ? TABLE.boost.trees.length : 0;
  if ($('tbStepStat')) $('tbStepStat').textContent = `${iters} / ${TABLE.params.nIters}`;

  if (iters > 0) {
    const tr = evalMetrics(d.Xtr.map(x => boostPredict(TABLE.boost, x)), d.ytr);
    const te = evalMetrics(d.Xte.map(x => boostPredict(TABLE.boost, x)), d.yte);
    if ($('tbStatTrain')) $('tbStatTrain').textContent = `${(tr.acc * 100).toFixed(1)}% / ${tr.loss.toFixed(3)}`;
    if ($('tbStatTest'))  $('tbStatTest').textContent  = d.Xte.length ? `${(te.acc * 100).toFixed(1)}% / ${te.loss.toFixed(3)}` : '—';
  } else {
    if ($('tbStatTrain')) $('tbStatTrain').textContent = '—';
    if ($('tbStatTest'))  $('tbStatTest').textContent  = '—';
  }
}
function getTableSamples() {
  const d = imputeTable();
  const samples = [];
  for (let i = 0; i < TABLE.X.length; i++) {
    if (isNaN(TABLE.y[i])) continue;
    const row = [];
    for (let j = 0; j < TABLE.feats; j++) {
      row.push(isNaN(TABLE.X[i][j]) ? d.medians[j] : TABLE.X[i][j]);
    }
    samples.push({ origIdx: i, x: row, y: TABLE.y[i], isTest: TABLE.isTest[i] });
  }
  return { samples, ...d };
}
function trainForestOnTable() {
  stopTablePlay();
  const d = imputeTable();
  if (d.Xtr.length < 4) { alert('Слишком мало валидных Train-строк для обучения'); return; }
  const nFeat = Math.min(TABLE.feats, Math.max(1, TABLE.params.nFeaturesF));
  TABLE.forest = buildForest(d.Xtr, d.ytr, {
    nTrees: TABLE.params.nTrees,
    maxDepth: TABLE.params.maxDepthF,
    nFeatures: nFeat,
    sampleFrac: TABLE.params.sampleFracF,
  });
  TABLE.forestStep = TABLE.forest.length;
  updateTableModelStats();
  renderTableCharts();
  renderTableForestResults();
}

function trainBoostOnTable() {
  stopTablePlay();
  const d = imputeTable();
  if (d.Xtr.length < 4) { alert('Слишком мало валидных Train-строк для обучения'); return; }
  if (!TABLE.boost) TABLE.boost = initBoost(d.Xtr, d.ytr, TABLE.params.lr);
  if (TABLE.boost.trees.length >= TABLE.params.nIters) return;
  TABLE.boost = boostStep(TABLE.boost, d.Xtr, d.ytr, { maxDepth: TABLE.params.maxDepthB });
  updateTableModelStats();
  renderTableCharts();
  renderTableBoostResults();
}


function renderTableForestResults() {
  const head = $('tfResHead'), body = $('tfResBody');
  if (!head || !body) return;
  const { samples } = getTableSamples();
  const f = TABLE.feats;
  const activeTrees = getActiveTableForest();

  head.innerHTML = '<tr><th>#</th><th>Выборка</th>' +
    Array.from({ length: f }, (_, j) => `<th>x${j + 1}</th>`).join('') +
    '<th>Истина (y)</th><th>Предсказание</th><th>Вероятность p</th></tr>';
  body.innerHTML = '';

  samples.forEach((s, idx) => {
    const tr = document.createElement('tr');
    if (TABLE.selectedForestRow === idx) tr.classList.add('selected');
    let probStr = '—', predStr = '—', predCls = 'muted';
    if (activeTrees.length > 0) {
      let sum = 0;
      for (const t of activeTrees) sum += predictTree(t, s.x);
      const p = sum / activeTrees.length;
      const pred = p > 0.5 ? 1 : 0;
      probStr = p.toFixed(3);
      predStr = `${pred} (${pred === s.y ? '✓' : '✗'})`;
      predCls = pred === s.y ? 'ok' : 'bad';
    }
    tr.innerHTML = `
      <td class="muted">${s.origIdx}</td>
      <td>${s.isTest ? '<span class="split split--test">Test</span>' : '<span class="split split--train">Train</span>'}</td>
      ${s.x.map(v => `<td>${v.toFixed(3)}</td>`).join('')}
      <td class="${s.y === 1 ? 'y1' : 'y0'}">${s.y}</td>
      <td class="${predCls}"><b>${predStr}</b></td>
      <td class="muted">${probStr}</td>
    `;
    tr.onclick = () => {
      if (!activeTrees.length) return;
      TABLE.selectedForestRow = idx;
      renderTableForestResults();
      setTitle(`🌲 Лес (табл.) · образец #${s.origIdx}`);
      openTree(activeTrees, 0, s.x);
    };
    body.appendChild(tr);
  });
  if ($('tfRowCount')) $('tfRowCount').textContent = `${samples.length} образцов`;
}


function renderTableBoostResults() {
  const head = $('tbResHead'), body = $('tbResBody');
  if (!head || !body) return;
  const { samples } = getTableSamples();
  const f = TABLE.feats;
  head.innerHTML = '<tr><th>#</th><th>Выборка</th>' +
    Array.from({ length: f }, (_, j) => `<th>x${j + 1}</th>`).join('') +
    '<th>Истина (y)</th><th>Предсказание</th><th>Вероятность p</th></tr>';
  body.innerHTML = '';

  samples.forEach((s, idx) => {
    const tr = document.createElement('tr');
    if (TABLE.selectedBoostRow === idx) tr.classList.add('selected');
    let probStr = '—', predStr = '—', predCls = 'muted';
    if (TABLE.boost) {
      const p = boostPredict(TABLE.boost, s.x);
      const pred = p > 0.5 ? 1 : 0;
      probStr = p.toFixed(3);
      predStr = `${pred} (${pred === s.y ? '✓' : '✗'})`;
      predCls = pred === s.y ? 'ok' : 'bad';
    }
    tr.innerHTML = `
      <td class="muted">${s.origIdx}</td>
      <td>${s.isTest ? '<span class="split split--test">Test</span>' : '<span class="split split--train">Train</span>'}</td>
      ${s.x.map(v => `<td>${v.toFixed(3)}</td>`).join('')}
      <td class="${s.y === 1 ? 'y1' : 'y0'}">${s.y}</td>
      <td class="${predCls}"><b>${predStr}</b></td>
      <td class="muted">${probStr}</td>
    `;
    tr.onclick = () => {
      if (!TABLE.boost) return;
      TABLE.selectedBoostRow = idx;
      renderTableBoostResults();
      setTitle(`🚀 Бустинг (табл.) · образец #${s.origIdx}`);
      openTree(TABLE.boost.trees, 0, s.x);
    };
    body.appendChild(tr);
  });
  if ($('tbRowCount')) $('tbRowCount').textContent = `${samples.length} образцов`;
}

if ($('btnGenTable')) $('btnGenTable').onclick = genTable;
if ($('btnAddTableRow')) $('btnAddTableRow').onclick = () => {
  if (!TABLE.X.length) genTable();
  const row = Array.from({ length: TABLE.feats }, () => +(Math.random() * 2 - 1).toFixed(3));
  const splitFrac = parseFloat(String($('tTestSplit')?.value || '0.25').replace(',', '.')) || 0.25;
  TABLE.X.push(row);
  TABLE.y.push(Math.random() < 0.5 ? 0 : 1);
  TABLE.isTest.push(Math.random() < splitFrac);
  markTableDirty();
  renderTableTab();
};
if ($('btnExportTable')) $('btnExportTable').onclick = () => {
  if (!TABLE.X.length) return;
  const head = Array.from({ length: TABLE.feats }, (_, j) => `x${j + 1}`).join(',') + ',y,split';
  const lines = [head];
  for (let i = 0; i < TABLE.X.length; i++) {
    const row = TABLE.X[i].map(v => isNaN(v) ? '' : v);
    lines.push(row.join(',') + ',' + (isNaN(TABLE.y[i]) ? '' : TABLE.y[i]) + ',' + (TABLE.isTest[i] ? 'test' : 'train'));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'table-data.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
if ($('tTestSplit')) $('tTestSplit').onchange = () => {
  if (TABLE.X.length) { splitTableTrainTest(); markTableDirty(); renderTableTab(); }
};
if ($('btnTrainTableForest')) $('btnTrainTableForest').onclick = trainForestOnTable;
if ($('btnResetTableForest')) $('btnResetTableForest').onclick = () => {
  stopTablePlay();
  TABLE.forest = null; TABLE.forestStep = 0; TABLE.selectedForestRow = null;
  updateTableModelStats();
  renderTableCharts();
  renderTableForestResults();
};

if ($('btnTrainTableBoost')) $('btnTrainTableBoost').onclick = trainBoostOnTable; // <-- ВСТАВИТЬ СЮДА
if ($('btnResetTableBoost')) $('btnResetTableBoost').onclick = () => {
  stopTablePlay();
  TABLE.boost = null; TABLE.selectedBoostRow = null;
  updateTableModelStats();
  renderTableCharts();
  renderTableBoostResults();
};


// Истинная вероятность P(y=1 | x) для текущей формы границы и шума
function trueProb(shape, x1, x2, noise) {
  let clean = 0;

  if (shape === 'circle') {
    clean = (x1 * x1 + x2 * x2 < 0.5) ? 1 : 0;
  } else if (shape === 'xor') {
    clean = (x1 * x2 > 0) ? 1 : 0;
  } else if (shape === 'spiral') {
    let bestD = Infinity;
    let bestCls = 0;
    for (let t = 0; t <= 3 * Math.PI; t += 0.15) {
      const rt = 0.1 + (t / (3 * Math.PI)) * 0.9;
      for (let cls = 0; cls <= 1; cls++) {
        const a = t + (cls ? Math.PI : 0);
        const dx = x1 - rt * Math.cos(a);
        const dy = x2 - rt * Math.sin(a);
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          bestCls = cls;
        }
      }
    }
    clean = bestCls;
  } else {
    let d0 = Infinity;
    let d1 = Infinity;
    for (let t = 0; t <= Math.PI; t += 0.1) {
      const dx0 = x1 - (Math.cos(t) - 0.5);
      const dy0 = x2 - (Math.sin(t) - 0.2);
      d0 = Math.min(d0, dx0 * dx0 + dy0 * dy0);

      const dx1 = x1 - (Math.cos(t + Math.PI) + 0.5);
      const dy1 = x2 - (Math.sin(t + Math.PI) + 0.2);
      d1 = Math.min(d1, dx1 * dx1 + dy1 * dy1);
    }
    clean = d1 < d0 ? 1 : 0;
  }

  return clean === 1 ? (1 - noise) : noise;
}


function runBvdAnalysis() {
  const btn = $('btnCalcBvd');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Считаем...'; }
  setTimeout(() => {
    compute2dBvdData();
    // Если модели ещё не были запущены, показываем кривые для текущего или полного шага
    if (!state.forest && !state.boost) {
      drawBvdChart(ctxChartFBias, 380, 115, [{ label: 'Bias²', values: state.bvdData.fBias, color: SERIES.forest }], { xLabel: 'число деревьев', title: 'Лес · Bias²' });
      drawBvdChart(ctxChartFVar,  380, 115, [{ label: 'Var',   values: state.bvdData.fVar,  color: SERIES.forestVar }], { xLabel: 'число деревьев', title: 'Лес · Variance' });
      drawBvdChart(ctxChartBBias, 380, 115, [{ label: 'Bias²', values: state.bvdData.bBias, color: SERIES.boost }], { xLabel: 'итерация', title: 'Бустинг · Bias²' });
      drawBvdChart(ctxChartBVar,  380, 115, [{ label: 'Var',   values: state.bvdData.bVar,  color: SERIES.boostVar }], { xLabel: 'итерация', title: 'Бустинг · Variance' });
    } else {
      renderCharts();
    }
    if (btn) { btn.disabled = false; btn.textContent = '📊 Оценить Bias² и Variance'; }
  }, 15);
}

if ($('btnCalcBvd')) $('btnCalcBvd').onclick = runBvdAnalysis;
function renderTableSummary() {
  const tbody = $('tSummaryBody');
  const balanceEl = $('tClassBalance');
  const badgeEl = $('statsBadge');
  if (!tbody) return;

  const n = TABLE.X.length;
  const f = TABLE.feats;
  tbody.innerHTML = '';
  if (badgeEl) badgeEl.textContent = `${n} строк · ${f} призн.`;

  if (!n) {
    if (balanceEl) balanceEl.textContent = 'Нет данных';
    tbody.innerHTML = `<tr><td colspan="8" class="muted" style="text-align:center;padding:16px">Сначала сгенерируйте таблицу</td></tr>`;
    return;
  }

  let c0 = 0, c1 = 0, yMiss = 0;
  for (let i = 0; i < n; i++) {
    if (isNaN(TABLE.y[i])) yMiss++;
    else if (TABLE.y[i] === 1) c1++;
    else c0++;
  }
  const yValid = c0 + c1;
  if (balanceEl) {
    const p1 = yValid ? ((c1 / yValid) * 100).toFixed(1) : 0;
    const p0 = yValid ? ((c0 / yValid) * 100).toFixed(1) : 0;
    balanceEl.innerHTML = `Классы: <b class="cls-1">1 → ${c1} (${p1}%)</b> · <b class="cls-0">0 → ${c0} (${p0}%)</b>${yMiss ? ` · NaN: <b>${yMiss}</b>` : ''}`;
  }

  for (let j = 0; j < f; j++) {
    const vals = [];
    let miss = 0;
    for (let i = 0; i < n; i++) {
      const v = TABLE.X[i][j];
      if (isNaN(v)) miss++;
      else vals.push(v);
    }

    vals.sort((a, b) => a - b);
    const cnt = vals.length;
    const missPct = ((miss / n) * 100).toFixed(1);

    let meanStr = '—', stdStr = '—', minStr = '—', medStr = '—', maxStr = '—';
    if (cnt > 0) {
      const mean = vals.reduce((a, b) => a + b, 0) / cnt;
      const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / cnt;
      const std = Math.sqrt(variance);
      const min = vals[0];
      const max = vals[cnt - 1];
      const med = cnt % 2 === 1
        ? vals[(cnt - 1) / 2]
        : (vals[cnt / 2 - 1] + vals[cnt / 2]) / 2;

      meanStr = mean.toFixed(3);
      stdStr  = std.toFixed(3);
      minStr  = min.toFixed(3);
      medStr  = med.toFixed(3);
      maxStr  = max.toFixed(3);
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>x${j + 1}</b></td>
      <td>${cnt}</td>
      <td class="${miss > 0 ? 'bad' : 'muted'}">${miss} (${missPct}%)</td>
      <td>${meanStr}</td>
      <td>${stdStr}</td>
      <td>${minStr}</td>
      <td><b>${medStr}</b></td>
      <td>${maxStr}</td>
    `;
    tbody.appendChild(tr);
  }
}

(function initStatsDrawer() {
  const drawer = $('statsDrawer');
  const overlay = $('statsOverlay');
  const closeBtn = $('statsClose');
  const openBtn = $('btnOpenStats');
  if (!drawer || !overlay || !openBtn) return;

  const close = () => {
    drawer.classList.remove('open');
    overlay.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
  };
  const open = () => {
    if (!TABLE.X.length) genTable();
    renderTableSummary();
    drawer.classList.add('open');
    overlay.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
  };

  openBtn.onclick = open;
  if (closeBtn) closeBtn.onclick = close;
  overlay.onclick = close;
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && drawer.classList.contains('open')) close();
  });
})();
// ============================================================
// ПОШАГОВОЕ ПРОИГРЫВАНИЕ (PLAY) ДЛЯ ТАБЛИЧНЫХ МОДЕЛЕЙ
// ============================================================
let timerTableF = null;
let timerTableB = null;

function stopTablePlay() {
  if (timerTableF) { clearInterval(timerTableF); timerTableF = null; }
  if (timerTableB) { clearInterval(timerTableB); timerTableB = null; }
  if ($('btnPlayTableForest')) $('btnPlayTableForest').textContent = '▶';
  if ($('btnPlayTableBoost'))  $('btnPlayTableBoost').textContent = '▶';
}

function playTableForest() {
  if (timerTableF) { stopTablePlay(); return; }
  stopTablePlay();

  const d = imputeTable();
  if (d.Xtr.length < 4) { alert('Слишком мало валидных Train-строк для обучения'); return; }

  if (!TABLE.forest || TABLE.forestStep >= TABLE.forest.length) {
    const nFeat = Math.min(TABLE.feats, Math.max(1, TABLE.params.nFeaturesF));
    TABLE.forest = buildForest(d.Xtr, d.ytr, {
      nTrees: TABLE.params.nTrees,
      maxDepth: TABLE.params.maxDepthF,
      nFeatures: nFeat,
      sampleFrac: TABLE.params.sampleFracF,
    });
    TABLE.forestStep = 0;
  }

  $('btnPlayTableForest').textContent = '⏸';
  timerTableF = setInterval(() => {
    if (TABLE.forestStep >= TABLE.forest.length) {
      stopTablePlay();
      return;
    }
    TABLE.forestStep++;
    updateTableModelStats();
    renderTableCharts();
    renderTableForestResults();
  }, 450);
}

function playTableBoost() {
  if (timerTableB) { stopTablePlay(); return; }
  stopTablePlay();

  const d = imputeTable();
  if (d.Xtr.length < 4) { alert('Слишком мало валидных Train-строк для обучения'); return; }

  // Если бустинг уже дошёл до максимума итераций — сбрасываем для нового прогона
  if (TABLE.boost && TABLE.boost.trees.length >= TABLE.params.nIters) {
    TABLE.boost = null;
  }

  $('btnPlayTableBoost').textContent = '⏸';
  timerTableB = setInterval(() => {
    if (!TABLE.boost) {
      TABLE.boost = initBoost(d.Xtr, d.ytr, TABLE.params.lr);
    }
    if (TABLE.boost.trees.length >= TABLE.params.nIters) {
      stopTablePlay();
      return;
    }
    TABLE.boost = boostStep(TABLE.boost, d.Xtr, d.ytr, { maxDepth: TABLE.params.maxDepthB });
    if (typeof updateTableModelStats === 'function') updateTableModelStats();
    if (typeof renderTableCharts === 'function') renderTableCharts();
    renderTableBoostResults();
  }, 450);
}

// Переназначаем клики на кнопки Play и Reset
if ($('btnPlayTableForest')) $('btnPlayTableForest').onclick = playTableForest;
if ($('btnPlayTableBoost'))  $('btnPlayTableBoost').onclick  = playTableBoost;
// ============================================================
// ВЫЕЗЖАЮЩАЯ ПАНЕЛЬ ГРАФИКОВ ДЛЯ ТАБЛИЧНЫХ МОДЕЛЕЙ
// ============================================================
let activeChartsMode = null; // 'forest' | 'boost'

function drawActiveChartsDrawer(recalcBvd = false) {
  const drawer = $('chartsDrawer');
  if (!drawer || !drawer.classList.contains('open') || !activeChartsMode) return;

  const d = imputeTable();
  const cTrL = getCtx('drwLossTrain'), cTeL = getCtx('drwLossTest');
  const cTrA = getCtx('drwAccTrain'),  cTeA = getCtx('drwAccTest');
  const cBias = getCtx('drwBias'),     cVar = getCtx('drwVar');

  const isForest = activeChartsMode === 'forest';
  const curStep = isForest
    ? getActiveTableForest().length
    : (TABLE.boost ? TABLE.boost.trees.length : 0);

  if (isForest) {
    $('chartsDrawerTitle').textContent = '🌲 Графики: Random Forest (табл.)';
    $('chartsDrawerBadge').textContent = `${curStep} / ${TABLE.forest?.length || TABLE.params.nTrees} дер.`;

    if (curStep > 0) {
      const activeTrees = getActiveTableForest();
      const trLoss = [], teLoss = [], trAcc = [], teAcc = [];
      for (let k = 1; k <= curStep; k++) {
        const sub = activeTrees.slice(0, k);
        const mTr = evalMetrics(forestPreds(sub, d.Xtr), d.ytr);
        trLoss.push(mTr.loss); trAcc.push(mTr.acc);
        if (d.Xte.length) {
          const mTe = evalMetrics(forestPreds(sub, d.Xte), d.yte);
          teLoss.push(mTe.loss); teAcc.push(mTe.acc);
        }
      }
      drawLossChart(cTrL, 330, 115, trLoss, SERIES.forest, { xLabel: 'деревьев', yLabel: 'loss', title: 'Train Loss' });
      drawLossChart(cTeL, 330, 115, teLoss, SERIES.forestTest, { xLabel: 'деревьев', yLabel: 'loss', title: 'Test Loss' });
      drawLossChart(cTrA, 330, 115, trAcc,  SERIES.ok, { xLabel: 'деревьев', yLabel: 'acc',  title: 'Train Accuracy' });
      drawLossChart(cTeA, 330, 115, teAcc,  SERIES.okDeep, { xLabel: 'деревьев', yLabel: 'acc',  title: 'Test Accuracy' });
    } else {
      drawLossChart(cTrL, 330, 115, [], SERIES.forest, { title: 'Train Loss' });
      drawLossChart(cTeL, 330, 115, [], SERIES.forestTest, { title: 'Test Loss' });
      drawLossChart(cTrA, 330, 115, [], SERIES.ok, { title: 'Train Accuracy' });
      drawLossChart(cTeA, 330, 115, [], SERIES.okDeep, { title: 'Test Accuracy' });
    }
  } else {
    $('chartsDrawerTitle').textContent = '🚀 Графики: Gradient Boosting (табл.)';
    $('chartsDrawerBadge').textContent = `${curStep} / ${TABLE.params.nIters} итер.`;

    if (curStep > 0) {
      const b = TABLE.boost;
      const trLoss = b.history.map(h => h.loss);
      const trAcc  = b.history.map(h => h.acc);
      const teLoss = [], teAcc = [];

      if (d.Xte.length) {
        let f = d.Xte.map(() => b.F0);
        teLoss.push(logLoss(d.yte, f));
        teAcc.push(accuracy(f, d.yte));
        for (let k = 0; k < b.trees.length; k++) {
          f = f.map((val, idx) => val + b.lr * predictTree(b.trees[k], d.Xte[idx]));
          teLoss.push(logLoss(d.yte, f));
          teAcc.push(accuracy(f, d.yte));
        }
      }
      drawLossChart(cTrL, 330, 115, trLoss, SERIES.boost, { xLabel: 'итераций', yLabel: 'loss', title: 'Train Loss' });
      drawLossChart(cTeL, 330, 115, teLoss, SERIES.boostTest, { xLabel: 'итераций', yLabel: 'loss', title: 'Test Loss' });
      drawLossChart(cTrA, 330, 115, trAcc,  SERIES.ok, { xLabel: 'итераций', yLabel: 'acc',  title: 'Train Accuracy' });
      drawLossChart(cTeA, 330, 115, teAcc,  SERIES.okDeep, { xLabel: 'итераций', yLabel: 'acc',  title: 'Test Accuracy' });
    } else {
      drawLossChart(cTrL, 330, 115, [], SERIES.boost, { title: 'Train Loss' });
      drawLossChart(cTeL, 330, 115, [], SERIES.boostTest, { title: 'Test Loss' });
      drawLossChart(cTrA, 330, 115, [], SERIES.ok, { title: 'Train Accuracy' });
      drawLossChart(cTeA, 330, 115, [], SERIES.okDeep, { title: 'Test Accuracy' });
    }
  }

  // Если BVD ещё не посчитан или запрошен пересчёт — вычисляем и кэшируем
  if (recalcBvd || (isForest && !TABLE.bvdForest) || (!isForest && !TABLE.bvdBoost)) {
    const res = computeTabularBvd(activeChartsMode);
    if (isForest) TABLE.bvdForest = res;
    else TABLE.bvdBoost = res;
  }

  const cached = isForest ? TABLE.bvdForest : TABLE.bvdBoost;
  const colB = isForest ? SERIES.forest : SERIES.boost;
  const colV = isForest ? SERIES.forestVar : SERIES.boostVar;
  const xLbl = isForest ? 'деревьев' : 'итераций';

  // Отрисовываем срез Bias² и Variance строго до текущего шага (или весь, если модель ещё не запущена)
  const sliceLen = curStep > 0 ? curStep : (cached ? cached.bias2Arr.length : 0);
  const bVals = (cached && sliceLen > 0) ? cached.bias2Arr.slice(0, sliceLen) : [];
  const vVals = (cached && sliceLen > 0) ? cached.varArr.slice(0, sliceLen)   : [];

  drawBvdChart(cBias, 330, 115, bVals.length ? [{ label: 'Bias²', values: bVals, color: colB }] : [], { xLabel: xLbl, title: 'Bias²' });
  drawBvdChart(cVar,  330, 115, vVals.length ? [{ label: 'Var',   values: vVals, color: colV }] : [], { xLabel: xLbl, title: 'Variance' });
}

(function initChartsDrawer() {
  const drawer = $('chartsDrawer');
  const overlay = $('chartsOverlay');
  const closeBtn = $('chartsClose');
  if (!drawer || !overlay) return;

  const close = () => {
    drawer.classList.remove('open');
    overlay.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
  };
  const open = (mode) => {
    activeChartsMode = mode;
    drawer.classList.add('open');
    overlay.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    drawActiveChartsDrawer(true);
  };

  if ($('btnChartsTableForest')) $('btnChartsTableForest').onclick = () => open('forest');
  if ($('btnChartsTableBoost'))  $('btnChartsTableBoost').onclick  = () => open('boost');
  if ($('btnRecalcDrawerBvd'))   $('btnRecalcDrawerBvd').onclick   = () => drawActiveChartsDrawer(true);
  if (closeBtn) closeBtn.onclick = close;
  overlay.onclick = close;
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && drawer.classList.contains('open')) close();
  });
})();
function render2dDataSummary() {
  const tbody = $('tSummaryBody');
  const balanceEl = $('tClassBalance');
  const badgeEl = $('statsBadge');
  if (!tbody) return;

  const n = state.X.length;
  tbody.innerHTML = '';
  if (badgeEl) badgeEl.textContent = `${n} точек · 2D`;

  if (!n) {
    if (balanceEl) balanceEl.textContent = 'Нет данных';
    tbody.innerHTML = `<tr><td colspan="8" class="muted" style="text-align:center;padding:16px">Нет данных</td></tr>`;
    return;
  }

  // Баланс классов и Train/Test
  let c0 = 0, c1 = 0, nTest = 0;
  for (let i = 0; i < n; i++) {
    if (state.y[i] === 1) c1++; else c0++;
    if (state.isTest[i]) nTest++;
  }
  if (balanceEl) {
    const p1 = ((c1 / n) * 100).toFixed(1);
    const p0 = ((c0 / n) * 100).toFixed(1);
    balanceEl.innerHTML = `Классы: <b class="cls-1">1 → ${c1} (${p1}%)</b> · <b class="cls-0">0 → ${c0} (${p0}%)</b> · Train/Test: <b>${n - nTest}/${nTest}</b>`;
  }

  // Метрики для x1 и x2
  for (let j = 0; j < 2; j++) {
    const vals = state.X.map(row => row[j]).filter(v => !isNaN(v)).sort((a, b) => a - b);
    const cnt = vals.length;
    const miss = n - cnt;
    const missPct = ((miss / n) * 100).toFixed(1);

    let meanStr = '—', stdStr = '—', minStr = '—', medStr = '—', maxStr = '—';
    if (cnt > 0) {
      const mean = vals.reduce((a, b) => a + b, 0) / cnt;
      const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / cnt;
      const std = Math.sqrt(variance);
      const min = vals[0];
      const max = vals[cnt - 1];
      const med = cnt % 2 === 1
        ? vals[(cnt - 1) / 2]
        : (vals[cnt / 2 - 1] + vals[cnt / 2]) / 2;

      meanStr = mean.toFixed(3);
      stdStr  = std.toFixed(3);
      minStr  = min.toFixed(3);
      medStr  = med.toFixed(3);
      maxStr  = max.toFixed(3);
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>x${j + 1}</b></td>
      <td>${cnt}</td>
      <td class="${miss > 0 ? 'bad' : 'muted'}">${miss} (${missPct}%)</td>
      <td>${meanStr}</td>
      <td>${stdStr}</td>
      <td>${minStr}</td>
      <td><b>${medStr}</b></td>
      <td>${maxStr}</td>
    `;
    tbody.appendChild(tr);
  }
}

if ($('btnOpenDataStats')) {
  $('btnOpenDataStats').onclick = () => {
    const drawer = $('statsDrawer');
    const overlay = $('statsOverlay');
    if (!drawer || !overlay) return;
    render2dDataSummary();
    drawer.classList.add('open');
    overlay.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
  };
}
