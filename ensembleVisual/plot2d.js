// ============================================================
// plot2d.js — декартова плоскость в стиле Desmos
// ============================================================

export function createView() {
  return { cx: 0, cy: 0, scale: 160 };
}

export function worldToScreen(view, size, x, y) {
  return [
    size / 2 + (x - view.cx) * view.scale,
    size / 2 - (y - view.cy) * view.scale,
  ];
}

export function screenToWorld(view, size, px, py) {
  return [
    view.cx + (px - size / 2) / view.scale,
    view.cy - (py - size / 2) / view.scale,
  ];
}

function niceStep(rawStep) {
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const base = rawStep / pow;
  let mult;
  if (base < 1.5) mult = 1;
  else if (base < 3) mult = 2;
  else if (base < 7) mult = 5;
  else mult = 10;
  return mult * pow;
}

export function drawAxes(ctx, size, view, opts = {}) {
  const { major = '#d9e2ec', minor = '#eef2f7', axis = '#94a3b8', text = '#8a99ac' } = opts;
  ctx.save();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);

  const unitsAcross = size / view.scale;
  const stepRaw = unitsAcross / 10;
  const step = niceStep(stepRaw);

  const xMin = view.cx - unitsAcross / 2;
  const xMax = view.cx + unitsAcross / 2;
  const yMin = view.cy - unitsAcross / 2;
  const yMax = view.cy + unitsAcross / 2;

  const minorStep = step / 5;
  const minorPx = minorStep * view.scale;
  if (minorPx > 6) {
    ctx.strokeStyle = minor; ctx.lineWidth = 1;
    for (let x = Math.ceil(xMin / minorStep) * minorStep; x <= xMax; x += minorStep) {
      const [px] = worldToScreen(view, size, x, 0);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, size); ctx.stroke();
    }
    for (let y = Math.ceil(yMin / minorStep) * minorStep; y <= yMax; y += minorStep) {
      const [, py] = worldToScreen(view, size, 0, y);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(size, py); ctx.stroke();
    }
  }

  ctx.strokeStyle = major; ctx.lineWidth = 1;
  for (let x = Math.ceil(xMin / step) * step; x <= xMax; x += step) {
    const [px] = worldToScreen(view, size, x, 0);
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, size); ctx.stroke();
  }
  for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
    const [, py] = worldToScreen(view, size, 0, y);
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(size, py); ctx.stroke();
  }

  const [ox, oy] = worldToScreen(view, size, 0, 0);
  ctx.strokeStyle = axis; ctx.lineWidth = 1.5;
  if (oy >= 0 && oy <= size) {
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(size, oy); ctx.stroke();
  }
  if (ox >= 0 && ox <= size) {
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, size); ctx.stroke();
  }

  ctx.fillStyle = text;
  ctx.font = '10px Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const labelY = Math.min(Math.max(oy + 4, 4), size - 14);
  for (let x = Math.ceil(xMin / step) * step; x <= xMax; x += step) {
    if (Math.abs(x) < 1e-9) continue;
    const [px] = worldToScreen(view, size, x, 0);
    if (px < 14 || px > size - 14) continue;
    ctx.fillText(fmt(x, step), px, labelY);
  }
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const labelX = Math.min(Math.max(ox - 4, 24), size - 4);
  for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
    if (Math.abs(y) < 1e-9) continue;
    const [, py] = worldToScreen(view, size, 0, y);
    if (py < 12 || py > size - 12) continue;
    ctx.fillText(fmt(y, step), labelX, py);
  }

  if (ox >= 0 && ox <= size && oy >= 0 && oy <= size) {
    ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText('0', ox - 4, oy + 3);
  }
  ctx.restore();
}

function fmt(v, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  return v.toFixed(Math.min(decimals, 4)).replace(/\.?0+$/, '') || '0';
}

export function drawProbField(ctx, size, view, predictFn) {
  ctx.save();
  const step = 4;
  for (let px = 0; px < size; px += step) {
    for (let py = 0; py < size; py += step) {
      const [x, y] = screenToWorld(view, size, px + step / 2, py + step / 2);
      const v = predictFn([x, y]);
      const r = Math.round(255 * (1 - v));
      const g = Math.round(255 * v);
      ctx.fillStyle = `rgba(${r},${g},80,0.55)`;
      ctx.fillRect(px, py, step, step);
    }
  }
  ctx.restore();
}

export function drawPointsOnPlot(ctx, size, view, X, y, opts = {}) {
  const { weights = null, selected = -1, scale = 1, isTest = null } = opts;
  const r0 = 4 * scale;
  for (let i = 0; i < X.length; i++) {
    const [px, py] = worldToScreen(view, size, X[i][0], X[i][1]);
    if (px < -20 || px > size + 20 || py < -20 || py > size + 20) continue;
    const w = weights ? weights[i] : 0;
    const r = r0 + 2.4 * w * scale;
    const isSel = i === selected;
    const testPt = isTest ? isTest[i] : false;

    const fill = isSel ? '#facc15' : (y[i] === 1 ? '#22c55e' : '#ef4444');
    const stroke = isSel ? '#a16207' : (testPt ? '#ffffff' : 'rgba(15,23,42,.55)');

    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = isSel ? 2 : (testPt ? 1.8 : 1);
    ctx.globalAlpha = testPt ? 0.85 : 0.95;

    if (y[i] === 1) {
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else {
      const s = r * 1.7;
      ctx.beginPath(); ctx.rect(px - s / 2, py - s / 2, s, s); ctx.fill(); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}
