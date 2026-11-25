// ============================================================
// tree-drawer.js — боковая панель с диаграммой дерева
// ============================================================

import { token, font, classColor, classInk } from './palette.js';

let ctx, canvas;
let currentNode = null;
let currentPath = null;
let layout = null;
let hoverNode = null;

let treesList = [];
let currentIndex = 0;
let titlePrefix = 'Дерево';
let selectedX = null;

const tooltip = () => document.getElementById('nodeTooltip');

function computeLayout(root, canvasW) {
  const nodes = [];
  const edges = [];
  const levelGap = 92;
  const nodeR = 22;

  function countLeaves(n) {
    if (!n.left) return 1;
    n._leaves = countLeaves(n.left) + countLeaves(n.right);
    return n._leaves;
  }
  countLeaves(root);

  function place(n, depth, xMin, xMax) {
    const x = (xMin + xMax) / 2;
    const y = 40 + depth * levelGap;
    const rec = { node: n, x, y, depth };
    nodes.push(rec);
    n._rec = rec;

    if (n.left) {
      const totalLeaves = n._leaves;
      const leftLeaves = n.left._leaves;
      const splitX = xMin + (xMax - xMin) * (leftLeaves / totalLeaves);
      place(n.left, depth + 1, xMin, splitX);
      place(n.right, depth + 1, splitX, xMax);
      edges.push({ from: n._rec, to: n.left._rec });
      edges.push({ from: n._rec, to: n.right._rec });
    }
  }
  place(root, 0, 0, canvasW);
  return { nodes, edges, nodeR };
}

function pathFor(node, x) {
  const path = [];
  let cur = node;
  while (cur) {
    path.push(cur);
    if (!cur.left) break;
    cur = (x[cur.feat] <= cur.thresh) ? cur.left : cur.right;
  }
  return new Set(path);
}

export function renderTree(root, opts = {}) {
  canvas = document.getElementById('treeCanvas');
  ctx = canvas.getContext('2d');
  currentNode = root;

  if (!root) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = token('--nn-surface');
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = token('--nn-text-muted');
    ctx.font = font(14);
    ctx.textAlign = 'center';
    ctx.fillText('Дерево пустое', canvas.width / 2, canvas.height / 2);
    return;
  }

  layout = computeLayout(root, canvas.width);

  const maxDepth = Math.max(...layout.nodes.map(n => n.depth));
  const neededH = 80 + maxDepth * 92;
  if (canvas.height < neededH) canvas.height = neededH;

  draw();

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    const hit = layout.nodes.find(n =>
      (n.x - mx) ** 2 + (n.y - my) ** 2 < layout.nodeR ** 2
    );
    hoverNode = hit ? hit.node : null;
    if (hit) showTooltip(hit.node, e.clientX, e.clientY);
    else hideTooltip();
    canvas.style.cursor = hit ? 'pointer' : 'default';
    draw();
  };
  canvas.onmouseleave = () => { hoverNode = null; hideTooltip(); draw(); };
}

export function setSelectedPoint(x) {
  selectedX = x || null;
  if (!currentNode) return;
  currentPath = x ? pathFor(currentNode, x) : null;
  if (layout) draw();
}

function draw() {
  if (!layout) return;
  const { nodes, edges, nodeR } = layout;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = token('--nn-surface');
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (const e of edges) {
    const a = e.from, b = e.to;
    const onPath = currentPath && currentPath.has(a.node) && currentPath.has(b.node);
    ctx.strokeStyle = onPath ? token('--nn-accent') : token('--nn-border-strong');
    ctx.lineWidth = onPath ? 3 : 1.5;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const n of nodes) {
    const node = n.node;
    const onPath = currentPath && currentPath.has(node);
    const isHover = hoverNode === node;
    drawNode(node, n.x, n.y, nodeR, { onPath, isHover });
  }
}

function drawNode(node, x, y, r, opts = {}) {
  const { onPath, isHover } = opts;
  const isLeaf = !node.left;
  const stroke = onPath ? token('--nn-accent') : (isHover ? token('--nn-text') : null);

  if (isLeaf) {
    const p = node.pred;
    ctx.fillStyle = classColor(p);
    ctx.strokeStyle = stroke ?? token('--nn-border-strong');
    ctx.lineWidth = onPath ? 3 : (isHover ? 2.5 : 1.5);
    ctx.beginPath();
    ctx.arc(x, y, r + 6, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = classInk(p);
    ctx.font = font(11, 'bold');
    ctx.textAlign = 'center';
    ctx.fillText(`p=${Number(p).toFixed(2)}`, x, y + 3);
    ctx.font = font(9);
    ctx.globalAlpha = 0.8;
    ctx.fillText(`n=${node.n || '?'}`, x, y + 16);
    ctx.globalAlpha = 1;
  } else {
    ctx.fillStyle = isHover ? token('--nn-accent-soft') : token('--nn-surface');
    ctx.strokeStyle = stroke ?? token('--nn-text-muted');
    ctx.lineWidth = onPath ? 3 : 1.5;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = token('--nn-text');
    ctx.font = font(11, 'bold');
    ctx.textAlign = 'center';
    ctx.fillText(`x${node.feat + 1}`, x, y - 4);
    ctx.font = font(10);
    ctx.fillStyle = token('--nn-text-secondary');
    ctx.fillText(`≤ ${Number(node.thresh).toFixed(2)}`, x, y + 9);
  }
}

function showTooltip(node, cx, cy) {
  const t = tooltip();
  const isLeaf = !node.left;
  if (isLeaf) {
    t.innerHTML = `<b>Лист</b><br>p(класс 1) = <code>${Number(node.pred).toFixed(3)}</code><br>Образцов в узле: <code>${node.n || '?'}</code>`;
  } else {
    const pL = meanOfSubtree(node.left);
    const pR = meanOfSubtree(node.right);
    t.innerHTML = `<b>Сплит</b> по признаку x${node.feat + 1}<br>
      условие: <code>x${node.feat + 1} ≤ ${Number(node.thresh).toFixed(3)}</code><br>
      слева p ≈ <code>${pL.toFixed(2)}</code>, справа p ≈ <code>${pR.toFixed(2)}</code><br>
      gain = <code>${Number(node.gain || 0).toFixed(4)}</code>`;
  }
  t.style.left = (cx + 14) + 'px';
  t.style.top = (cy - 10) + 'px';
  t.classList.add('show');
}
function hideTooltip() { tooltip().classList.remove('show'); }

function meanOfSubtree(node) {
  if (!node.left) return node.pred;
  const leaves = [];
  const collect = n => { if (!n.left) leaves.push(n.pred); else { collect(n.left); collect(n.right); } };
  collect(node);
  return leaves.reduce((a, b) => a + b, 0) / leaves.length;
}

export function initDrawer() {
  const overlay = document.getElementById('drawerOverlay');
  const drawer = document.getElementById('treeDrawer');
  const closeBtn = document.getElementById('drawerClose');
  const prevBtn = document.getElementById('drawerPrev');
  const nextBtn = document.getElementById('drawerNext');

  const close = () => {
    drawer.classList.remove('open');
    overlay.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
    hideTooltip();
  };
  const open = () => {
    drawer.classList.add('open');
    overlay.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
  };

  closeBtn.onclick = close;
  overlay.onclick = close;
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
    if (!drawer.classList.contains('open')) return;
    if (e.key === 'ArrowLeft') prevBtn.click();
    if (e.key === 'ArrowRight') nextBtn.click();
  });

  prevBtn.onclick = () => {
    if (!treesList.length) return;
    currentIndex = (currentIndex - 1 + treesList.length) % treesList.length;
    updateDrawer();
  };
  nextBtn.onclick = () => {
    if (!treesList.length) return;
    currentIndex = (currentIndex + 1) % treesList.length;
    updateDrawer();
  };

  return { open, close };
}

export function openTree(trees, index, x) {
  treesList = trees;
  currentIndex = Math.max(0, Math.min(index, trees.length - 1));
  selectedX = x || null;
  updateDrawer();

  document.getElementById('treeDrawer').classList.add('open');
  document.getElementById('drawerOverlay').classList.add('open');
  document.getElementById('treeDrawer').setAttribute('aria-hidden', 'false');
}

export function updateDrawer(x) {
  if (x !== undefined) selectedX = x;
  if (!treesList.length) return;
  const tree = treesList[currentIndex];
  document.getElementById('drawerTitle').textContent = `${titlePrefix} #${currentIndex + 1}`;
  document.getElementById('drawerPos').textContent = `${currentIndex + 1} / ${treesList.length}`;

  const stats = treeStats(tree);
  document.getElementById('drawerBadge').textContent =
    `${stats.leaves} листьев · глубина ${stats.depth}`;

  renderTree(tree);
  setSelectedPoint(selectedX);
}

export function setTitle(prefix) { titlePrefix = prefix; }

function treeStats(root) {
  let leaves = 0, maxDepth = 0;
  const walk = (n, d) => {
    if (!n.left) { leaves++; maxDepth = Math.max(maxDepth, d); return; }
    walk(n.left, d + 1); walk(n.right, d + 1);
  };
  walk(root, 0);
  return { leaves, depth: maxDepth };
}