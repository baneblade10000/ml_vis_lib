#!/usr/bin/env bash
# Build the hand-rolled CNN WASM into packages/playground/src/wasm/cnn/pkg (Vite-importable).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/crates/cnn"
OUT="$ROOT/packages/playground/src/wasm/cnn/pkg"

if ! command -v rustc >/dev/null 2>&1; then
  echo "error: rustc not found. Install Rust via https://rustup.rs" >&2
  exit 1
fi

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found — installing via cargo..."
  cargo install wasm-pack --locked
fi

rustup target add wasm32-unknown-unknown >/dev/null

mkdir -p "$OUT"
cd "$CRATE"

echo "Building cnn (wasm32, hand-rolled + WebGPU)..."
wasm-pack build \
  --target web \
  --release \
  --out-dir "$OUT" \
  --out-name cnn \
  --no-default-features \
  --features "wasm"

# wasm-pack drops a .gitignore that would hide the pkg from git — keep assets trackable.
rm -f "$OUT/.gitignore"

echo "OK → $OUT"
ls -lh "$OUT"/*.wasm "$OUT"/*.js 2>/dev/null || ls -lh "$OUT"
