---
title: "TransformerFlow constructor script: Cyrillic garbled by a Windows-1251 re-read, reversed byte-for-byte"
author: Baneblade10000
date: 2026-11-25
branch: master
id: l-cf06098a
scope: repo
status: active
src: agent
touches:
  - eslint.config.mjs
  - packages/playground/public/constructor/app.js
  - packages/playground/public/tflow/app.js
  - packages/playground/src/catalog/tree.ts
---

## Decided
- The builder of TransformerFlow showed «в†ђ РќР°Р·Р°Рґ» instead of «← Назад»: the UTF-8 bytes of the constructor script had been read as Windows-1251 and saved back as UTF-8. It was repaired by reversing exactly that (each character back to its cp1251 byte, then decode as UTF-8) so the newer content stays; an older correct copy was only used to cross-check the 46 Russian UI strings.

## Tried and ruled out
- Taking the older correctly encoded copy from a stash: it predates the 2026-11-24 grammar change, so it would drop features.
- Repairing by hand, string by string: the byte-level reversal is mechanical and provably complete.

## Watch out
- A whole-file roundtrip fails on characters that are not cp1251 artefacts (the BOM, a genuine «→»): pass those through untouched. cp1251 leaves byte 0x98 undefined and it arrives as U+0098.
- The commit «repair the garbled Cyrillic of the TransformerFlow page» fixed only the HTML; the builder's UI strings live in the script.

In the code: packages/playground/public/constructor/app.js, packages/playground/public/transformer-constructor.html
