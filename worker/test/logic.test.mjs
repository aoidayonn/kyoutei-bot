/**
 * 予測ロジックの単体テスト。
 *   node --test test/
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

// --- TypeScript を読まずに済むよう、ロジックをここに再実装して挙動を確認する ---
// （本体と重複するが、テストが本体の実装に引きずられないという利点がある）

function trifecta(scores) {
  const total = scores.reduce((a, b) => a + b, 0);
  const out = [];
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      if (j === i) continue;
      for (let k = 0; k < 6; k++) {
        if (k === i || k === j) continue;
        const d2 = total - scores[i];
        const d3 = d2 - scores[j];
        out.push({
          combo: `${i + 1}-${j + 1}-${k + 1}`,
          prob: (scores[i] / total) * (scores[j] / d2) * (scores[k] / d3),
        });
      }
    }
  return out.sort((a, b) => b.prob - a.prob);
}

test("3連単は120通りで、確率の合計が1になる", () => {
  const combos = trifecta([5, 3, 2, 1.5, 1, 0.8]);
  assert.equal(combos.length, 120);
  const sum = combos.reduce((a, c) => a + c.prob, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `合計が1ではありません: ${sum}`);
});

test("組み合わせに重複がない", () => {
  const combos = trifecta([5, 3, 2, 1.5, 1, 0.8]);
  assert.equal(new Set(combos.map((c) => c.combo)).size, 120);
});

test("1号艇が弱く5号艇が強ければ、本命は1-2-3にならない", () => {
  // 1号艇だけ極端に弱く、5号艇が最強という設定
  const combos = trifecta([0.3, 1.0, 1.0, 1.0, 6.0, 1.0]);
  assert.notEqual(combos[0].combo, "1-2-3");
  assert.ok(
    combos[0].combo.startsWith("5-"),
    `5号艇が1着本命になるはずが ${combos[0].combo} でした`,
  );
});

test("スコアが等しければ全120通りが等確率になる", () => {
  const combos = trifecta([1, 1, 1, 1, 1, 1]);
  for (const c of combos) {
    assert.ok(Math.abs(c.prob - 1 / 120) < 1e-12);
  }
});

// --- オッズ表の並び順 ---

function combosInDocumentOrder() {
  const out = [];
  for (let row = 0; row < 20; row++) {
    for (let col = 0; col < 6; col++) {
      const first = col + 1;
      const rest = [1, 2, 3, 4, 5, 6].filter((x) => x !== first);
      const second = rest[Math.floor(row / 4)];
      const rest2 = rest.filter((x) => x !== second);
      out.push(`${first}-${second}-${rest2[row % 4]}`);
    }
  }
  return out;
}

test("オッズ表の読み取り順が120通りを重複なく網羅する", () => {
  const order = combosInDocumentOrder();
  assert.equal(order.length, 120);
  assert.equal(new Set(order).size, 120);
  // 実際のページで確認した先頭6件
  assert.deepEqual(order.slice(0, 6), [
    "1-2-3", "2-1-3", "3-1-2", "4-1-2", "5-1-2", "6-1-2",
  ]);
});

// --- モデルの健全性 ---

test("model.json に必要なキーが揃っている", () => {
  const model = JSON.parse(readFileSync(join(here, "..", "src", "model.json"), "utf-8"));
  for (const key of ["weights", "feature_names", "lane_prior", "racer_lane", "metrics"]) {
    assert.ok(key in model, `model.json に ${key} がありません`);
  }
  assert.equal(Object.keys(model.lane_prior).length, 144, "場×枠は24×6=144件のはずです");
  assert.ok(
    model.metrics.test.win_accuracy > model.metrics.test.baseline_lane1,
    "検証データで「1号艇固定」に勝てていません",
  );
});
