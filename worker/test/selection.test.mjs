/**
 * 買い目選定のテスト。
 *
 * 実運用で「期待値9.65」というあり得ない買い目が推奨された事故があったので、
 * その再発を防ぐためのテストを厚めに置いている。
 * 控除率25%の市場で期待値が2倍を超えることは、まず起きない。
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { parseOdds3t } from "../src/scrape.ts";
import {
  buildPicks,
  selectPicks,
  marketProbabilities,
  DEFAULT_OPTIONS,
} from "../src/selection.ts";

const here = dirname(fileURLToPath(import.meta.url));
const odds = parseOdds3t(readFileSync(join(here, "fixtures", "odds3t.html"), "utf-8"));

/** 適当なモデル確率（合計1）を作る。 */
function uniformCombos() {
  const combos = Object.keys(odds).map((combo) => ({ combo, prob: 1 / 120 }));
  return combos;
}

test("市場の暗黙確率は合計1になる", () => {
  const m = marketProbabilities(odds);
  const sum = Object.values(m).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `合計が ${sum}`);
});

test("ブレンド後の確率も合計1になる", () => {
  const picks = buildPicks(uniformCombos(), odds);
  const sum = picks.reduce((a, p) => a + p.prob, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `合計が ${sum}`);
});

test("モデルが穴目を極端に高く見ても、推奨に暴走した期待値が出ない", () => {
  // 実際に起きた事故の再現:
  // 市場が 946.5倍（暗黙確率0.08%）と見ている買い目を、モデルが1.0%と主張する
  const combos = Object.keys(odds).map((combo) => ({ combo, prob: 0 }));
  const longshot = Object.entries(odds).sort((a, b) => b[1] - a[1])[0][0];

  // 最も人気のない買い目に大きな確率を振り、残りを均等に配る
  for (const c of combos) {
    c.prob = c.combo === longshot ? 0.05 : 0.95 / 119;
  }

  const picks = buildPicks(combos, odds);
  const selected = selectPicks(picks);

  for (const p of selected) {
    assert.ok(
      p.ev <= DEFAULT_OPTIONS.evCeiling,
      `${p.combo} の期待値が ${p.ev.toFixed(2)}。控除率25%の市場では起きえない値です`,
    );
  }

  // 暴走した買い目そのものが推奨に入っていないこと
  assert.ok(
    !selected.some((p) => p.combo === longshot),
    `モデルが市場の何十倍も強気な買い目 ${longshot} が推奨に混ざっています`,
  );
});

test("モデルが市場と同じ見立てなら、期待値はどれも控除率ぶん1未満になる", () => {
  const market = marketProbabilities(odds);
  const combos = Object.entries(market).map(([combo, prob]) => ({ combo, prob }));

  const picks = buildPicks(combos, odds);
  for (const p of picks) {
    assert.ok(
      p.ev < 1.0,
      `${p.combo} の期待値が ${p.ev.toFixed(3)}。市場と同じ確率なら必ず1未満のはずです`,
    );
  }
  // したがって推奨は0点（見送り）になる
  assert.equal(selectPicks(picks).length, 0);
});

test("推奨は確率の高い順に並ぶ（期待値順にすると万券ばかりになる）", () => {
  const market = marketProbabilities(odds);
  // 市場よりわずかに強気な見立てを作る
  const combos = Object.entries(market).map(([combo, q]) => ({
    combo,
    prob: q * (1 + 0.3 * Math.sin(combo.charCodeAt(0) + combo.charCodeAt(2))),
  }));
  const z = combos.reduce((a, c) => a + c.prob, 0);
  for (const c of combos) c.prob /= z;

  const selected = selectPicks(buildPicks(combos, odds));
  for (let i = 1; i < selected.length; i++) {
    assert.ok(
      selected[i - 1].prob >= selected[i].prob,
      "確率の降順になっていません",
    );
  }
});

test("確率が低すぎる買い目は推奨に入らない", () => {
  const picks = buildPicks(uniformCombos(), odds);
  for (const p of selectPicks(picks)) {
    assert.ok(
      p.prob >= DEFAULT_OPTIONS.minProb,
      `${p.combo} の確率が ${(p.prob * 100).toFixed(2)}% しかありません`,
    );
  }
});

test("オッズが取れていないときはモデル確率のまま返り、期待値はnullになる", () => {
  const picks = buildPicks(uniformCombos(), {});
  assert.equal(picks.length, 120);
  for (const p of picks) {
    assert.equal(p.ev, null);
    assert.equal(p.marketProb, null);
    assert.equal(p.prob, p.modelProb);
  }
  // 期待値が計算できない以上、推奨は出さない
  assert.equal(selectPicks(picks).length, 0);
});

test("オッズが一部欠けていても確率の合計は1になり、欠けた組は候補から外れる", () => {
  // 実際に起きうる状況: 120通り中10通りだけオッズが取れなかった
  const partial = { ...odds };
  const removed = Object.keys(partial).slice(0, 10);
  for (const k of removed) delete partial[k];
  assert.equal(Object.keys(partial).length, 110);

  const picks = buildPicks(uniformCombos(), partial);
  assert.equal(picks.length, 110, "オッズの無い組は候補に含めない");

  const sum = picks.reduce((a, p) => a + p.prob, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `確率の合計が ${sum}`);

  // 以前の実装はオッズの無い組へ「正規化前のモデル生確率」をそのまま入れており、
  // 穴目を過大評価する形で合計が1を超えていた
  for (const p of picks) {
    assert.ok(p.marketProb !== null, `${p.combo} に市場確率がありません`);
  }

  // modelProb / marketProb は同じ母集団（オッズのある110通り）で正規化されて
  // いること。片方だけ生のままだと maxProbRatio フィルタが甘くなる
  const mSum = picks.reduce((a, p) => a + p.modelProb, 0);
  const qSum = picks.reduce((a, p) => a + p.marketProb, 0);
  assert.ok(Math.abs(mSum - 1) < 1e-9, `モデル確率の合計が ${mSum}（正規化されていない）`);
  assert.ok(Math.abs(qSum - 1) < 1e-9, `市場確率の合計が ${qSum}`);
});
