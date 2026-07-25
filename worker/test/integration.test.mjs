/**
 * 結合テスト: 実際のHTML → 特徴量 → Plackett-Luce → 期待値 まで通す。
 *
 * 使うのは 2026-07-20 福岡1R のページ。
 * このレースの実際の結果は 1-4-2（払戻 1,530円）だった。
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { parseRacelist, parseBeforeInfo, parseOdds3t } from "../src/scrape.ts";
import { buildFeatures } from "../src/features.ts";
import { scoresFromUtilities, trifectaProbabilities, winProbabilities } from "../src/plackettLuce.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fx = (n) => readFileSync(join(here, "fixtures", n), "utf-8");
const model = JSON.parse(readFileSync(join(here, "..", "src", "model.json"), "utf-8"));
const STAGE = model.stage_exponents ?? [1, 1, 1];

function buildPrediction() {
  const base = parseRacelist(fx("racelist.html"));
  const before = parseBeforeInfo(fx("beforeinfo.html"));
  const odds = parseOdds3t(fx("odds3t.html"));

  const race = {
    jcd: 22,
    windSpeed: before.windSpeed,
    waveHeight: before.waveHeight,
    entries: base.map((e, i) => ({
      lane: i + 1,
      racerId: e.racerId,
      racerClass: e.racerClass,
      winRateNational: e.winRateNational,
      top2National: e.top2National,
      winRateLocal: e.winRateLocal,
      top2Local: e.top2Local,
      motorTop2: e.motorTop2,
      boatTop2: e.boatTop2,
      age: e.age,
      weight: e.weight,
      exTime: before.exhibition[i]?.exTime ?? null,
    })),
  };

  const priors = { lane_prior: model.lane_prior, racer_lane: model.racer_lane ?? {} };
  const X = buildFeatures(race, priors);
  const utilities = X.map((row) => row.reduce((a, x, i) => a + x * model.weights[i], 0));
  return {
    winProbs: winProbabilities(scoresFromUtilities(utilities)),
    combos: trifectaProbabilities(utilities, STAGE),
    odds,
  };
}

test("実データから6艇の1着確率が出て、合計が1になる", () => {
  const { winProbs } = buildPrediction();
  assert.equal(winProbs.length, 6);
  const sum = winProbs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
  for (const p of winProbs) assert.ok(p > 0 && p < 1, `確率が範囲外: ${p}`);
});

test("1号艇の1着確率が現実的な範囲に入る", () => {
  const { winProbs } = buildPrediction();
  // 全国平均で1号艇の1着率は約55%。個別レースで20〜85%なら妥当。
  assert.ok(
    winProbs[0] > 0.2 && winProbs[0] < 0.85,
    `1号艇の1着確率が不自然です: ${(winProbs[0] * 100).toFixed(1)}%`,
  );
});

test("3連単120通りの確率が出て、実際の結果にも確率が割り当たっている", () => {
  const { combos } = buildPrediction();
  assert.equal(combos.length, 120);

  const actual = combos.find((c) => c.combo === "1-4-2"); // このレースの実際の結果
  assert.ok(actual, "1-4-2 が候補にありません");
  assert.ok(actual.prob > 0.001, `実際の結果の予測確率が低すぎます: ${actual.prob}`);
});

test("オッズから逆算した払戻率が公表値（約75%）と一致する", () => {
  const { odds } = buildPrediction();

  // Σ(1/オッズ) は控除率の逆数になる。
  // 読み取り順が1つでもズレるとこの値は壊れるので、
  // スクレイパーが正しく動いているかの強力なチェックになる。
  const overround = Object.values(odds).reduce((a, v) => a + 1 / v, 0);
  const payoutRate = 1 / overround;

  assert.ok(
    payoutRate > 0.72 && payoutRate < 0.78,
    `払戻率が ${(payoutRate * 100).toFixed(1)}%。約75%になるはずです。` +
      `オッズの読み取り順がずれている可能性があります`,
  );
});

test("モデルの確率が市場のオッズと概ね同じ順位付けになる", () => {
  const { combos, odds } = buildPrediction();

  // 市場のオッズは「多くの人の予想の集約」なので、
  // まともなモデルなら順位付けは強く相関するはず。
  // 相関がほぼ0なら、特徴量かオッズの読み取りが壊れている。
  const rankOf = (entries) => {
    const sorted = [...entries].sort((a, b) => b[1] - a[1]);
    return new Map(sorted.map(([k], i) => [k, i + 1]));
  };
  const modelRank = rankOf(combos.map((c) => [c.combo, c.prob]));
  const marketRank = rankOf(Object.entries(odds).map(([k, v]) => [k, 1 / v]));

  const keys = [...modelRank.keys()];
  const n = keys.length;
  const d2 = keys.reduce(
    (a, k) => a + (modelRank.get(k) - marketRank.get(k)) ** 2,
    0,
  );
  const spearman = 1 - (6 * d2) / (n * (n * n - 1));

  assert.ok(
    spearman > 0.5,
    `モデルと市場の順位相関が ${spearman.toFixed(3)} しかありません`,
  );
});

test("弱いイン・強いアウトのレースでは本命が1-2-3にならない", () => {
  // 実データではなく、極端な設定で挙動を確認する
  const priors = { lane_prior: model.lane_prior, racer_lane: {} };
  const mk = (lane, wr, cls) => ({
    lane, racerId: null, racerClass: cls,
    winRateNational: wr, top2National: wr * 6,
    winRateLocal: wr, top2Local: wr * 6,
    motorTop2: 35, boatTop2: 35, age: 30, weight: 52, exTime: 6.9,
  });
  const race = {
    jcd: 22, windSpeed: 2, waveHeight: 2,
    entries: [
      mk(1, 3.2, "B2"), // 1枠に極端に弱い選手
      mk(2, 5.0, "B1"), mk(3, 5.0, "B1"), mk(4, 5.0, "B1"),
      mk(5, 8.5, "A1"), // 5枠に最強クラス
      mk(6, 5.0, "B1"),
    ],
  };
  const X = buildFeatures(race, priors);
  const u = X.map((row) => row.reduce((a, x, i) => a + x * model.weights[i], 0));
  const combos = trifectaProbabilities(u, STAGE);

  assert.notEqual(
    combos[0].combo,
    "1-2-3",
    "1枠が弱く5枠が強いのに、本命が1-2-3のままです",
  );
});
