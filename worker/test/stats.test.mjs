/**
 * 実運用成績の集計テスト。
 *
 * ここが間違っていると「勝っているつもりで負けている」ことになるので、
 * 回収率の計算は特に厳密に確認する。
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { summarize, formatSummary } from "../src/stats.ts";
import { parseRaceResult } from "../src/scrape.ts";

const here = dirname(fileURLToPath(import.meta.url));

function row(overrides = {}) {
  return {
    race_id: "20260720-22-01",
    hd: "20260720",
    jcd: 22,
    rno: 1,
    verdict: "buy",
    picks_json: JSON.stringify({
      byEv: [
        { combo: "1-4-2", prob: 0.05, odds: 15.3, ev: 1.1 },
        { combo: "1-2-4", prob: 0.06, odds: 10.7, ev: 1.08 },
      ],
      byProb: [{ combo: "1-2-4", prob: 0.06, odds: 10.7, ev: 1.08 }],
    }),
    win_probs_json: "[0.5,0.2,0.1,0.1,0.05,0.05]",
    actual: null,
    payout: null,
    ...overrides,
  };
}

test("結果が未確定の予想は集計に含まれない", () => {
  const s = summarize([row(), row(), row()]);
  assert.equal(s.settled, 0);
  assert.equal(s.roi, null);
});

test("的中したレースの回収率が正しく計算される", () => {
  // 2点買い（200円）で 1-4-2 が的中、払戻1,530円
  const s = summarize([row({ actual: "1-4-2", payout: 1530 })]);

  assert.equal(s.settled, 1);
  assert.equal(s.buyRaces, 1);
  assert.equal(s.totalPicks, 2);
  assert.equal(s.hitRaces, 1);
  assert.equal(s.invested, 200);
  assert.equal(s.returned, 1530);
  assert.equal(s.roi, 1530 / 200);
});

test("外れたレースは払戻0で計上される", () => {
  const s = summarize([row({ actual: "3-5-6", payout: 8000 })]);
  assert.equal(s.hitRaces, 0);
  assert.equal(s.invested, 200);
  assert.equal(s.returned, 0);
  assert.equal(s.roi, 0);
});

test("的中1回・外れ4回で回収率が正しく出る", () => {
  const rows = [
    row({ actual: "1-4-2", payout: 1530 }),
    row({ actual: "2-1-3", payout: 2000 }),
    row({ actual: "3-1-2", payout: 3000 }),
    row({ actual: "4-1-2", payout: 4000 }),
    row({ actual: "5-1-2", payout: 5000 }),
  ];
  const s = summarize(rows);
  assert.equal(s.buyRaces, 5);
  assert.equal(s.invested, 1000); // 2点 × 5レース × 100円
  assert.equal(s.returned, 1530);
  assert.equal(s.roi, 1.53);
  assert.equal(s.hitRate, 1 / 5);
});

test("見送りレースは投資に含まれない", () => {
  const skip = row({
    actual: "1-2-4",
    payout: 1070,
    verdict: "skip",
    picks_json: JSON.stringify({
      byEv: [],
      byProb: [{ combo: "1-2-4", prob: 0.06, odds: 10.7, ev: 0.64 }],
    }),
  });
  const s = summarize([skip]);

  assert.equal(s.skipRaces, 1);
  assert.equal(s.buyRaces, 0);
  assert.equal(s.invested, 0);
  assert.equal(s.roi, null);
  // 見送ったが本命は当たっていた
  assert.equal(s.skipWouldHaveHit, 1);
});

test("払戻金が取れていない場合は予想時のオッズで代用する", () => {
  const s = summarize([row({ actual: "1-4-2", payout: null })]);
  assert.equal(s.returned, 1530); // 15.3倍 × 100円
});

test("比較用の「的中率トップ1点」が別に集計される", () => {
  const rows = [
    row({ actual: "1-2-4", payout: 1070 }), // byProb[0] が的中
    row({ actual: "9-9-9", payout: 0 }),
  ];
  const s = summarize(rows);
  assert.equal(s.topPickHitRate, 0.5);
});

test("件数が少ないうちは「ほぼ運」だと明示する", () => {
  const text = formatSummary(summarize([row({ actual: "1-4-2", payout: 1530 })]), 0);
  assert.match(text, /ほぼ運/);
});

test("記録がないときは分かりやすく案内する", () => {
  const text = formatSummary(summarize([]), 3);
  assert.match(text, /まだ結果が確定した予想がありません/);
  assert.match(text, /結果待ち 3 件/);
});

test("結果ページから3連単の結果・払戻金・人気を取れる", () => {
  const r = parseRaceResult(
    readFileSync(join(here, "fixtures", "raceresult.html"), "utf-8"),
  );
  // 競走成績ファイル(K)側の記録と一致すること
  assert.equal(r.trifecta, "1-4-2");
  assert.equal(r.payout, 1530);
  assert.equal(r.popularity, 3);
});

test("結果がまだ出ていないページでは null が返る", () => {
  assert.deepEqual(parseRaceResult("<html><body>準備中</body></html>"), {
    trifecta: null,
    payout: null,
    popularity: null,
  });
});
