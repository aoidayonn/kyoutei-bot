/**
 * スクレイパーのテスト。実際に保存した boatrace.jp の HTML を使う。
 *
 * fixtures/ は 2026-07-20 福岡1R のページを保存したもの。
 * サイトのマークアップが変わるとここが落ちるので、
 * 「ボットの予想がおかしい」より先にこのテストで気づけるようにしている。
 *
 * fixtures を更新するには:
 *   node scripts/fetch-fixtures.mjs 22 1 20260720
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  parseRacelist,
  parseBeforeInfo,
  parseOdds3t,
} from "../src/scrape.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fx = (name) => readFileSync(join(here, "fixtures", name), "utf-8");

test("出走表から6艇分の選手データを取れる", () => {
  const entries = parseRacelist(fx("racelist.html"));
  assert.equal(entries.length, 6, "6艇そろっていません");

  const first = entries[0];
  assert.equal(first.lane, 1);
  assert.equal(first.racerId, 5009);
  assert.equal(first.racerClass, "A2");
  assert.equal(first.winRateNational, 5.42);
  assert.equal(first.top2National, 36.0);
  assert.equal(first.motorNo, 56);
  assert.equal(first.motorTop2, 44.44);
  assert.equal(first.boatNo, 170);
  assert.equal(first.boatTop2, 34.62);
  assert.equal(first.age, 29);
  assert.equal(first.weight, 55.1);

  // 全艇に主要な数値が入っていること
  for (const e of entries) {
    assert.ok(e.racerId, `${e.lane}号艇の登録番号が取れていません`);
    assert.ok(e.winRateNational !== null, `${e.lane}号艇の全国勝率が取れていません`);
    assert.ok(e.motorTop2 !== null, `${e.lane}号艇のモーター2連率が取れていません`);
  }
});

test("直前情報から展示タイムと気象を取れる", () => {
  const info = parseBeforeInfo(fx("beforeinfo.html"));
  assert.equal(info.exhibition.length, 6);
  assert.equal(info.exhibition[0].exTime, 6.92);

  assert.equal(info.temperature, 31.0);
  assert.equal(info.windSpeed, 2);
  assert.equal(info.waterTemp, 26.0);
  assert.equal(info.waveHeight, 2);

  for (const e of info.exhibition) {
    assert.ok(e.exTime !== null && e.exTime > 5 && e.exTime < 9, `展示タイムが異常: ${e.exTime}`);
  }
});

test("3連単オッズを120通りすべて取れる", () => {
  const odds = parseOdds3t(fx("odds3t.html"));
  assert.equal(Object.keys(odds).length, 120, "120通りそろっていません");

  // 実ページで目視確認した値
  assert.equal(odds["1-2-3"], 13.9);
  assert.equal(odds["2-1-3"], 26.8);
  assert.equal(odds["3-1-2"], 76.9);
  assert.equal(odds["2-1-4"], 16.4);

  // オッズは必ず1.0以上
  for (const [combo, v] of Object.entries(odds)) {
    assert.ok(v >= 1.0, `${combo} のオッズが不正: ${v}`);
  }

  // 控除率チェック: Σ(1/オッズ) は 1/0.75 ≒ 1.33 前後になるはず
  const overround = Object.values(odds).reduce((a, v) => a + 1 / v, 0);
  assert.ok(
    overround > 1.2 && overround < 1.5,
    `Σ(1/オッズ)=${overround.toFixed(3)} が想定外。読み取り順がずれている可能性があります`,
  );
});
