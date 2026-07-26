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
  parseDeadline,
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

test("今節成績グリッドから当節の成績を取れる", () => {
  // 手動デコード済みの期待値（racelist-setsu.html は 2026-07-26 桐生11R）。
  // lane1 西村: 節の6走 (4R:2着, 12R:2着, 5R:1着, 12R:1着, 3R:1着, 11R:1着)。
  // rno=11 を渡すと「いま見ている11R自身」の列が除外され、5走になる。
  const entries = parseRacelist(fx("racelist-setsu.html"), 11);
  assert.equal(entries.length, 6);
  const e1 = entries[0];
  assert.equal(e1.setsuN, 5);
  assert.equal(e1.setsuWins, 3);
  assert.ok(Math.abs(e1.setsuAvgRank - 1.4) < 1e-9);
  // ST行 (.08,.02,.15,.10,.18) の平均。現レース11Rの .21 は除外される
  assert.ok(Math.abs(e1.setsuAvgSt - 0.106) < 1e-9);
  // lane2 三川: (3,1,1,1,5) → 平均2.2
  assert.equal(entries[1].setsuN, 5);
  assert.equal(entries[1].setsuWins, 3);
  assert.ok(Math.abs(entries[1].setsuAvgRank - 2.2) < 1e-9);
  // 全艇に値が入ること（当日夜のレースなので全員走っている）
  for (const e of entries) assert.ok(e.setsuN >= 4, `${e.lane}号艇の当節成績が取れていません`);
});

test("今節成績: rno を渡さなければ全列を数える（節の初走はゼロ）", () => {
  // 古いフィクスチャ（初日近く）: lane1 は前走1本（5R 6着）のみ
  const entries = parseRacelist(fx("racelist.html"), 1);
  assert.equal(entries[0].setsuN, 1);
  assert.equal(entries[0].setsuWins, 0);
  assert.equal(entries[0].setsuAvgRank, 6);
  // rno=5 を渡すと 5R の列（唯一の前走）が自レースとして除外される
  const excl = parseRacelist(fx("racelist.html"), 5);
  assert.equal(excl[0].setsuN, 0);
  assert.equal(excl[0].setsuAvgRank, null);
  assert.equal(excl[0].setsuAvgSt, null);
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

test("締切時刻はレース番号ごとに正しく取れる（以前は常に1Rの時刻を返していた）", () => {
  const html = fx("racelist.html");
  // 「電話投票締切予定」行には12レース分の時刻が並ぶ
  assert.equal(parseDeadline(html, 1), "12:17");
  assert.equal(parseDeadline(html, 7), "15:23");
  assert.equal(parseDeadline(html, 12), "18:00");
});

test("展示タイムは出現順ではなく艇番で割り当てられる", () => {
  // 3号艇のブロックが読み飛ばされた（セル不足の）HTMLを合成する。
  // 以前の実装では4〜6号艇のタイムが1つずつ前にずれていた。
  const mk = (lane, ex) => `<tbody><td>${lane}</td><td>x</td><td>選手</td><td>52.0kg</td><td>${ex}</td><td>0.5</td></tbody>`;
  const html =
    mk(1, "6.91") + mk(2, "6.92") +
    "<tbody><td>3</td></tbody>" +      // 3号艇: セルが足りず読めない
    mk(4, "6.94") + mk(5, "6.95") + mk(6, "6.96");
  const info = parseBeforeInfo(html);
  assert.equal(info.exhibition[0].exTime, 6.91);
  assert.equal(info.exhibition[2], null);            // 3号艇は欠損
  assert.equal(info.exhibition[3].exTime, 6.94);     // 4号艇はずれない
  assert.equal(info.exhibition[5].exTime, 6.96);
});

test("展示タイムの異常値（0.00等）は欠損として扱われる", () => {
  const mk = (lane, ex) => `<tbody><td>${lane}</td><td>x</td><td>選手</td><td>52.0kg</td><td>${ex}</td><td>0.5</td></tbody>`;
  const html = mk(1, "0.00") + mk(2, "6.92") + mk(3, "12.5") + mk(4, "6.94") + mk(5, "6.95") + mk(6, "6.96");
  const info = parseBeforeInfo(html);
  assert.equal(info.exhibition[0].exTime, null); // 0.00 は展示タイムとしてありえない
  assert.equal(info.exhibition[2].exTime, null); // 12.5 も範囲外
  assert.equal(info.exhibition[1].exTime, 6.92);
});
