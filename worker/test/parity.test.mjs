/**
 * Python(pipeline/features.py) と TypeScript(worker/src/features.ts) の
 * 特徴量が数値レベルで完全に一致することを検証する。
 *
 * この2つがズレると学習した重みが無意味になり、しかも
 * 「なんとなく動いているが予想が微妙」という気づきにくい壊れ方をする。
 *
 * 期待値ファイルは `python pipeline/export_fixture.py` で生成する。
 * features.py を触ったら必ず再生成してこのテストを通すこと。
 *
 *   npm test
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

function loadFixture() {
  try {
    return JSON.parse(readFileSync(join(here, "fixture.json"), "utf-8"));
  } catch {
    return null;
  }
}

test("特徴量の並びが Python 側と一致している", (t) => {
  const fixture = loadFixture();
  if (!fixture) {
    t.skip("fixture.json がありません。python pipeline/export_fixture.py を実行してください");
    return;
  }

  const tsSource = readFileSync(join(here, "..", "src", "features.ts"), "utf-8");
  const block = tsSource.match(/FEATURE_NAMES = \[([\s\S]*?)\] as const;/);
  assert.ok(block, "FEATURE_NAMES を抽出できませんでした");

  const tsNames = [...block[1].matchAll(/"([a-zA-Z0-9_]+)"/g)].map((m) => m[1]);
  assert.deepEqual(
    tsNames,
    fixture.feature_names,
    "並びが違います。features.py と features.ts の両方を更新してください",
  );
});

test("特徴量の数値が Python 側と一致している", async (t) => {
  const fixture = loadFixture();
  if (!fixture) {
    t.skip("fixture.json がありません");
    return;
  }

  let buildFeatures;
  try {
    ({ buildFeatures } = await import("../src/features.ts"));
  } catch (e) {
    t.skip(`features.ts を読み込めませんでした（Node 22以上で --experimental-strip-types が必要）: ${e}`);
    return;
  }

  // fixture の snake_case を features.ts の camelCase に詰め替える
  const race = {
    jcd: fixture.race.jcd,
    windSpeed: fixture.race.wind_speed,
    waveHeight: fixture.race.wave_height,
    entries: fixture.race.entries.map((e) => ({
      lane: e.lane,
      racerId: e.racer_id,
      racerClass: e.racer_class,
      winRateNational: e.win_rate_national,
      top2National: e.top2_national,
      winRateLocal: e.win_rate_local,
      top2Local: e.top2_local,
      motorTop2: e.motor_top2,
      boatTop2: e.boat_top2,
      age: e.age,
      weight: e.weight,
      exTime: e.ex_time,
    })),
  };

  const actual = buildFeatures(race, fixture.priors);
  assert.equal(actual.length, fixture.expected.length);

  for (let boat = 0; boat < actual.length; boat++) {
    for (let f = 0; f < fixture.feature_names.length; f++) {
      const diff = Math.abs(actual[boat][f] - fixture.expected[boat][f]);
      assert.ok(
        diff < 1e-9,
        `${boat + 1}号艇の ${fixture.feature_names[f]} が一致しません: ` +
          `TS=${actual[boat][f]} / Python=${fixture.expected[boat][f]}`,
      );
    }
  }
});

test("model.json の構造が実装と整合している", () => {
  const model = JSON.parse(readFileSync(join(here, "..", "src", "model.json"), "utf-8"));
  const type = model.model_type ?? "linear";
  if (type === "linear") {
    assert.equal(model.weights.length, model.feature_names.length,
      "重みの本数と特徴量の数が違います。学習をやり直してください");
  } else if (type === "lightgbm_rank") {
    assert.ok(Array.isArray(model.trees) && model.trees.length > 0, "木がありません");
    assert.ok(typeof model.temperature === "number", "temperature がありません");
    for (const tr of model.trees.slice(0, 5)) {
      for (const key of ["f", "t", "l", "r", "v"]) assert.ok(key in tr, `木に ${key} がありません`);
    }
  } else {
    assert.fail(`未知の model_type: ${type}`);
  }
});

test("GBMの木トラバーサルが Python 実装と数値一致する", async (t) => {
  const model = JSON.parse(readFileSync(join(here, "..", "src", "model.json"), "utf-8"));
  if ((model.model_type ?? "linear") !== "lightgbm_rank") {
    t.skip("線形モデルなのでスキップ");
    return;
  }
  const fixture = loadFixture();
  if (!fixture?.expected_utilities) {
    t.skip("fixture に expected_utilities がありません。export_fixture.py を実行してください");
    return;
  }

  const { buildFeatures } = await import("../src/features.ts");
  const { evalTrees } = await import("../src/gbm.ts");

  const race = {
    jcd: fixture.race.jcd,
    windSpeed: fixture.race.wind_speed,
    waveHeight: fixture.race.wave_height,
    entries: fixture.race.entries.map((e) => ({
      lane: e.lane, racerId: e.racer_id, racerClass: e.racer_class,
      winRateNational: e.win_rate_national, top2National: e.top2_national,
      winRateLocal: e.win_rate_local, top2Local: e.top2_local,
      motorTop2: e.motor_top2, boatTop2: e.boat_top2,
      age: e.age, weight: e.weight, exTime: e.ex_time,
    })),
  };
  // 期待値は「本番モデルのpriors」で作られている（fixtureの合成priorsではない）
  const priors = { lane_prior: model.lane_prior, racer_lane: model.racer_lane ?? {} };
  const X = buildFeatures(race, priors);
  const temp = model.temperature ?? 1;
  X.forEach((row, i) => {
    const u = temp * evalTrees(model.trees, row);
    const diff = Math.abs(u - fixture.expected_utilities[i]);
    assert.ok(diff < 1e-9,
      `${i + 1}号艇の効用が一致しません: TS=${u} / Python=${fixture.expected_utilities[i]}`);
  });
});
