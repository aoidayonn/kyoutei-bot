/**
 * 予測の中核。
 *
 *   1. 出走表・直前情報をスクレイプ
 *   2. 特徴量 → 効用 v_i = w·x_i → スコア s_i = exp(v_i)
 *   3. Plackett-Luce で120通りの確率へ展開
 *   4. 現在オッズと突き合わせて期待値 EV = 確率 × オッズ を計算
 *   5. 「的中率トップ」と「期待値トップ」を返す
 */

import model from "./model.json";
import { buildFeatures, FEATURE_NAMES, type Priors, type RaceInput } from "./features";
import { evalTrees, type Tree } from "./gbm";
import {
  scoresFromUtilities,
  trifectaProbabilities,
  type StageCalib,
  winProbabilities,
} from "./plackettLuce";
import { buildPicks, selectPicks, DEFAULT_OPTIONS, type Pick } from "./selection";
import { fetchOdds, fetchRace, type ScrapedRace } from "./scrape";
import { stadiumName } from "./stadiums";

export type { Pick };

type ModelFile = {
  model_type?: string;
  feature_names?: string[];
  weights?: number[];
  trees?: Tree[];
  temperature?: number;
  lane_prior: Record<string, number>;
  racer_lane?: Record<string, number>;
  stage_exponents?: [number, number, number];
  stage_calib?: StageCalib;
};
const MODEL = model as unknown as ModelFile;
const MODEL_TYPE = MODEL.model_type ?? "linear";
const PRIORS: Priors = {
  lane_prior: MODEL.lane_prior,
  racer_lane: MODEL.racer_lane ?? {},
};
const STAGE_EXPONENTS = (MODEL.stage_exponents ?? [1, 1, 1]) as [number, number, number];
// 2・3着の枠ペア相互作用。無ければ従来のスカラー指数だけで展開する（後方互換）。
const STAGE_CALIB = MODEL.stage_calib;

// モデルと実装の不整合はここで止める。
// 特徴量の数や並びがズレていても推論は例外を出さず「静かに誤った確率」を
// 返し続けるため、起動時に落として気づけるようにする（死活監視が検知する）。
{
  const names = MODEL.feature_names ?? [];
  if (names.length !== FEATURE_NAMES.length || names.some((n, i) => n !== FEATURE_NAMES[i])) {
    throw new Error(
      "model.json の feature_names が features.ts と一致しません。再学習が必要です",
    );
  }
  if (MODEL_TYPE === "linear") {
    if ((MODEL.weights ?? []).length !== FEATURE_NAMES.length) {
      throw new Error("model.json の重みの本数が特徴量の数と一致しません");
    }
  } else if (MODEL_TYPE === "lightgbm_rank") {
    if (!MODEL.trees?.length) throw new Error("model.json に木がありません");
    if (typeof MODEL.temperature !== "number" || !Number.isFinite(MODEL.temperature)) {
      throw new Error("model.json に temperature がありません（確率のスケールが未定義になります）");
    }
    const se = MODEL.stage_exponents;
    if (!se || se.length !== 3 || se.some((x) => !Number.isFinite(x))) {
      throw new Error("model.json に stage_exponents がありません（穴目が過大評価されます）");
    }
    // 枠ペア較正は任意だが、あるなら形が正しいことを起動時に確かめる。
    // 行列の形が崩れていると undefined が数値演算に流れ込み、
    // 例外ではなく NaN 確率として静かに広がる。
    if (MODEL.stage_calib) {
      for (const key of ["second", "third"] as const) {
        const b = MODEL.stage_calib[key];
        if (!b) throw new Error(`stage_calib.${key} がありません`);
        if (!Number.isFinite(b.exponent)) throw new Error(`stage_calib.${key}.exponent が不正です`);
        for (const p of [b.pair, b.pair2]) {
          if (p === undefined) continue;
          if (p.length !== 6 || p.some((row) => row.length !== 6 || row.some((x) => !Number.isFinite(x)))) {
            throw new Error(`stage_calib.${key} の枠ペア行列が 6x6 の有限値ではありません`);
          }
        }
        if (b.weights && (b.weights.length !== FEATURE_NAMES.length
                          || b.weights.some((x) => !Number.isFinite(x)))) {
          throw new Error(`stage_calib.${key}.weights の長さが特徴量の数と違います`);
        }
      }
    }
    // 木の構造検証。壊れた子インデックスは evalTree の無限ループになるため
    // 起動時に全ノードを検査する（150本×31ノードなので一瞬）
    for (const [ti, tr] of MODEL.trees.entries()) {
      const n = tr.f.length;
      if (tr.t.length !== n || tr.l.length !== n || tr.r.length !== n) {
        throw new Error(`木${ti}の配列長が不整合です`);
      }
      for (let i = 0; i < n; i++) {
        for (const c of [tr.l[i], tr.r[i]]) {
          const okInternal = Number.isInteger(c) && c > i && c < n; // 前方参照のみ
          const okLeaf = Number.isInteger(c) && c < 0 && ~c < tr.v.length;
          if (!okInternal && !okLeaf) throw new Error(`木${ti}ノード${i}の子参照が不正です`);
        }
        if (!Number.isFinite(tr.t[i])) throw new Error(`木${ti}のしきい値が不正です`);
      }
      if (tr.v.some((x) => !Number.isFinite(x))) throw new Error(`木${ti}の葉値が不正です`);
    }
  } else {
    throw new Error(`未知の model_type: ${MODEL_TYPE}`);
  }
}

/** 特徴量行列 → 効用ベクトル（モデル形式を吸収する唯一の入口） */
function computeUtilities(X: number[][]): number[] {
  if (MODEL_TYPE === "lightgbm_rank") {
    const t = MODEL.temperature ?? 1;
    return X.map((row) => t * evalTrees(MODEL.trees!, row));
  }
  const w = MODEL.weights!;
  return X.map((row) => row.reduce((acc, x, i) => acc + x * w[i], 0));
}

export interface Prediction {
  jcd: number;
  rno: number;
  hd: string;
  stadium: string;
  deadline: string | null;
  hasBeforeInfo: boolean;
  hasOdds: boolean;
  windSpeed: number | null;
  waveHeight: number | null;
  weather: string | null;
  winProbs: number[];
  byEv: Pick[];
  byProb: Pick[];
  reasons: string[];
  verdict: "buy" | "skip";
  race: ScrapedRace;
}


export async function predict(
  jcd: number,
  rno: number,
  hd: string,
): Promise<Prediction> {
  // LINE の replyToken には有効期限があるため、直列で待たない。
  // boatrace.jp は1リクエスト5〜10秒かかることがあり、直列だと期限切れで
  // 返信が丸ごと消える（ユーザーには無反応に見える）。
  const [race, odds] = await Promise.all([
    fetchRace(jcd, rno, hd),
    fetchOdds(jcd, rno, hd),
  ]);
  const hasOdds = Object.keys(odds).length >= 100;

  const input: RaceInput = {
    jcd,
    rno,
    windSpeed: race.windSpeed,
    waveHeight: race.waveHeight,
    entries: race.entries.map((e) => ({
      lane: e.lane,
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
      exTime: e.exTime,
      setsuN: e.setsuN,
      setsuWins: e.setsuWins,
      setsuAvgRank: e.setsuAvgRank,
      setsuAvgSt: e.setsuAvgSt,
    })),
  };

  const X = buildFeatures(input, PRIORS);
  const utilities = computeUtilities(X);
  const scores = scoresFromUtilities(utilities);
  const winProbs = winProbabilities(scores);
  const combos = trifectaProbabilities(utilities, STAGE_EXPONENTS, STAGE_CALIB, X);

  const withEv = buildPicks(combos, odds, DEFAULT_OPTIONS.modelWeight);
  const byProb = withEv.slice(0, DEFAULT_OPTIONS.maxPicks);
  const byEv = selectPicks(withEv);

  return {
    jcd,
    rno,
    hd,
    // 死活監視が「本番に古いモデル/コードが残っていないか」を検知するために出す。
    // リポジトリの model.json の version と一致しなければデプロイ漏れ
    modelVersion: (MODEL as { version?: string }).version ?? "unknown",
    stadium: stadiumName(jcd),
    deadline: race.deadline,
    hasBeforeInfo: race.hasBeforeInfo,
    hasOdds,
    windSpeed: race.windSpeed,
    waveHeight: race.waveHeight,
    weather: race.weather,
    winProbs,
    byEv,
    byProb,
    reasons: buildReasons(race, winProbs, byProb, byEv, hasOdds),
    verdict: byEv.length > 0 ? "buy" : "skip",
    race,
  };
}

// ---------------------------------------------------------------- 根拠の生成

function buildReasons(
  race: ScrapedRace,
  winProbs: number[],
  byProb: Pick[],
  byEv: Pick[],
  hasOdds: boolean,
): string[] {
  const reasons: string[] = [];
  const order = winProbs
    .map((p, i) => ({ lane: i + 1, p }))
    .sort((a, b) => b.p - a.p);

  const top = order[0];
  const e = race.entries[top.lane - 1];
  const bits: string[] = [];
  if (e.racerClass) bits.push(e.racerClass);
  if (e.winRateLocal) bits.push(`当地勝率${e.winRateLocal.toFixed(2)}`);
  if (e.motorTop2) bits.push(`モーター2連率${e.motorTop2.toFixed(1)}%`);
  // 選手名の抽出に失敗しても「1号艇 （A2…）」のような欠けた文にしない
  const name = e.racerName ? ` ${e.racerName}` : "";
  const detail = bits.length ? `（${bits.join("・")}）` : "";
  reasons.push(
    `${top.lane}号艇${name}${detail}が1着本命。1着確率${pct(top.p)}`,
  );

  // 展示タイムの評価
  if (race.hasBeforeInfo) {
    const times = race.entries
      .map((x) => ({ lane: x.lane, t: x.exTime }))
      .filter((x): x is { lane: number; t: number } => x.t !== null)
      .sort((a, b) => a.t - b.t);
    if (times.length) {
      reasons.push(
        `展示タイム最速は${times[0].lane}号艇 ${times[0].t.toFixed(2)}秒`,
      );
    }
  } else {
    reasons.push("直前情報が未公開のため、展示タイムを加味していない暫定予想");
  }

  // 気象
  const w: string[] = [];
  if (race.weather) w.push(race.weather);
  if (race.windSpeed !== null) w.push(`風${race.windSpeed}m`);
  if (race.waveHeight !== null) w.push(`波${race.waveHeight}cm`);
  if (w.length) {
    let note = w.join(" / ");
    if ((race.windSpeed ?? 0) >= 5) note += " — 強風でインが不安定になりやすい";
    else if ((race.waveHeight ?? 0) >= 5) note += " — 荒水面でアウト艇に目がある";
    reasons.push(note);
  }

  // 本命の妙味
  if (hasOdds && byProb.length) {
    const honmei = byProb[0];
    if (honmei.ev !== null) {
      if (honmei.ev < 1.0) {
        reasons.push(
          `最有力の${honmei.combo}は${honmei.odds!.toFixed(1)}倍で期待値${honmei.ev.toFixed(2)}。売れすぎで妙味なし`,
        );
      } else {
        reasons.push(
          `最有力の${honmei.combo}は期待値${honmei.ev.toFixed(2)}で数字上も買える`,
        );
      }
    }
    reasons.push(
      "確率は市場オッズを事前分布としてモデルで補正した値（モデル単体は穴目を過大評価するため）",
    );
  }

  if (!hasOdds) {
    reasons.push(
      "オッズを取得できず期待値を判定できていない。この状態の推奨は参考程度に",
    );
  } else if (byEv.length === 0) {
    reasons.push("期待値が基準を超える買い目がなく、このレースは見送り推奨");
  }

  return reasons;
}

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}
