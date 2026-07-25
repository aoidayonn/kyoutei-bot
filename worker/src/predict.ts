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
import { buildFeatures, type Priors, type RaceInput } from "./features";
import {
  scoresFromUtilities,
  trifectaProbabilities,
  winProbabilities,
  type Combo,
} from "./plackettLuce";
import { fetchOdds, fetchRace, type ScrapedRace } from "./scrape";
import { stadiumName } from "./stadiums";

const WEIGHTS: number[] = model.weights as number[];
const PRIORS: Priors = {
  lane_prior: model.lane_prior as Record<string, number>,
  racer_lane: (model.racer_lane ?? {}) as Record<string, number>,
};
const STAGE_EXPONENTS = (model.stage_exponents ?? [1, 1, 1]) as [number, number, number];

export interface Pick {
  combo: string;
  prob: number;
  odds: number | null;
  ev: number | null;
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

/** 期待値の閾値。オッズは締切直前まで動くので少し余裕を持たせる。 */
const EV_THRESHOLD = 1.05;
const MAX_PICKS = 5;

export async function predict(
  jcd: number,
  rno: number,
  hd: string,
): Promise<Prediction> {
  const race = await fetchRace(jcd, rno, hd);
  const odds = await fetchOdds(jcd, rno, hd);
  const hasOdds = Object.keys(odds).length >= 100;

  const input: RaceInput = {
    jcd,
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
    })),
  };

  const X = buildFeatures(input, PRIORS);
  const utilities = X.map((row) =>
    row.reduce((acc, x, i) => acc + x * WEIGHTS[i], 0),
  );
  const scores = scoresFromUtilities(utilities);
  const winProbs = winProbabilities(scores);
  const combos = trifectaProbabilities(utilities, STAGE_EXPONENTS);

  const withEv: Pick[] = combos.map((c: Combo) => {
    const o = odds[c.combo] ?? null;
    return { combo: c.combo, prob: c.prob, odds: o, ev: o === null ? null : c.prob * o };
  });

  const byProb = withEv.slice(0, MAX_PICKS);

  const candidates = withEv
    .filter((p) => p.ev !== null && p.ev >= EV_THRESHOLD)
    // 期待値だけだと超万券に偏るので、確率が低すぎるものは落とす
    .filter((p) => p.prob >= 0.005)
    .sort((a, b) => (b.ev! - a.ev!));
  const byEv = candidates.slice(0, MAX_PICKS);

  return {
    jcd,
    rno,
    hd,
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
  reasons.push(
    `${top.lane}号艇 ${e.racerName}（${bits.join("・")}）が1着本命。1着確率${pct(top.p)}`,
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
  }

  if (!hasOdds) {
    reasons.push("オッズを取得できなかったため、期待値の判定はできていない");
  } else if (byEv.length === 0) {
    reasons.push("期待値が基準を超える買い目がなく、このレースは見送り推奨");
  }

  return reasons;
}

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}
