/**
 * 特徴量の定義（pipeline/features.py の完全なミラー）
 *
 * 片方だけ直すと学習した重みが無意味になるので、必ず両方を同時に更新すること。
 */

export const FEATURE_NAMES = [
  "lane_1", "lane_2", "lane_3", "lane_4", "lane_5", "lane_6",
  "lane_prior",
  "racer_lane_edge",
  "win_rate_national", "top2_national",
  "win_rate_local", "top2_local",
  "motor_top2", "boat_top2",
  "class_A1", "class_A2", "class_B1",
  "age", "weight",
  "ex_time", "ex_missing",
  "wind_x_lane1", "wind_x_lane2", "wind_x_lane3",
  "wind_x_lane4", "wind_x_lane5", "wind_x_lane6",
  "wave_x_lane1", "wave_x_lane2", "wave_x_lane3",
  "wave_x_lane4", "wave_x_lane5", "wave_x_lane6",
  "wr_x_lane1", "wr_x_lane2", "wr_x_lane3",
  "wr_x_lane4", "wr_x_lane5", "wr_x_lane6",
  "motor_x_lane1", "motor_x_lane2", "motor_x_lane3",
  "motor_x_lane4", "motor_x_lane5", "motor_x_lane6",
  "ex_x_lane1", "ex_x_lane2", "ex_x_lane3",
  "ex_x_lane4", "ex_x_lane5", "ex_x_lane6",
  // 当節成績（今節成績グリッドから）。実測で3連単NLL -0.016〜-0.021（2窓で再現）
  "setsu_n",
  "setsu_wins",
  "setsu_avg_rank",
  // レース番号（12R=優勝戦などの番組情報）
  "race_no",
  // 注意: 当節ST平均は実験2窓で効いたが本番構成のゲートで再現せず不採用
  // （詳細は features.py 側のコメント）。scrape.ts の ST 解析は将来用に残す。
] as const;

export const N_FEATURES = FEATURE_NAMES.length;

const IDX: Record<string, number> = Object.fromEntries(
  FEATURE_NAMES.map((n, i) => [n, i]),
);

const DEFAULTS: Record<string, number> = {
  win_rate_national: 5.5,
  top2_national: 35.0,
  win_rate_local: 5.5,
  top2_local: 35.0,
  motor_top2: 35.0,
  boat_top2: 35.0,
  age: 35.0,
  weight: 52.0,
  ex_time: 6.85,
};

// 当地未走などで 0.00 が入る項目は欠損扱いする。
// top2 系を外していた頃は、当地未走の選手（全体の約1割）に
// 「当地勝率=平均5.5、当地2連率=0%」という矛盾した組が渡っていた。
const ZERO_IS_MISSING = new Set([
  "win_rate_national", "top2_national",
  "win_rate_local", "top2_local",
  "motor_top2", "boat_top2",
]);

export interface Priors {
  /** "24-1" -> 場×枠の事前ロジット */
  lane_prior: Record<string, number>;
  /** "4320-1" -> その選手がその枠で平均よりどれだけ強いか */
  racer_lane: Record<string, number>;
}

export interface EntryInput {
  lane: number;
  racerId?: number | null;
  racerClass?: string | null;
  winRateNational?: number | null;
  top2National?: number | null;
  winRateLocal?: number | null;
  top2Local?: number | null;
  motorTop2?: number | null;
  boatTop2?: number | null;
  age?: number | null;
  weight?: number | null;
  exTime?: number | null;
  /** 当節ここまでのレース数（今節成績グリッドの着順セルを数えたもの） */
  setsuN?: number | null;
  /** うち1着の数 */
  setsuWins?: number | null;
  /** 平均着順。走っていなければ null（→3.5扱い） */
  setsuAvgRank?: number | null;
  /** 当節の実ST平均（秒）。F・欠測は除く。無ければ null */
  setsuAvgSt?: number | null;
}

export interface RaceInput {
  jcd: number;
  /** レース番号 1〜12 */
  rno?: number | null;
  windSpeed?: number | null;
  waveHeight?: number | null;
  entries: EntryInput[]; // 6件、lane 昇順
}

function num(v: number | null | undefined, key: string): [number, boolean] {
  if (v === null || v === undefined || Number.isNaN(v)) return [DEFAULTS[key], true];
  if (v === 0 && ZERO_IS_MISSING.has(key)) return [DEFAULTS[key], true];
  return [v, false];
}

/** 1レース分（6艇）の特徴量行列を返す。 */
export function buildFeatures(race: RaceInput, priors: Priors): number[][] {
  const lanePrior = priors.lane_prior ?? {};
  const racerLane = priors.racer_lane ?? {};
  // NaN は ?? をすり抜ける。木の分岐 x<=t は NaN で常に false になり、
  // 例外を出さず「静かに間違った確率」を返すため、ここで完全に遮断する
  const wind = Number.isFinite(race.windSpeed as number) ? (race.windSpeed as number) : 0;
  const wave = Number.isFinite(race.waveHeight as number) ? (race.waveHeight as number) : 0;

  // 展示タイムはレース内平均を基準に相対化する
  const exVals: (number | null)[] = race.entries.map((e) => {
    const [v, miss] = num(e.exTime, "ex_time");
    if (miss) return null;
    // 展示タイムの妥当範囲外（0.00 やチルト値の混入など）は欠損として扱う。
    // 異常値を1つ通すと ex_mean ごと歪んでレース全体の確率が崩壊する。
    return v >= 5.0 && v <= 9.0 ? v : null;
  });
  const present = exVals.filter((v): v is number => v !== null);
  const exMean = present.length
    ? present.reduce((a, b) => a + b, 0) / present.length
    : DEFAULTS.ex_time;

  return race.entries.map((e, i) => {
    const lane = e.lane;
    const row = new Array<number>(N_FEATURES).fill(0);

    row[IDX[`lane_${lane}`]] = 1;
    row[IDX["lane_prior"]] = lanePrior[`${race.jcd}-${lane}`] ?? 0;
    row[IDX["racer_lane_edge"]] = e.racerId
      ? racerLane[`${e.racerId}-${lane}`] ?? 0
      : 0;

    const scaled: [string, number | null | undefined, number][] = [
      ["win_rate_national", e.winRateNational, 10],
      ["top2_national", e.top2National, 100],
      ["win_rate_local", e.winRateLocal, 10],
      ["top2_local", e.top2Local, 100],
      ["motor_top2", e.motorTop2, 100],
      ["boat_top2", e.boatTop2, 100],
      ["age", e.age, 10],
      ["weight", e.weight, 10],
    ];
    for (const [key, raw, div] of scaled) {
      const [v] = num(raw, key);
      row[IDX[key]] = v / div;
    }

    const cls = (e.racerClass ?? "B2").trim().toUpperCase();
    if (cls === "A1" || cls === "A2" || cls === "B1") row[IDX[`class_${cls}`]] = 1;

    const ex = exVals[i];
    let exRel = 0;
    if (ex === null) {
      row[IDX["ex_time"]] = 0;
      row[IDX["ex_missing"]] = 1;
    } else {
      // 展示が平均より速い（値が小さい）ほど正。±0.3秒相当でクリップし、
      // 万一の異常値がレース全体を支配しないようにする
      exRel = Math.max(-3, Math.min(3, (exMean - ex) * 10));
      row[IDX["ex_time"]] = exRel;
    }

    row[IDX[`wind_x_lane${lane}`]] = wind / 10;
    row[IDX[`wave_x_lane${lane}`]] = wave / 10;

    // 交互作用。全国勝率は 5.5、モーター2連率は 35% を中心に振る
    const [wr] = num(e.winRateNational, "win_rate_national");
    const [mt] = num(e.motorTop2, "motor_top2");
    row[IDX[`wr_x_lane${lane}`]] = (wr - 5.5) / 2;
    row[IDX[`motor_x_lane${lane}`]] = (mt - 35) / 20;
    row[IDX[`ex_x_lane${lane}`]] = exRel;

    // 当節成績。Python側 (features.py) と同一のロジック:
    // n>0 のときだけ平均着順を使い、それ以外は 3.5（節の初走）
    const sn = Number.isFinite(e.setsuN as number) ? (e.setsuN as number) : 0;
    row[IDX["setsu_n"]] = sn;
    row[IDX["setsu_wins"]] = Number.isFinite(e.setsuWins as number) ? (e.setsuWins as number) : 0;
    row[IDX["setsu_avg_rank"]] =
      sn > 0 && Number.isFinite(e.setsuAvgRank as number) ? (e.setsuAvgRank as number) : 3.5;
    row[IDX["race_no"]] = Number.isFinite(race.rno as number) ? (race.rno as number) : 0;

    // 最終防衛線: どの経路から来た NaN/Inf もモデルに渡さない
    for (let k = 0; k < row.length; k++) {
      if (!Number.isFinite(row[k])) row[k] = 0;
    }
    return row;
  });
}
