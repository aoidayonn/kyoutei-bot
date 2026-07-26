/**
 * Plackett-Luce モデルによる3連単120通りの確率展開。
 *
 *   P(i→j→k) = P(1着=i) · P(2着=j | 1着=i) · P(3着=k | 1着=i, 2着=j)
 *
 * 120通りの合計はちょうど 1 になる。
 * 「1-2-3 が常に本命」ではなく、各艇の効用 v の大小でレースごとに変わる。
 *
 * 素のPLは各段階を exp(v_j) に比例させるが、それだと
 * **1着が誰であっても2着の相対順位は同じ**という誤った仮定になる。
 * 実データでは外枠がまくると他の外枠が繰り上がる:
 *
 *     1着=1枠のとき 2着が5枠 12.1% / 6枠  7.3%
 *     1着=4枠のとき 2着が5枠 20.0% / 6枠 13.1%
 *
 * そこで較正パラメータがあれば条件付きスコアに枠ペアの項を足す:
 *
 *     2着: a2 * v_j + A[1着の枠][j]                  + w2 · x_j
 *     3着: a3 * v_k + B[1着の枠][k] + C[2着の枠][k]   + w3 · x_k
 *
 * ★ pipeline/plackett_luce.py と完全に一致させること。
 *   片方だけ直すと予想が静かに壊れる。npm test で数値一致を強制している。
 */

export interface Combo {
  combo: string; // "1-2-3"
  prob: number;
}

export interface StageBlock {
  exponent: number;
  pair?: number[][];   // [条件の枠][対象の枠]
  pair2?: number[][];  // 3着のみ: [2着の枠][対象の枠]
  weights?: number[];  // 特徴量の線形補正（標準化は畳み込み済み）
}

export interface FirstBlock {
  bias: number[];      // b[枠]（6個）
  weights?: number[];  // w1（標準化は畳み込み済み）
}

export interface StageCalib {
  /** 1着ステージの残差補正。2・3着と同じ思想の対称的な完成形 */
  first?: FirstBlock;
  second: StageBlock;
  third: StageBlock;
}

const LANES = 6;

/** 効用ベクトル（長さ6）からスコア s_i = exp(v_i) を作る。数値安定のため最大値を引く。 */
export function scoresFromUtilities(v: number[]): number[] {
  const m = Math.max(...v);
  if (!Number.isFinite(m)) return v.map(() => 0);
  return v.map((x) => Math.exp(x - m));
}

/** 各艇の1着確率。 */
export function winProbabilities(scores: number[]): number[] {
  const total = scores.reduce((a, b) => a + b, 0);
  return scores.map((s) => s / total);
}

function dot(w: number[], x: number[]): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * x[i];
  return s;
}

function stageBase(
  v: number[], m: number, block: StageBlock, features?: number[][],
): number[] {
  const a = block.exponent ?? 1;
  const w = block.weights;
  const out: number[] = [];
  for (let j = 0; j < LANES; j++) {
    let b = a * (v[j] - m);
    if (w && features) b += dot(w, features[j]);
    out.push(b);
  }
  return out;
}

/** 1着ステージのスコア（"first" ブロック適用済み）。v は温度適用済み効用。 */
export function firstStageScores(
  v: number[], calib?: StageCalib, features?: number[][],
): number[] {
  const m = Math.max(...v);
  const blk = calib?.first;
  const out: number[] = [];
  for (let j = 0; j < LANES; j++) {
    let s = v[j] - m;
    if (blk) {
      s += blk.bias[j];
      if (blk.weights && features) s += dot(blk.weights, features[j]);
    }
    out.push(s);
  }
  return out;
}

/** 1着確率（firstブロック込み）。★pipeline/plackett_luce.py と一致させる。 */
export function winProbabilitiesCalibrated(
  utilities: number[], calib?: StageCalib, features?: number[][],
): number[] {
  const sc = firstStageScores(utilities.map(Number), calib, features);
  const m = Math.max(...sc);
  if (!Number.isFinite(m)) return utilities.map(() => 0);
  const e = sc.map((x) => Math.exp(x - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / t);
}

/** base[j] + Σextras[j] を、excluded を除いた集合でソフトマックス。 */
function condProbs(
  base: number[], extras: number[][], excluded: number[],
): number[] | null {
  const sc = new Array(LANES).fill(0);
  let mx = -Infinity;
  for (let j = 0; j < LANES; j++) {
    if (excluded.includes(j)) continue;
    let s = base[j];
    for (const e of extras) s += e[j];
    sc[j] = s;
    if (s > mx) mx = s;
  }
  if (!Number.isFinite(mx)) return null;
  const ex = new Array(LANES).fill(0);
  let tot = 0;
  for (let j = 0; j < LANES; j++) {
    if (excluded.includes(j)) continue;
    const e = Math.exp(sc[j] - mx);
    ex[j] = e;
    tot += e;
  }
  if (!(tot > 0)) return null;
  return ex.map((e) => e / tot);
}

/**
 * 120通りすべての3連単確率。確率降順でソートして返す。
 *
 * calib があれば枠ペア相互作用つきの条件付きモデルを使う。
 * 無ければ従来どおり exponents=[1, α, β] のスカラー指数のみ。
 */
export function trifectaProbabilities(
  utilities: number[],
  exponents: [number, number, number] = [1, 1, 1],
  calib?: StageCalib,
  features?: number[][],
): Combo[] {
  const v = utilities.map(Number);
  const m = Math.max(...v);
  // 特徴量やモデルの不整合で NaN が混ざった場合、NaN の確率を返すより
  // 空を返して「取得失敗」として扱わせる方が安全側。
  if (!Number.isFinite(m)) return [];

  let base1: number[];
  let base2: number[];
  let base3: number[];
  let A: number[][] | undefined;
  let B: number[][] | undefined;
  let C: number[][] | undefined;

  if (calib) {
    base1 = firstStageScores(v, calib, features);
    base2 = stageBase(v, m, calib.second, features);
    base3 = stageBase(v, m, calib.third, features);
    A = calib.second.pair;
    B = calib.third.pair;
    C = calib.third.pair2;
  } else {
    const [e1, a2, a3] = exponents;
    base1 = v.map((x) => e1 * (x - m));
    base2 = v.map((x) => a2 * (x - m));
    base3 = v.map((x) => a3 * (x - m));
  }

  const p1 = condProbs(base1, [], []);
  if (!p1) return [];

  const out: Combo[] = [];
  for (let i = 0; i < LANES; i++) {
    const p2 = condProbs(base2, A ? [A[i]] : [], [i]);
    if (!p2) continue;
    for (let j = 0; j < LANES; j++) {
      if (j === i) continue;
      const extras3: number[][] = [];
      if (B) extras3.push(B[i]);
      if (C) extras3.push(C[j]);
      const p3 = condProbs(base3, extras3, [i, j]);
      if (!p3) continue;
      for (let k = 0; k < LANES; k++) {
        if (k === i || k === j) continue;
        out.push({
          combo: `${i + 1}-${j + 1}-${k + 1}`,
          prob: p1[i] * p2[j] * p3[k],
        });
      }
    }
  }
  out.sort((a, b) => b.prob - a.prob);
  return out;
}
