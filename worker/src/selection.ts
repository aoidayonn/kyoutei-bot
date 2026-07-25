/**
 * 買い目の選定ロジック。
 *
 * predict.ts から切り出してあるのは、model.json に依存せず単体テストできるようにするため。
 *
 * ここが担うこと
 * --------------
 * 1. モデルの確率を市場オッズとブレンドして、現実的な確率に直す
 * 2. その確率で期待値を計算し、買う価値のある買い目だけを残す
 *
 * なぜブレンドが必要か
 * --------------------
 * 舟券のオッズは何千人もの予想の集約で、単体の線形モデルよりずっと正確。
 * 一方このモデルは分布が市場より平らで、穴目を系統的に過大評価する。
 * 生のモデル確率で期待値を出すと、市場が0.08%と見る買い目を1.0%と主張して
 * 「期待値9.65」のようなあり得ない数字が出る。
 *
 * そこで市場を事前分布、モデルを補正項として対数線形プーリングする:
 *
 *     p_final ∝ q_market^(1-λ) · p_model^λ
 *
 * λ を小さくするほど市場寄りになる。
 * つまりこのボットの edge は「市場からの小さなズレ」からしか生まれない。
 * それが控除率25%の市場で戦うということ。
 */

export interface Combo {
  combo: string;
  prob: number;
}

export interface Pick {
  combo: string;
  /** 最終的な予測確率（オッズがあれば市場とブレンド済み） */
  prob: number;
  /** ブレンド前のモデル単体の確率 */
  modelProb: number;
  /** オッズから逆算した市場の暗黙確率 */
  marketProb: number | null;
  odds: number | null;
  ev: number | null;
}

export interface SelectionOptions {
  /** モデルをどれだけ信用するか（0=市場そのまま, 1=モデルそのまま） */
  modelWeight: number;
  /** これ未満の期待値は買わない */
  evThreshold: number;
  /** これを超える期待値は「妙味」ではなく「モデルの誤り」とみなして捨てる */
  evCeiling: number;
  /** これ未満の確率は実用性がないので捨てる */
  minProb: number;
  /** モデルが市場の何倍まで強気なら信用するか */
  maxProbRatio: number;
  /** 最大何点まで推奨するか */
  maxPicks: number;
}

export const DEFAULT_OPTIONS: SelectionOptions = {
  // λ の最適値はオッズ履歴がないと決められないため、暫定的に控えめな値。
  // snapshot_odds.py のデータが数週間たまったら実測して調整すること。
  modelWeight: 0.25,
  evThreshold: 1.05,
  evCeiling: 2.5,
  minProb: 0.01,
  maxProbRatio: 4.0,
  maxPicks: 5,
};

/** オッズから市場の暗黙確率を求める。控除率を戻して合計1に正規化する。 */
export function marketProbabilities(
  odds: Record<string, number>,
): Record<string, number> {
  let invSum = 0;
  for (const v of Object.values(odds)) {
    if (v > 0) invSum += 1 / v;
  }
  const out: Record<string, number> = {};
  if (invSum <= 0) return out;
  for (const [combo, v] of Object.entries(odds)) {
    if (v > 0) out[combo] = 1 / v / invSum;
  }
  return out;
}

/**
 * モデル確率と市場確率をブレンドして Pick の一覧を作る。
 * オッズが取れていない場合はモデル確率をそのまま使う。
 */
export function buildPicks(
  combos: Combo[],
  odds: Record<string, number>,
  modelWeight: number = DEFAULT_OPTIONS.modelWeight,
): Pick[] {
  const hasOdds = Object.keys(odds).length >= 100;

  if (!hasOdds) {
    return combos
      .map((c) => ({
        combo: c.combo,
        prob: c.prob,
        modelProb: c.prob,
        marketProb: null,
        odds: odds[c.combo] ?? null,
        ev: null,
      }))
      .sort((a, b) => b.prob - a.prob);
  }

  const market = marketProbabilities(odds);

  // 対数線形プーリング
  const raw: Record<string, number> = {};
  let z = 0;
  for (const c of combos) {
    const q = market[c.combo];
    if (!q || c.prob <= 0) continue;
    const v = Math.pow(q, 1 - modelWeight) * Math.pow(c.prob, modelWeight);
    raw[c.combo] = v;
    z += v;
  }

  return combos
    .map((c) => {
      const prob = z > 0 && raw[c.combo] ? raw[c.combo] / z : c.prob;
      const o = odds[c.combo] ?? null;
      return {
        combo: c.combo,
        prob,
        modelProb: c.prob,
        marketProb: market[c.combo] ?? null,
        odds: o,
        ev: o === null ? null : prob * o,
      };
    })
    .sort((a, b) => b.prob - a.prob);
}

/** 買う価値のある買い目だけを残す。該当なしなら空配列（= 見送り推奨）。 */
export function selectPicks(
  picks: Pick[],
  options: Partial<SelectionOptions> = {},
): Pick[] {
  const o = { ...DEFAULT_OPTIONS, ...options };

  return picks
    .filter((p) => {
      if (p.ev === null) return false;
      if (p.ev < o.evThreshold) return false;
      // 高すぎる期待値は妙味ではなくモデルの誤り
      if (p.ev > o.evCeiling) return false;
      if (p.prob < o.minProb) return false;
      // モデルが市場から極端に離れている買い目は外挿とみなす
      if (p.marketProb && p.modelProb / p.marketProb > o.maxProbRatio) return false;
      return true;
    })
    // 期待値順に並べると必ず万券が上に来るが、実際に的中する見込みは低い。
    // 確率順（buildPicks でソート済み）のまま上位を採る。
    .slice(0, o.maxPicks);
}
