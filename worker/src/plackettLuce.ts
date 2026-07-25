/**
 * Plackett-Luce モデルによる3連単120通りの確率展開。
 *
 *   P(i→j→k) = s_i/Σs · s_j/(Σs−s_i) · s_k/(Σs−s_i−s_j)
 *
 * 120通りの合計はちょうど 1 になる。
 * 「1-2-3 が常に本命」ではなく、各艇のスコア s の大小でレースごとに変わる。
 */

export interface Combo {
  combo: string; // "1-2-3"
  prob: number;
}

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

/**
 * 120通りすべての3連単確率。確率降順でソートして返す。
 *
 * exponents = [1, α, β] は各着順での「強さの効き方」。
 * 素の Plackett-Luce は全段階で 1 を仮定するが、実際のボートレースは
 * 1着が決まった後の2・3着争いの方が横並びに近い（学習では α≈0.87, β≈0.62）。
 * ここを 1 のままにすると穴目の確率が過大評価され、
 * 「期待値が高い買い目」が万券ばかりになってしまう。
 */
export function trifectaProbabilities(
  utilities: number[],
  exponents: [number, number, number] = [1, 1, 1],
): Combo[] {
  const m = Math.max(...utilities);
  // 特徴量やモデルの不整合で NaN が混ざった場合、NaN の確率を返すより
  // 空を返して「取得失敗」として扱わせる方が安全側。
  if (!Number.isFinite(m)) return [];
  const [e1, e2, e3] = exponents;
  const s1 = utilities.map((v) => Math.exp(e1 * (v - m)));
  const s2 = utilities.map((v) => Math.exp(e2 * (v - m)));
  const s3 = utilities.map((v) => Math.exp(e3 * (v - m)));

  const t1 = s1.reduce((a, b) => a + b, 0);
  const out: Combo[] = [];
  if (t1 <= 0) return out;

  for (let i = 0; i < 6; i++) {
    let d2 = 0;
    for (let x = 0; x < 6; x++) if (x !== i) d2 += s2[x];
    if (d2 <= 0) continue;

    for (let j = 0; j < 6; j++) {
      if (j === i) continue;
      let d3 = 0;
      for (let x = 0; x < 6; x++) if (x !== i && x !== j) d3 += s3[x];
      if (d3 <= 0) continue;

      for (let k = 0; k < 6; k++) {
        if (k === i || k === j) continue;
        out.push({
          combo: `${i + 1}-${j + 1}-${k + 1}`,
          prob: (s1[i] / t1) * (s2[j] / d2) * (s3[k] / d3),
        });
      }
    }
  }
  out.sort((a, b) => b.prob - a.prob);
  return out;
}
