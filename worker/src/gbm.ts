/**
 * LightGBM の木を辿る推論器（pipeline/model_io.py の完全なミラー）。
 *
 * 正解仕様は Python 側（lightgbm 本体と誤差0で一致することを学習時に検証済み）。
 * このファイルを変更したら、必ず fixture の expected_utilities と一致することを
 * テストで確認すること。
 *
 * 木のフォーマット:
 *   tree = { f: 特徴番号[], t: しきい値[], l: 左子[], r: 右子[], v: 葉値[] }
 *   - ノード i が内部ノードのとき: x[f[i]] <= t[i] なら l[i]、そうでなければ r[i]
 *   - 子が負のとき: ~child が葉番号（v[~child] を出力に加算）
 *   - 特徴量に NaN は来ない前提（features.ts が既定値で埋める）
 */

export interface Tree {
  f: number[];
  t: number[];
  l: number[];
  r: number[];
  v: number[];
}

export function evalTree(tree: Tree, x: number[]): number {
  const { f, t, l, r, v } = tree;
  if (f.length === 0) return v[0]; // 分割なし（葉1枚）
  let i = 0;
  for (;;) {
    i = x[f[i]] <= t[i] ? l[i] : r[i];
    if (i < 0) return v[~i];
  }
}

export function evalTrees(trees: Tree[], x: number[]): number {
  let s = 0;
  for (const tree of trees) s += evalTree(tree, x);
  return s;
}
