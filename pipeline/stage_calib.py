"""2着・3着の条件付き分布を較正する（枠ペア相互作用 + 特徴量の再学習）。

なぜ必要か
----------
素のPlackett-Luceは、2着の確率を exp(alpha * v_j) に比例させる。
これは **1着が誰であっても2着の相対順位は同じ** という強い仮定であり、
実データはそうなっていない:

        1着=1枠のとき  2着が5枠 12.1% / 6枠  7.3%
        1着=4枠のとき  2着が5枠 20.0% / 6枠 13.1%

外枠がまくると内枠が押し出され、他の外枠が繰り上がる。
段階指数（スカラー）をどう調整してもこの構造は表現できない。

そこで条件付きスコアに枠ペアの項を足す:

        2着: alpha * v_j + A[1着の枠][j]         + w2 . x_j
        3着: beta  * v_k + B[1着の枠][k] + C[2着の枠][k] + w3 . x_k

w2, w3 は「2着/3着に効く特徴は1着に効く特徴と違う」ぶんの線形補正。
いずれも条件付きロジットなので凸で、素直な勾配法で最適解に届く。

実測（学習2024-02〜2026-01 / 較正2026-02〜04 / 評価2026-05〜07、評価窓は未使用）:

        現行PL                    3.8569   市場との差 +0.1912
        + 枠ペア                  3.8050   市場との差 +0.1393
        + 枠ペア + 特徴量再学習     3.7898   市場との差 +0.1240

1着ステージの補正（"first" ブロック）
------------------------------------
2・3着に行列と特徴量補正があるのに、1着だけ温度スカラー1個だった。
較正窓で 1着にも b[枠] + w1・x の残差補正を当てると、直近のドリフト
（枠の強さの季節変動・モーター周期など）を拾える:

        1着: v_j + b[j] + w1 . x_j        （v は温度適用済み）

実測: 1着LL -0.0046（窓1）/ -0.0049（窓2）。ステージ加法なので
3連単NLLにそのまま乗る。

較正はテスト期間に触れてはいけない。train_gbm.py では
補助モデル用の val 区間（学習期間の末尾3か月）だけで推定する。
"""
from __future__ import annotations

import numpy as np

# 標準化を重みに畳み込むと TS 側は内積1本で済む。
# score = w.((x-mu)/sd) = (w/sd).x - const で、const は全艇共通なので
# ソフトマックスで消える。よって w/sd だけ配ればよい。


def _softmax_nll(scores, mask, target):
    s = np.where(mask, scores, -1e30)
    mx = s.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(s - mx), 0.0)
    p = e / e.sum(axis=1, keepdims=True)
    n = scores.shape[0]
    nll = float(-np.log(np.clip(p[np.arange(n), target], 1e-15, None)).mean())
    return nll, p


def _fit_stage(V, Xs, mask, target, conds, use_feat, steps=400, lr=0.15,
               l2_pair=1e-3, l2_feat=1e-2):
    """score_j = a*V_j + sum_c M_c[cond_c, j] (+ w.Xs_j) を最尤で当てる。

    conds: [(カーディナリティ, インデックス配列), ...]
    Xs:    標準化済み特徴量 (N, 6, F)
    """
    N, K = V.shape
    idx = np.arange(N)
    a = np.array([1.0])
    Ms = [np.zeros((c, 6)) for c, _ in conds]
    w = np.zeros(Xs.shape[2]) if use_feat else None
    X2 = Xs.reshape(N * K, -1) if use_feat else None
    state: dict[str, tuple] = {}

    def adam(name, p, g, step):
        m, v = state.setdefault(name, (np.zeros_like(p), np.zeros_like(p)))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        state[name] = (m, v)
        return p - lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)

    for step in range(1, steps + 1):
        sc = a[0] * V
        for M, (_, i) in zip(Ms, conds):
            sc = sc + M[i]
        if use_feat:
            sc = sc + (X2 @ w).reshape(N, K)
        _, p = _softmax_nll(sc, mask, target)
        g = p.copy()
        g[idx, target] -= 1.0
        g /= N
        a = adam("a", a, np.array([float((g * V).sum())]), step)
        for j, (M, (c, i)) in enumerate(zip(Ms, conds)):
            acc = np.zeros((c, 6))
            np.add.at(acc, i, g)
            Ms[j] = adam(f"M{j}", M, acc + l2_pair * M, step)
            # 行ごとの定数はソフトマックスで消えるので中心化して一意にする
            Ms[j] -= Ms[j].mean(axis=1, keepdims=True)
        if use_feat:
            w = adam("w", w, (X2.T @ g.reshape(-1)) + l2_feat * w, step)
    return float(a[0]), Ms, w


def _fit_first(V, Xs, target, steps=500, lr=0.1, l2_feat=1e-2):
    """1着ステージの残差補正: score_j = V_j + b[j] + w.Xs_j（凸）。"""
    N = V.shape[0]
    idx = np.arange(N)
    bias = np.zeros(6)
    w = np.zeros(Xs.shape[2])
    X2 = Xs.reshape(N * 6, -1)
    mask = np.ones((N, 6), dtype=bool)
    state: dict[str, tuple] = {}

    def adam(name, p, g, step):
        m, v = state.setdefault(name, (np.zeros_like(p), np.zeros_like(p)))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        state[name] = (m, v)
        return p - lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)

    for step in range(1, steps + 1):
        sc = V + bias + (X2 @ w).reshape(N, 6)
        _, p = _softmax_nll(sc, mask, target)
        g = p.copy()
        g[idx, target] -= 1.0
        g /= N
        bias = adam("b", bias, g.sum(axis=0), step)
        bias -= bias.mean()  # ソフトマックスで消える定数を固定して一意にする
        w = adam("w", w, X2.T @ g.reshape(-1) + l2_feat * w, step)
    return bias, w


def first_scores(V, X, calib):
    """1着ステージのスコア（firstブロック適用済み）。V: (N,6) 温度適用済み。"""
    V = np.asarray(V, dtype=float)
    blk = (calib or {}).get("first")
    if not blk:
        return V
    N = V.shape[0]
    sc = V + np.asarray(blk["bias"], dtype=float)
    if blk.get("weights") is not None:
        w = np.asarray(blk["weights"], dtype=float)
        sc = sc + (np.asarray(X, dtype=float).reshape(N * 6, -1) @ w).reshape(N, 6)
    return sc


def trifecta_nll(V, X, order, calib):
    """較正パラメータ込みの3連単NLL（1〜3着の完全対数尤度）をベクトル化して計算。

    plackett_luce.trifecta_probabilities と同じモデルを、行列演算で回すだけ。
    重みは標準化を畳み込み済みなので生の X にそのまま掛けてよい
    （引き算した定数は全艇共通なのでソフトマックスで消える）。
    """
    V = np.asarray(V, dtype=float)
    X = np.asarray(X, dtype=float)
    order = np.asarray(order)
    N = V.shape[0]
    idx = np.arange(N)

    total = 0.0
    mask = np.ones((N, 6), dtype=bool)
    for stage in range(3):
        if stage == 0:
            sc = first_scores(V, X, calib)
        else:
            blk = calib["second" if stage == 1 else "third"]
            sc = float(blk["exponent"]) * V
            sc = sc + np.asarray(blk["pair"], dtype=float)[order[:, 0]]
            if stage == 2 and blk.get("pair2") is not None:
                sc = sc + np.asarray(blk["pair2"], dtype=float)[order[:, 1]]
            if blk.get("weights") is not None:
                w = np.asarray(blk["weights"], dtype=float)
                sc = sc + (X.reshape(N * 6, -1) @ w).reshape(N, 6)
        nll, _ = _softmax_nll(sc, mask, order[:, stage])
        total += nll
        mask = mask.copy()
        mask[idx, order[:, stage]] = False
    return total


def fit(V, X, order, use_features=True, verbose=True):
    """較正パラメータ一式を推定する。

    V: (N,6) 温度適用済みの効用   X: (N,6,F) 生の特徴量   order: (N,3) 着順の艇index
    戻り値は model.json にそのまま入る dict。
    """
    V = np.asarray(V, dtype=float)
    X = np.asarray(X, dtype=float)
    order = np.asarray(order)
    N = V.shape[0]
    idx = np.arange(N)

    flat = X.reshape(-1, X.shape[2])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    out: dict = {}

    # 1着ステージの残差補正（2・3着と同じ思想の、対称的な完成形）
    if use_features:
        bias, w1 = _fit_first(V, Xs, order[:, 0])
        out["first"] = {
            "bias": [round(float(b), 6) for b in bias],
            # 標準化を畳み込んで配る（TS側は内積1本で済む）
            "weights": [round(float(x), 8) for x in (w1 / sd)],
        }
        if verbose:
            nll, _ = _softmax_nll(first_scores(V, X, out), np.ones((N, 6), bool),
                                  order[:, 0])
            print(f"  first: b[枠]+w1  1着NLL {nll:.4f}")

    for stage, key in ((1, "second"), (2, "third")):
        mask = np.ones((N, 6), dtype=bool)
        for j in range(stage):
            mask[idx, order[:, j]] = False
        target = order[:, stage]
        conds = [(6, order[:, j]) for j in range(stage)]
        a, Ms, w = _fit_stage(V, Xs, mask, target, conds, use_features)
        nll, _ = _softmax_nll(
            a * V + sum(M[i] for M, (_, i) in zip(Ms, conds))
            + ((Xs.reshape(N * 6, -1) @ w).reshape(N, 6) if w is not None else 0.0),
            mask, target)
        block = {
            "exponent": round(a, 6),
            # 1着の枠で条件付け。stage3 は 1着と2着の両方。
            "pair": [[round(x, 6) for x in row] for row in Ms[0]],
        }
        if stage == 2:
            block["pair2"] = [[round(x, 6) for x in row] for row in Ms[1]]
        if w is not None:
            # 標準化を畳み込んで配る（TS側は内積1本で済む）
            block["weights"] = [round(x, 8) for x in (w / sd)]
        out[key] = block
        if verbose:
            print(f"  {key}: 指数 {a:.3f}  枠ペア項あり"
                  f"{' + 特徴量補正' if w is not None else ''}  NLL {nll:.4f}")
    return out
