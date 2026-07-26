"""実験: PLの2着・3着に「枠ペアの相互作用」を入れると当たるようになるか。

動機
----
現行PLは 2着の確率を exp(alpha * v_j) に比例させる。これは
**1着が誰であっても2着の相対順位は同じ**という強い仮定である。
実データはそうなっていない:

    1着=1枠のとき 2着が5枠 12.1% / 6枠 7.3%
    1着=4枠のとき 2着が5枠 20.0% / 6枠 13.1%

外枠がまくると内枠が押し出され、他の外枠が浮上する。
現行の定式化ではこの構造を原理的に表現できない。

そこで段階2・3のスコアに枠ペアの項を足す:

    2着: alpha * v_j + A[1着の枠][j]
    3着: beta  * v_k + B[1着の枠][k] + C[2着の枠][k]

いずれも条件付きロジットで凸なので、素直な勾配法で最適解に到達する。

検証プロトコル
--------------
GBMの学習に使っていない期間を前半・後半に割り、
**前半でA,B,Cを推定し、後半でだけ評価する**。
ベースライン(現行PL)も同じ前半でalpha,betaを推定して後半で測る。
そうしないと「パラメータを増やした分だけ良く見える」だけになる。

    python exp_lanepair.py --start 2026-05-01
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import train as T
import model_io
import evaluate_model as E

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"


def _softmax_masked(scores, mask):
    s = np.where(mask, scores, -np.inf)
    mx = s.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(s - mx), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def _nll(scores, mask, target):
    p = _softmax_masked(scores, mask)
    n = scores.shape[0]
    return float(-np.log(np.clip(p[np.arange(n), target], 1e-15, None)).mean()), p


def fit_conditional(V, mask, target, cond, n_cond, steps=400, lr=0.2, l2=1e-3):
    """score_j = a*V_j + sum_c M_c[cond_c, j] を最尤で当てる（条件付きロジット）。

    cond: (n_cond, N) の条件インデックス（例: 1着の枠, 2着の枠）
    戻り値: a, [M_c (6,6)], 収束時のNLL
    """
    N, K = V.shape
    idx = np.arange(N)
    a = 1.0
    Ms = [np.zeros((6, 6)) for _ in range(n_cond)]
    # Adam
    ma, va = 0.0, 0.0
    mM = [np.zeros((6, 6)) for _ in range(n_cond)]
    vM = [np.zeros((6, 6)) for _ in range(n_cond)]
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(1, steps + 1):
        scores = a * V
        for c in range(n_cond):
            scores = scores + Ms[c][cond[c]]
        nll, p = _nll(scores, mask, target)

        # dNLL/dscore
        g = p.copy()
        g[idx, target] -= 1.0
        g /= N

        ga = float((g * V).sum())
        gM = []
        for c in range(n_cond):
            acc = np.zeros((6, 6))
            np.add.at(acc, cond[c], g)
            acc += l2 * Ms[c]
            gM.append(acc)

        ma = b1 * ma + (1 - b1) * ga
        va = b2 * va + (1 - b2) * ga * ga
        a -= lr * (ma / (1 - b1**step)) / (np.sqrt(va / (1 - b2**step)) + eps)
        for c in range(n_cond):
            mM[c] = b1 * mM[c] + (1 - b1) * gM[c]
            vM[c] = b2 * vM[c] + (1 - b2) * gM[c] ** 2
            Ms[c] -= lr * (mM[c] / (1 - b1**step)) / (np.sqrt(vM[c] / (1 - b2**step)) + eps)
            # 行ごとの定数はソフトマックスで消えるので中心化して一意にする
            Ms[c] -= Ms[c].mean(axis=1, keepdims=True)

    scores = a * V
    for c in range(n_cond):
        scores = scores + Ms[c][cond[c]]
    nll, _ = _nll(scores, mask, target)
    return a, Ms, nll


def fit_exponent_only(V, mask, target, steps=300, lr=0.05):
    """現行方式（指数スカラーのみ）。同じ最適化器で公平に当てる。"""
    N = V.shape[0]
    idx = np.arange(N)
    a = 1.0
    m, v = 0.0, 0.0
    for step in range(1, steps + 1):
        nll, p = _nll(a * V, mask, target)
        g = p.copy()
        g[idx, target] -= 1.0
        g /= N
        ga = float((g * V).sum())
        m = 0.9 * m + 0.1 * ga
        v = 0.999 * v + 0.001 * ga * ga
        a -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    nll, _ = _nll(a * V, mask, target)
    return a, nll


def stage_inputs(V, order, stage):
    """stage(0,1,2) の mask / target / 条件インデックスを作る。"""
    N = V.shape[0]
    idx = np.arange(N)
    mask = np.ones((N, 6), dtype=bool)
    for j in range(stage):
        mask[idx, order[:, j]] = False
    target = order[:, stage]
    cond = [order[:, j] for j in range(stage)]
    return mask, target, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent
                                           / "worker" / "src" / "model.json"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", default=str(DB))
    a = ap.parse_args()

    m = model_io.load_model(Path(a.model))
    con = sqlite3.connect(a.db)
    races = T.load_races(con, a.start, a.end)
    con.close()
    races = [r for r in races if r.get("payout") and r["payout"] > 100]

    X, order, _ = T.build_matrices(races, m.priors)
    S = np.array([m.raw_utilities(rows) for rows in X])
    t, _ = E.calibrate_temperature(S, order)
    V = t * S
    N = len(races)
    half = N // 2
    payout = np.array([r["payout"] for r in races], dtype=float)
    print(f"N={N:,}  前半{half:,}で推定 / 後半{N-half:,}で評価  (温度 {t})")

    tr = slice(0, half)
    te = slice(half, N)
    market_te = float(-np.log(0.75 / (payout[te] / 100.0)).mean())

    base_total = 0.0
    new_total = 0.0
    print(f"\n{'ステージ':<12}{'現行(指数のみ)':>16}{'枠ペア相互作用':>18}{'改善':>10}")
    for stage in range(3):
        mask, target, cond = stage_inputs(V, order, stage)

        a0, _ = fit_exponent_only(V[tr], mask[tr], target[tr])
        nll0, _ = _nll(a0 * V[te], mask[te], target[te])

        if stage == 0:
            nll1 = nll0          # 1着には条件が無いので同じ
        else:
            a1, Ms, _ = fit_conditional(
                V[tr], mask[tr], target[tr], [c[tr] for c in cond], stage)
            sc = a1 * V[te]
            for c in range(stage):
                sc = sc + Ms[c][cond[c][te]]
            nll1, _ = _nll(sc, mask[te], target[te])

        base_total += nll0
        new_total += nll1
        name = ["1着", "2着|1着", "3着|1,2着"][stage]
        print(f"{name:<12}{nll0:>16.4f}{nll1:>18.4f}{nll0-nll1:>+10.4f}")

    print(f"{'3連単NLL':<12}{base_total:>16.4f}{new_total:>18.4f}{base_total-new_total:>+10.4f}")
    print(f"\n  市場（同じ後半区間）  {market_te:.4f}")
    print(f"  市場との差   現行 {base_total-market_te:+.4f}  →  新 {new_total-market_te:+.4f}")


if __name__ == "__main__":
    main()
