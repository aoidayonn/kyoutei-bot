"""Plackett-Luce モデルによる3連単120通りの確率展開。

各艇の効用 v_i から、着順全体の確率を整合的に導く:

    P(i→j→k) = P(1着=i) · P(2着=j | 1着=i) · P(3着=k | 1着=i, 2着=j)

120通りの合計はちょうど 1 になる。
「1-2-3 が常に本命」ではなく、v の大小関係がレースごとに変わるので
弱いイン・強いアウトのレースでは自然に別の組み合わせが上位に来る。

素のPLは各段階を exp(v_j) に比例させるが、それだと
**1着が誰であっても2着の相対順位は同じ**という誤った仮定になる。
実データでは外枠がまくると他の外枠が繰り上がる（stage_calib.py 参照）。
そこで較正パラメータがある場合は条件付きスコアに枠ペアの項を足す:

    2着: a2 * v_j + A[1着の枠][j]                  + w2 . x_j
    3着: a3 * v_k + B[1着の枠][k] + C[2着の枠][k]   + w3 . x_k

★ worker/src/plackettLuce.ts と完全に一致させること。
   片方だけ直すと予想が静かに壊れる。npm test で数値一致を強制している。
"""
from __future__ import annotations

import math

LANES = 6


def _dot(w, x) -> float:
    s = 0.0
    for i in range(len(w)):
        s += w[i] * x[i]
    return s


def _stage_base(v, m, block, features):
    """a * (v_j - m) + w . x_j を6艇分。"""
    a = float(block.get("exponent", 1.0))
    w = block.get("weights")
    out = []
    for j in range(LANES):
        b = a * (v[j] - m)
        if w and features is not None:
            b += _dot(w, features[j])
        out.append(b)
    return out


def first_stage_scores(v, calib, features):
    """1着ステージのスコア（"first" ブロック適用済み）。v は温度適用済み効用。"""
    m = max(v)
    blk = (calib or {}).get("first")
    out = []
    for j in range(LANES):
        s = v[j] - m
        if blk:
            s += blk["bias"][j]
            w = blk.get("weights")
            if w and features is not None:
                s += _dot(w, features[j])
        out.append(s)
    return out


def win_probabilities_calibrated(utilities, calib, features):
    """1着確率（firstブロック込み）。★worker側 predict.ts の winProbs と一致させる。"""
    sc = first_stage_scores([float(x) for x in utilities], calib, features)
    m = max(sc)
    e = [math.exp(x - m) for x in sc]
    t = sum(e)
    return [x / t for x in e]


def _cond_probs(base, extras, excluded):
    """base[j] + Σextras[j] を、excluded を除いた集合でソフトマックス。"""
    sc = [0.0] * LANES
    mx = -math.inf
    for j in range(LANES):
        if j in excluded:
            continue
        s = base[j]
        for e in extras:
            s += e[j]
        sc[j] = s
        if s > mx:
            mx = s
    tot = 0.0
    ex = [0.0] * LANES
    for j in range(LANES):
        if j in excluded:
            continue
        e = math.exp(sc[j] - mx)
        ex[j] = e
        tot += e
    if tot <= 0:
        return None
    return [e / tot for e in ex]


def trifecta_probabilities(utilities, exponents=(1.0, 1.0, 1.0),
                           calib=None, features=None) -> dict[str, float]:
    """効用ベクトル（長さ6）から120通りの3連単確率を返す。

    calib があれば枠ペア相互作用つきの条件付きモデルを使う。
    無ければ従来どおり exponents=(1, α, β) のスカラー指数のみ。
    """
    v = [float(x) for x in utilities]
    m = max(v)
    if not math.isfinite(m):
        return {}

    if calib:
        b2 = calib["second"]
        b3 = calib["third"]
        base2 = _stage_base(v, m, b2, features)
        base3 = _stage_base(v, m, b3, features)
        A = b2.get("pair")
        B = b3.get("pair")
        C = b3.get("pair2")
        # 1着ステージの残差補正（b[枠] + w1・x）。stage_calib.py の "first"
        base1 = first_stage_scores(v, calib, features)
    else:
        e1, a2, a3 = exponents
        base1 = [e1 * (x - m) for x in v]
        base2 = [a2 * (x - m) for x in v]
        base3 = [a3 * (x - m) for x in v]
        A = B = C = None

    p1 = _cond_probs(base1, [], set())
    if p1 is None:
        return {}

    out: dict[str, float] = {}
    for i in range(LANES):
        extras2 = [A[i]] if A else []
        p2 = _cond_probs(base2, extras2, {i})
        if p2 is None:
            continue
        for j in range(LANES):
            if j == i:
                continue
            extras3 = []
            if B:
                extras3.append(B[i])
            if C:
                extras3.append(C[j])
            p3 = _cond_probs(base3, extras3, {i, j})
            if p3 is None:
                continue
            for k in range(LANES):
                if k == i or k == j:
                    continue
                out[f"{i+1}-{j+1}-{k+1}"] = p1[i] * p2[j] * p3[k]
    return out


def win_probabilities(scores) -> list[float]:
    total = sum(scores)
    return [s / total for s in scores] if total > 0 else [0.0] * len(scores)


def softmax(utilities) -> list[float]:
    m = max(utilities)
    exps = [math.exp(u - m) for u in utilities]
    t = sum(exps)
    return [e / t for e in exps]


def scores_from_utilities(utilities) -> list[float]:
    """効用 v_i -> スコア s_i = exp(v_i)。数値安定のため最大値を引く。"""
    m = max(utilities)
    return [math.exp(u - m) for u in utilities]
