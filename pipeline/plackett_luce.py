"""Plackett-Luce モデルによる3連単120通りの確率展開。

各艇の「強さスコア」s_i（= exp(効用)）から、着順全体の確率を整合的に導く:

    P(i→j→k) = s_i/Σs · s_j/(Σs−s_i) · s_k/(Σs−s_i−s_j)

120通りの合計はちょうど 1 になる。
「1-2-3 が常に本命」ではなく、s の大小関係がレースごとに変わるので
弱いイン・強いアウトのレースでは自然に別の組み合わせが上位に来る。
"""
from __future__ import annotations

import math
from itertools import permutations


def trifecta_probabilities(utilities, exponents=(1.0, 1.0, 1.0)) -> dict[str, float]:
    """効用ベクトル（長さ6）から120通りの3連単確率を返す。

    exponents = (1, α, β) は各着順での「強さの効き方」。
    素のPLは (1,1,1) だが、実際は2着以降ほど差がつきにくいので
    学習で推定した α, β を使うと確率が現実に近づく。
    """
    v = [float(x) for x in utilities]
    m = max(v)
    p1, p2, p3 = exponents
    s1 = [math.exp(p1 * (x - m)) for x in v]
    s2 = [math.exp(p2 * (x - m)) for x in v]
    s3 = [math.exp(p3 * (x - m)) for x in v]

    t1 = sum(s1)
    if t1 <= 0:
        return {}

    out = {}
    for i, j, k in permutations(range(6), 3):
        d2 = sum(s2[x] for x in range(6) if x != i)
        d3 = sum(s3[x] for x in range(6) if x != i and x != j)
        if d2 <= 0 or d3 <= 0:
            continue
        out[f"{i+1}-{j+1}-{k+1}"] = (s1[i] / t1) * (s2[j] / d2) * (s3[k] / d3)
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
