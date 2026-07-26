"""モデルが「市場（オッズ）」に勝てているかを測る。

なぜこれが最重要か
------------------
控除率25%の市場では、無作為に買えば回収率は75%に収束する。
100%を超える唯一の道は **市場より正確な確率を出すこと** であり、
「1着的中率が高い」ことでも「線形モデルより良い」ことでもない。

比較相手はベースラインではなく市場である。

市場の確率をどう手に入れるか
----------------------------
公式は過去オッズを公開していないが、**払戻金から逆算できる**。
パリミュチュエルでは

    払戻金(円/100円) = 総売上 × 0.75 / その組の売上
                     = 0.75 / q_market

なので q_market = 0.75 / (払戻金/100)。
的中した組についてしか分からないが、NLL に必要なのはまさに実現した
組の確率だけなので、これで**市場の3連単NLLが正確に測れる**。

    python market_edge.py --start 2026-05-01

出力の見方: delta = log p_model − log q_market（実現した組について）
  delta < 0 … 市場のほうが正確（＝エッジなし。買えば負ける）
  delta > 0 … モデルのほうが正確（＝その区分は攻める価値がある）
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import train as T
import model_io
import evaluate_model as E
import stage_calib as SC

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"

# 3連単の払戻率（控除率25%）
PAYOUT_RATE = 0.75


def model_logp_trifecta(V, order, exps):
    """各レースについて、実現した1-2-3着の対数確率を返す。(N,)"""
    n = V.shape[0]
    idx = np.arange(n)
    lp = np.zeros(n)
    mask = np.ones((n, 6), dtype=bool)
    for k, p in enumerate(exps):
        Vp = p * V
        Vm = np.where(mask, Vp, -np.inf)
        mx = Vm.max(axis=1, keepdims=True)
        ex = np.where(mask, np.exp(Vm - mx), 0.0)
        lp += Vp[idx, order[:, k]] - (mx[:, 0] + np.log(ex.sum(axis=1)))
        mask = mask.copy()
        mask[idx, order[:, k]] = False
    return lp


def _stage_logp(V, X, order, calib):
    """較正込みで、実現した1-2-3着の対数確率をレースごとに返す（符号は正の負対数）。"""
    N = V.shape[0]
    idx = np.arange(N)
    out = np.zeros(N)
    mask = np.ones((N, 6), dtype=bool)
    for stage in range(3):
        if stage == 0:
            # firstブロック（1着の残差較正）も本番同様に適用する
            sc = SC.first_scores(V, X, calib)
        else:
            blk = calib["second" if stage == 1 else "third"]
            sc = float(blk["exponent"]) * V
            sc = sc + np.asarray(blk["pair"], dtype=float)[order[:, 0]]
            if stage == 2 and blk.get("pair2") is not None:
                sc = sc + np.asarray(blk["pair2"], dtype=float)[order[:, 1]]
            if blk.get("weights") is not None:
                w = np.asarray(blk["weights"], dtype=float)
                sc = sc + (X.reshape(N * 6, -1) @ w).reshape(N, 6)
        s = np.where(mask, sc, -1e30)
        mx = s.max(axis=1, keepdims=True)
        e = np.where(mask, np.exp(s - mx), 0.0)
        out += -(sc[idx, order[:, stage]] - (mx[:, 0] + np.log(e.sum(axis=1))))
        mask = mask.copy()
        mask[idx, order[:, stage]] = False
    return out


def summarize(name, deltas):
    d = np.asarray(deltas)
    if len(d) == 0:
        return
    # 平均の標準誤差。区分ごとの差がノイズかどうかを見るために必ず出す。
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    mark = ""
    if len(d) >= 200 and abs(d.mean()) > 2 * se:
        mark = " ◀ 有意" if d.mean() > 0 else ""
    print(f"    {name:<22} N={len(d):>6,}  delta={d.mean():+.4f} ± {2*se:.4f}"
          f"  勝率={float((d > 0).mean()):.1%}{mark}")


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

    # 払戻金が取れているレースだけに絞る（元返し100円は下限で潰れているので除く）
    races = [r for r in races if r.get("payout") and r["payout"] > 100]
    if len(races) < 200:
        raise SystemExit(f"レースが {len(races)} 件しかありません")

    X, order, _ = T.build_matrices(races, m.priors)
    S = np.array([m.raw_utilities(rows) for rows in X])
    Xa = np.asarray(X, dtype=float)

    # 評価窓で温度と段階指数を再推定する（evaluate_model.py と同じ対称手順）。
    # モデルに最大限有利な条件で測る。
    t, _ = E.calibrate_temperature(S, order)
    exps = T.fit_stage_exponents_from_V(t * S, order, verbose=False)
    V = t * S

    if m.stage_calib:
        # 較正入りのモデルなら、本番と同じ展開で測らないと過小評価になる。
        # 温度も評価窓での再推定ではなく焼き込み値を使う（較正は
        # その温度スケールで推定されており、混ぜるとスケールが崩れる）
        V = getattr(m, "temperature", 1.0) * S
        lp_model = -_stage_logp(V, Xa, order, m.stage_calib)
        print("  ※ 較正ありモデル: 焼き込み温度・較正のまま（出荷状態）で測定")
    else:
        lp_model = model_logp_trifecta(V, order, exps)
    payout = np.array([r["payout"] for r in races], dtype=float)
    lp_market = np.log(PAYOUT_RATE / (payout / 100.0))

    delta = lp_model - lp_market

    print(f"\n対象: {a.start} 〜 {a.end or '最新'}   {len(races):,} レース"
          f"   (温度 {t}, 段階指数 {[round(x,3) for x in exps]})")
    print(f"\n  市場の3連単NLL   {-lp_market.mean():.4f}")
    print(f"  モデルの3連単NLL {-lp_model.mean():.4f}")
    print(f"  差               {-lp_model.mean() + lp_market.mean():+.4f}"
          f"   （プラスならモデルが劣る）")
    print(f"\n  モデルが市場より的中組に高い確率を置けたレース: "
          f"{float((delta > 0).mean()):.1%}")

    # 「全体では負けていても、どこかに勝てる区分があるか」を探す。
    # あればそこだけ買えばよい。無ければ現状の設計では100%は無理。
    print("\n─── 区分別（delta>0 の区分だけが攻める価値がある） ───")

    print("\n  ■ 人気（1番人気＝市場の本命）")
    by = defaultdict(list)
    for r, d in zip(races, delta):
        pop = r.get("trifecta_pop")
        if not pop:
            continue
        k = "1番人気" if pop == 1 else "2-3番人気" if pop <= 3 else \
            "4-10番人気" if pop <= 10 else "11-30番人気" if pop <= 30 else "31番人気以下"
        by[k].append(d)
    for k in ["1番人気", "2-3番人気", "4-10番人気", "11-30番人気", "31番人気以下"]:
        summarize(k, by.get(k, []))

    print("\n  ■ 風速")
    by = defaultdict(list)
    for r, d in zip(races, delta):
        w = r.get("wind_speed")
        if w is None:
            continue
        by["0-2m" if w <= 2 else "3-4m" if w <= 4 else "5m以上"].append(d)
    for k in ["0-2m", "3-4m", "5m以上"]:
        summarize(k, by.get(k, []))

    print("\n  ■ モデルの1着確信度")
    pw = np.exp(V - V.max(axis=1, keepdims=True))
    pw /= pw.sum(axis=1, keepdims=True)
    conf = pw.max(axis=1)
    by = defaultdict(list)
    for c, d in zip(conf, delta):
        by["〜40%" if c < 0.4 else "40-60%" if c < 0.6 else "60-80%" if c < 0.8 else "80%〜"].append(d)
    for k in ["〜40%", "40-60%", "60-80%", "80%〜"]:
        summarize(k, by.get(k, []))

    print("\n  ■ レース場（delta上位5・下位3）")
    by = defaultdict(list)
    for r, d in zip(races, delta):
        by[r["jcd"]].append(d)
    ranked = sorted(((np.mean(v), k, v) for k, v in by.items() if len(v) >= 200),
                    reverse=True)
    for mean, k, v in ranked[:5]:
        summarize(f"{k:02d}", v)
    print("    ...")
    for mean, k, v in ranked[-3:]:
        summarize(f"{k:02d}", v)


if __name__ == "__main__":
    main()
