"""市場ブレンド率 λ を実測し、本番の買い方の回収率を初めて測る。

これがこのプロジェクトに残された唯一の未検証項目。

なぜこれが本質なのか（Bill Benter の定式化）
--------------------------------------------
Benter は「勝者を当てる」のではなく「市場の確率と真の確率のズレ」を推定した。
彼の結合モデルは条件付きロジットで、市場とモデルの対数確率に自由な係数を張る:

    p_final ∝ q_market^α · p_model^β    （α+β=1 に固定しない）

  β = 0            → モデルは市場に何も足せていない。この設計では勝てない
  β > 0 で有意     → モデルは市場が見落としている情報を持っている
  α < 1            → 市場自身に favorite-longshot bias があり、
                     本命側を強め・穴側を弱めるだけで市場より正確になる
                     （53,694レースの人気帯別回収率で兆候は確認済み）

現行本番は α=1-λ, β=λ=0.25 という根拠のない暫定値。ここで初めて実測する。

必要なもの
----------
snapshot_odds.py が貯めた data/odds.db（全120通りのオッズ）。
公式は過去オッズを公開していないので、これは自分で貯めるしかない。

    python fit_lambda.py --odds-db ../data/odds.db

前半で λ を決め、後半で採点する。後半は一度も見ない。
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import features as F
import model_io
import plackett_luce as PL
from train import load_races

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "kyotei.db"
ODDS_DB = ROOT / "data" / "odds.db"
PAYOUT_RATE = 0.75

# worker/src/selection.ts の DEFAULT_OPTIONS と一致させること
EV_THRESHOLD = 1.05
EV_CEILING = 2.5
MIN_PROB = 0.01
MAX_PROB_RATIO = 4.0
MAX_PICKS = 5


def load_latest_odds(odds_db) -> dict[str, dict[str, float]]:
    """レースごとに締切に最も近いスナップショットを1つ取る。"""
    con = sqlite3.connect(odds_db)
    rows = con.execute("""
        SELECT race_id, odds_json, minutes_left, captured_at
        FROM odds_snapshots ORDER BY race_id, captured_at
    """).fetchall()
    con.close()
    best: dict[str, tuple] = {}
    for rid, js, ml, cap in rows:
        # minutes_left が取れているならそれが最良の基準。
        # 取れていない行は「最後に取れたもの」で代用する。
        key = (0 if ml is None else 1, -(ml if ml is not None else 0), cap)
        if rid not in best or key > best[rid][0]:
            best[rid] = (key, js)
    out = {}
    for rid, (_, js) in best.items():
        try:
            o = json.loads(js)
        except json.JSONDecodeError:
            continue
        if len(o) >= 110:
            out[rid] = o
    return out


def market_probs(odds: dict[str, float]) -> dict[str, float] | None:
    """オッズ -> 市場の確率。Σ(0.75/odds) が1から大きく外れる行は捨てる。"""
    q = {}
    tot = 0.0
    for k, v in odds.items():
        if not v or v <= 1.0:
            continue
        p = PAYOUT_RATE / v
        q[k] = p
        tot += p
    if not (0.85 < tot < 1.20) or len(q) < 110:
        return None
    return {k: v / tot for k, v in q.items()}


def pool(q: dict, p: dict, alpha: float, beta: float) -> dict:
    """Benter式の対数線形結合 s ∝ q^α · p^β（正規化して確率に戻す）。"""
    out = {}
    tot = 0.0
    for k, qv in q.items():
        pv = p.get(k, 1e-9)
        v = math.exp(alpha * math.log(max(qv, 1e-12))
                     + beta * math.log(max(pv, 1e-12)))
        out[k] = v
        tot += v
    return {k: v / tot for k, v in out.items()}


def picks_for(prob: dict, odds: dict, q: dict) -> list[str]:
    """selection.ts と同じ絞り込み。確率降順で最大5点。"""
    cands = []
    for k, p in prob.items():
        o = odds.get(k)
        if not o:
            continue
        ev = p * o
        if ev < EV_THRESHOLD or ev > EV_CEILING:
            continue
        if p < MIN_PROB:
            continue
        qm = q.get(k, 1e-9)
        if qm > 0 and p / qm > MAX_PROB_RATIO:
            continue
        cands.append((p, k))
    cands.sort(reverse=True)
    return [k for _, k in cands[:MAX_PICKS]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "worker" / "src" / "model.json"))
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--odds-db", default=str(ODDS_DB))
    ap.add_argument("--min-races", type=int, default=500)
    a = ap.parse_args()

    if not Path(a.odds_db).exists():
        raise SystemExit(
            f"{a.odds_db} がありません。\n"
            "GitHub Actions の「オッズ収集」ワークフローの成果物 odds-db を\n"
            "ダウンロードして data/odds.db に置いてください。"
        )

    odds_all = load_latest_odds(a.odds_db)
    print(f"オッズのあるレース: {len(odds_all):,}")

    m = model_io.load_model(Path(a.model))
    con = sqlite3.connect(a.db)
    races = [r for r in load_races(con, None, None) if r["race_id"] in odds_all]
    con.close()
    races.sort(key=lambda r: (r["date"], r["jcd"], r["rno"]))
    print(f"結果と突き合わせできたレース: {len(races):,}")

    rows = []
    for r in races:
        q = market_probs(odds_all[r["race_id"]])
        if q is None:
            continue
        feats = F.build_features(r, m.priors)
        p = PL.trifecta_probabilities(m.utilities(feats), m.stage_exponents,
                                      m.stage_calib, feats)
        if not p:
            continue
        rows.append((r, q, p, r["trifecta"], odds_all[r["race_id"]]))

    n = len(rows)
    if n < a.min_races:
        raise SystemExit(
            f"有効なレースが {n} 件しかありません（{a.min_races} 件以上必要）。\n"
            "オッズが貯まるまで待ってください。"
        )
    half = n // 2
    print(f"有効レース {n:,}（前半 {half:,} で λ を決め、後半 {n-half:,} で採点）\n")

    def nll(subset, alpha, beta):
        tot = 0.0
        for _, q, p, win, _ in subset:
            pr = pool(q, p, alpha, beta)
            tot += -math.log(max(pr.get(win, 1e-12), 1e-12))
        return tot / len(subset)

    tr_rows = rows[:half]

    # 1) 従来のλ（α+β=1 に拘束）
    lam_grid = [round(x, 3) for x in np.arange(0.0, 1.001, 0.1)]
    lam_scores = [(nll(tr_rows, 1 - l, l), l) for l in lam_grid]
    _, lam = min(lam_scores)

    # 2) Benter式（α, β を自由に推定）。粗い格子 → 最良点の周りを細かく
    best = (1e9, 1.0, 0.0)
    for a in np.arange(0.6, 1.31, 0.15):
        for bt in np.arange(0.0, 0.61, 0.1):
            v = nll(tr_rows, a, bt)
            if v < best[0]:
                best = (v, float(a), float(bt))
    for a in np.arange(best[1] - 0.1, best[1] + 0.11, 0.05):
        for bt in np.arange(max(0.0, best[2] - 0.08), best[2] + 0.081, 0.02):
            v = nll(tr_rows, a, bt)
            if v < best[0]:
                best = (v, float(a), round(float(bt), 3))
    _, alpha, beta = best

    te = rows[half:]
    print(f"\n[後半で採点]  λ* = {lam} / Benter式 α* = {alpha:.2f}, β* = {beta:.3f}")
    n_mkt = nll(te, 1.0, 0.0)
    n_lam = nll(te, 1 - lam, lam)
    n_ab = nll(te, alpha, beta)
    print(f"  市場のみ (α=1, β=0)        {n_mkt:.4f}")
    print(f"  モデルのみ (α=0, β=1)      {nll(te, 0.0, 1.0):.4f}")
    print(f"  λブレンド (α={1-lam:.2f}, β={lam:.2f}) {n_lam:.4f}")
    print(f"  Benter式  (α={alpha:.2f}, β={beta:.2f}) {n_ab:.4f}")
    gain = n_mkt - min(n_lam, n_ab)
    print(f"\n  市場に対する改善 {gain:+.4f}")
    if beta <= 0.005 and abs(alpha - 1.0) < 0.05:
        print("  → モデルは市場に何も足せておらず、市場の歪みも実用量ではありません。")
    elif beta <= 0.005:
        print("  → モデルの寄与はゼロですが、α<1 なら市場自身の歪み"
              "（favorite-longshot bias）だけで僅かに市場を上回れます。")
    else:
        print("  → モデルは市場が見ていない情報を持っています。")
    if n_ab < n_lam - 1e-4:
        print(f"  → α+β=1 の拘束を外す価値があります（worker/src/selection.ts の"
              f" ブレンドを α={alpha:.2f}, β={beta:.2f} に）")

    # 本番の買い方は最良の組で評価
    use_a, use_b = (alpha, beta) if n_ab <= n_lam else (1 - lam, lam)

    # ---- 本番の買い方をそのまま再現して回収率を出す ----
    inv = ret = 0
    hits = 0
    bets = 0
    bought_races = 0
    per_race = []
    for r, q, p, win, odds in te:
        pr = pool(q, p, use_a, use_b)
        ks = picks_for(pr, odds, q)
        if not ks:
            per_race.append(0.0)
            continue
        bought_races += 1
        bets += len(ks)
        inv += 100 * len(ks)
        pay = r.get("payout") or 0
        got = pay if win in ks else 0
        if win in ks:
            hits += 1
        ret += got
        per_race.append(got - 100 * len(ks))

    print(f"\n[本番の買い方（期待値{EV_THRESHOLD}超・確率順・最大{MAX_PICKS}点）]")
    if inv == 0:
        print("  買える買い目が1つもありませんでした。")
        return
    roi = ret / inv
    # 「収支ちょうど0円の購入レース」を数え落とさないよう、購入時に直接数える
    print(f"  対象 {len(te):,}レース中 {bought_races:,}レースで購入  計 {bets:,}点")
    print(f"  投資 {inv:,}円 / 払戻 {ret:,}円 → 回収率 {roi:.2%}   的中 {hits}レース")

    # ブートストラップ信頼区間。3連単は分散が巨大なので点推定だけ見てはいけない。
    arr = np.array(per_race, dtype=float)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        s = rng.choice(arr, size=len(arr), replace=True)
        boot.append(s.sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  収支 {arr.sum():+,.0f}円   95%信頼区間 [{lo:+,.0f}, {hi:+,.0f}]")
    if lo > 0:
        print("  → 統計的に有意な黒字です。")
    elif hi < 0:
        print("  → 統計的に有意な赤字です。この買い方では負けます。")
    else:
        print("  → まだ運と区別がつきません。判断するには件数が足りません。")
        # 必要件数の目安。3連単は分散が巨大で、数百件では何も言えない。
        need = (2 * arr.std(ddof=1) / max(abs(arr.mean()), 1e-9)) ** 2
        if math.isfinite(need) and need > len(arr):
            print(f"     現在の効果量なら、有意になるまで概算 {need:,.0f} レース必要です。")


if __name__ == "__main__":
    main()
