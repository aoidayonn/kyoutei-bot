"""バックテスト: 学習していない期間で的中率と回収率を検証する。

回収率の計算にはオッズが必要だが、公式の過去オッズは公開されていない。
そこで **3連単の払戻金** を使う。払戻金 = 100円が何円になったか なので、
的中したレースの払戻金の合計 ÷ 投資額 で回収率が正確に出せる。

未的中の組み合わせのオッズは分からないため、「期待値>1.0の買い目だけ買う」戦略の
評価には近似が必要。ここでは以下の2つを併記する:

  戦略A（確率上位N点）  : 予測確率の高い順にN点を100円ずつ購入
  戦略B（人気薄フィルタ）: 戦略Aのうち、モデル確率が市場人気より高い買い目のみ購入
                           （市場人気の代理として trifecta_pop を使用）

使い方:
    python backtest.py --model ../worker/src/model.json --start 2026-04-01
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

import features as F
import model_io
from plackett_luce import trifecta_probabilities
from train import load_races

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"


def predict_race(race, model: "model_io.Model"):
    rows = F.build_features(race, model.priors)
    v = model.utilities(rows)
    # 本番（worker/src/predict.ts）と同じ展開にする。
    # 較正を渡し忘れると「バックテストだけ古いモデル」を測ることになる。
    return trifecta_probabilities(v, model.stage_exponents, model.stage_calib, rows)


def run(model_path, start, end, n_points, db=DB):
    model = model_io.load_model(model_path)  # feature_names の照合込み

    # 学習期間と検証期間が重なっているとインサンプル評価になり数字が甘く出る
    tp = model.data.get("train_period")
    if tp and start and str(start) <= str(tp[1]):
        print(f"⚠️  検証開始日 {start} が学習期間（〜{tp[1]}）と重なっています。"
              f"この数字は楽観的に偏ります。\n")

    con = sqlite3.connect(db)
    races = load_races(con, start, end)
    con.close()
    if not races:
        raise SystemExit("対象期間にデータがありません")

    print(f"検証対象: {len(races)} レース ({races[0]['date']} 〜 {races[-1]['date']})\n")

    stats = {
        "A": dict(bet=0, hit=0, ret=0, races=0),
        "B": dict(bet=0, hit=0, ret=0, races=0),
    }
    top1_hit = 0
    lane1_hit = 0
    prob_of_actual = []
    combo_counter = Counter()

    for r in races:
        probs = predict_race(r, model)
        if not probs:
            continue
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        actual = r["trifecta"]
        payout = r["payout"] or 0

        prob_of_actual.append(probs.get(actual, 1e-9))
        if ranked[0][0] == actual:
            top1_hit += 1
        if actual.startswith("1-"):
            lane1_hit += 1
        combo_counter[ranked[0][0]] += 1

        # 戦略A: 確率上位 n_points 点
        picks_a = ranked[:n_points]
        stats["A"]["races"] += 1
        stats["A"]["bet"] += 100 * len(picks_a)
        for combo, _p in picks_a:
            if combo == actual:
                stats["A"]["hit"] += 1
                stats["A"]["ret"] += payout

        # 戦略B: モデルが市場より強気な買い目に絞る
        # 市場の見立ては「何番人気だったか」で近似する。
        # モデル順位が市場人気順位より上 = モデルの方が高く評価している = 妙味あり。
        # ※ 人気順位は的中した組しか分からないため、外れた買い目は
        #   「市場でも上位人気だった」とみなして除外する保守的な近似にしている。
        market_rank = r.get("trifecta_pop") or 999
        picks_b = []
        for rank_i, (combo, p) in enumerate(ranked[:n_points], start=1):
            if combo == actual and rank_i >= market_rank:
                continue  # 市場の方が強気だった = 妙味なし
            picks_b.append((combo, p))
        stats["B"]["races"] += 1
        stats["B"]["bet"] += 100 * len(picks_b)
        for combo, _p in picks_b:
            if combo == actual:
                stats["B"]["hit"] += 1
                stats["B"]["ret"] += payout

    n = len(races)
    print("── 基礎指標 ─────────────────────────────")
    print(f"  最有力1点の的中率      {top1_hit / n:>7.2%}")
    print(f"  1号艇が1着だった割合   {lane1_hit / n:>7.2%}")
    print(f"  的中組の平均予測確率   {np.mean(prob_of_actual):>7.2%}")
    print(f"  幾何平均(LogLoss相当)  {-np.mean(np.log(prob_of_actual)):>7.4f}")
    print(f"  最有力に選ばれた組の種類 {len(combo_counter)} 通り / 120")
    top5 = combo_counter.most_common(5)
    print("  最有力の内訳: " + ", ".join(f"{c}({v})" for c, v in top5))

    print("\n── 購入戦略 ─────────────────────────────")
    for key, label in (("A", f"モデル素の確率上位{n_points}点（市場ブレンドなし）"),
                       ("B", "うちモデルが市場より強気な買い目のみ（悲観側の下限）")):
        s = stats[key]
        if s["bet"] == 0:
            continue
        roi = s["ret"] / s["bet"]
        hit_rate = s["hit"] / s["races"]
        print(f"  [{key}] {label}")
        print(f"       投資 {s['bet']:,}円 / 払戻 {s['ret']:,}円 → 回収率 {roi:>7.2%}")
        print(f"       レース的中率 {hit_rate:.2%}  ({s['hit']}/{s['races']})")

    print("\n※ 控除率25%のため、無作為に買うと回収率は約75%に収束します。")
    print("※ この数字は本番の買い方（市場オッズとのブレンド+期待値フィルタ）とは別物です。")
    print("  本番戦略の回収率は、snapshot_odds.py で貯めたオッズがないと計算できません。")
    print("※ 戦略Bは「外れた買い目は全部買い、当たった買い目だけ条件で除外する」という")
    print("  作りなので、真の回収率の**下限**です。実力はAとBの間のどこかにあります。")
    print("  正確に測るには締切前のオッズが全120通り必要なので、")
    print("  snapshot_odds.py で今日からオッズを貯めてください（数週間で検証可能になります）。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(__file__).resolve().parent.parent
                                          / "worker" / "src" / "model.json"))
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--points", type=int, default=5)
    p.add_argument("--db", default=str(DB))
    a = p.parse_args()
    run(a.model, a.start, a.end, a.points, a.db)


if __name__ == "__main__":
    main()
