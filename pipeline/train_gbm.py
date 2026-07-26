"""LightGBM (lambdarank) の学習と model.json の書き出し。

線形モデル(train.py)との違い:
  - レース単位のランキング損失で学習する。ラベルは 1着=3, 2着=2, 3着=1。
    「1着のみ=1」で学習すると2・3着の並びの情報が捨てられ、商品指標である
    3連単NLLが線形モデルより悪化した（実測 3.8713 vs 3.8578）。
    3着までラベルに入れると全指標で最良になる（3連単NLL 3.8531）。
  - 出力スコアはそのままでは確率ではないため、温度 t を推定して
    u = t * score を Plackett-Luce の効用として使う。

キャリブレーションの手順（out-of-fold 方式）:
  1. 学習期間を K_FOLDS 個の連続ブロックに分ける
  2. 各ブロックを「残りのブロックで学習したモデル」で採点し、
     学習期間全体の out-of-fold スコアを作る
  3. 温度 t・段階指数・段階較正（first + 枠ペア + 特徴量補正）を
     OOFスコア上で推定する（テスト期間には一切触れない）
  4. 本番モデルは学習期間ぜんぶで学習し、3. の較正を流用する
     （旧方式=末尾3ヶ月のvalに対し、実測で 3連単NLL -0.004〜-0.008）

使い方:
    python train_gbm.py --train-end 2026-04-30 --out ../worker/src/model.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb

import features as F
import train as T
import model_io
import stage_calib as stage_calib_mod

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"

PARAMS = dict(
    objective="lambdarank",
    learning_rate=0.1,
    num_leaves=31,
    feature_fraction=0.9,
    min_data_in_leaf=300,
    verbose=-1,
    num_threads=4,
    seed=7,
    # label_gain は既定（2^label - 1）。ラベル3値なので [0,1,3,7] 相当
)
NUM_ROUNDS = 150  # 150本×31葉 ≈ gzip 0.13MB。250本×63葉との差は誤差程度だった
K_FOLDS = 5       # out-of-fold較正の分割数


def build(races, priors):
    X, o, _ = T.build_matrices(races, priors)
    return X, o


def flatten(X, order):
    """ラベル: 1着=3, 2着=2, 3着=1, 他=0（2・3着の情報を捨てない）。"""
    n = X.shape[0]
    Xf = X.reshape(n * 6, X.shape[2])
    y = np.zeros(n * 6)
    base = np.arange(n) * 6
    y[base + order[:, 0]] = 3
    y[base + order[:, 1]] = 2
    y[base + order[:, 2]] = 1
    return Xf, y


def fit_gbm(X, order):
    Xf, y = flatten(X, order)
    rounds = 30 if os.environ.get("KYOTEI_SMOKE") == "1" else NUM_ROUNDS
    return lgb.train(PARAMS, lgb.Dataset(Xf, y, group=[6] * X.shape[0]),
                     num_boost_round=rounds)


def scores_matrix(booster, X):
    n = X.shape[0]
    return booster.predict(X.reshape(n * 6, X.shape[2])).reshape(n, 6)


def winner_logloss(V, order):
    m = V.max(axis=1, keepdims=True)
    p = np.exp(V - m)
    p /= p.sum(axis=1, keepdims=True)
    idx = np.arange(V.shape[0])
    return float(-np.log(np.clip(p[idx, order[:, 0]], 1e-12, None)).mean())


def calibrate_temperature(S, order):
    """u = t * score の t を、1着のLogLoss最小で選ぶ。"""
    best_t, best_ll = 1.0, float("inf")
    for t in np.arange(0.4, 2.51, 0.05):
        ll = winner_logloss(t * S, order)
        if ll < best_ll:
            best_t, best_ll = float(t), ll
    if best_t <= 0.45 or best_t >= 2.45:
        print(f"⚠️  温度がグリッド端に張り付いています (t={best_t})。スコアのスケールを確認すること")
    for t in np.arange(best_t - 0.05, best_t + 0.051, 0.01):
        ll = winner_logloss(t * S, order)
        if ll < best_ll:
            best_t, best_ll = float(t), ll
    return round(best_t, 3), best_ll


def evaluate(V, order, label):
    n = V.shape[0]
    m = V.max(axis=1, keepdims=True)
    p = np.exp(V - m)
    p /= p.sum(axis=1, keepdims=True)
    idx = np.arange(n)
    ll = float(-np.log(np.clip(p[idx, order[:, 0]], 1e-12, None)).mean())
    acc = float((p.argmax(axis=1) == order[:, 0]).mean())
    base = float((order[:, 0] == 0).mean())
    print(f"  [{label}] N={n}  1着的中率 {acc:.3%} (1号艇固定 {base:.3%})  LogLoss {ll:.4f}")
    return dict(n=n, win_accuracy=acc, baseline_lane1=base, logloss=ll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", required=True)
    p.add_argument("--train-start", default=None)
    p.add_argument("--db", default=str(DB))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                        / "worker" / "src" / "model.json"))
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    train_races = T.load_races(con, a.train_start, a.train_end)
    test_races = T.load_races(con, _next_day(a.train_end), None)
    con.close()

    if len(train_races) < 10000:
        raise SystemExit(f"学習データが足りません（{len(train_races)}件）")

    train_races = T.exclude_nonstarter_races(train_races)

    # 推論用テーブル（全学習期間 = 推論時にはすべて過去）
    lane_prior = T.compute_lane_prior(train_races)
    racer_lane = T.compute_racer_lane(train_races)
    priors = {"lane_prior": lane_prior, "racer_lane": racer_lane}
    # 学習特徴量は各レースより前の実績のみ（リーク防止）
    T.annotate_racer_lane_expanding(train_races)

    # ---- 行列は一度だけ作り、日付でスライスする ----
    # 以前は sub/val/full を別々に build しており、ピークメモリが約2.8倍
    # （実データで3.4GB）になって小さいマシンではOOM killされた。
    # races は日付順なので、行列も日付順に並ぶ。
    train_races.sort(key=lambda r: (r["date"], r["jcd"], r["rno"]))
    Xtr, otr = build(train_races, priors)

    # ---- キャリブレーションは out-of-fold 方式 ----
    # 以前は「末尾3ヶ月をvalに、その前で補助モデルを学習」して較正していたが、
    # 較正パラメータ（first + 枠ペア行列 + w、260個超）を3ヶ月分だけで
    # 推定するのはノイズが大きい。学習期間をK個の連続ブロックに分け、
    # 各ブロックを「残りで学習したモデル」で採点した out-of-fold スコアなら、
    # 学習期間**全体**を較正に使える。
    # 実測: 3連単NLL -0.0076（窓1）/ -0.0043（窓2）。
    # コストは学習K+1回ぶんだが、CIのランナーなら1回50秒程度。
    smoke = os.environ.get("KYOTEI_SMOKE") == "1"  # サンドボックス検証用の縮小モード
    n_races = Xtr.shape[0]
    print(f"学習 {n_races} レース（較正は {K_FOLDS} 分割の out-of-fold）")
    fold_bounds = np.array_split(np.arange(n_races), K_FOLDS)
    if any(len(f) < 2000 for f in fold_bounds) and not smoke:
        raise SystemExit("foldが小さすぎます。学習期間を延ばしてください")

    S_oof = np.zeros((n_races, 6))
    for i, fold in enumerate(fold_bounds):
        mask = np.ones(n_races, dtype=bool)
        mask[fold] = False
        bst_f = fit_gbm(Xtr[mask], otr[mask])
        S_oof[fold] = scores_matrix(bst_f, Xtr[fold])
        print(f"  fold {i+1}/{K_FOLDS} 完了", flush=True)
    del bst_f

    temp, oof_ll = calibrate_temperature(S_oof, otr)
    print(f"  温度 t = {temp}（OOF LogLoss {oof_ll:.4f}）")
    exponents = T.fit_stage_exponents_from_V(temp * S_oof, otr)

    print("段階較正（first + 枠ペア + 特徴量補正）を全学習期間で推定...")
    calib_steps = dict(steps_first=120, steps_stage=100) if smoke else {}
    stage_calib = stage_calib_mod.fit(temp * S_oof, Xtr, otr, **calib_steps)
    del S_oof

    # ---- 本番モデルは全学習期間で ----
    print("本番モデルを学習...")
    booster = fit_gbm(Xtr, otr)

    metrics = {"train": evaluate(temp * scores_matrix(booster, Xtr), otr, "train")}
    if len(test_races) > 100:
        Xte, ote = build(test_races, priors)
        metrics["test"] = evaluate(temp * scores_matrix(booster, Xte), ote, "test")

    # ---- 純Python推論器とのパリティ検証（TS移植の正解仕様） ----
    trees = model_io.compact_trees(booster.dump_model())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, Xtr.shape[0], 300)
    sample = Xtr[idx].reshape(-1, Xtr.shape[2])
    expect = booster.predict(sample)
    got = np.array([model_io.eval_trees(trees, row) for row in sample])
    err = float(np.abs(expect - got).max())
    # assert は python -O で消えるため使わない
    if err >= 1e-9:
        raise SystemExit(f"純Python推論器がlightgbmと一致しません (max err {err})")
    print(f"木の変換パリティ OK（{len(trees)}本, max err {err:.2e}）")

    model = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "lightgbm_rank",
        "train_period": [a.train_start or "all", a.train_end],
        "n_train_races": len(train_races),
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "num_rounds": NUM_ROUNDS,
        "temperature": temp,
        "feature_names": F.FEATURE_NAMES,
        "lane_prior": {k: round(v, 6) for k, v in lane_prior.items()},
        "racer_lane": racer_lane,
        "stage_exponents": exponents,
        "stage_calib": stage_calib,
        "metrics": metrics,
        "trees": trees,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"書き出しました -> {out}  ({out.stat().st_size/1e6:.1f}MB)")


def _next_day(d: str) -> str:
    return (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()


if __name__ == "__main__":
    main()
