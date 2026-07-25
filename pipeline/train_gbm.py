"""LightGBM (lambdarank) の学習と model.json の書き出し。

線形モデル(train.py)との違い:
  - レース単位のランキング損失で学習する（「どの艇が1着か」の相対比較）。
    バイナリ分類より一貫して良かった（検証 LogLoss 1.221 → 1.195）。
  - 出力スコアはそのままでは確率ではないため、温度 t を推定して
    u = t * score を Plackett-Luce の効用として使う。

キャリブレーションの手順（テストセットを触らないため二段構え）:
  1. 学習期間の末尾3か月を val として切り出す
  2. val より前だけで学習した補助モデルで、val 上の温度 t と段階指数 α,β を推定
  3. 本番モデルは学習期間ぜんぶで学習し、t, α, β は 2. の値を流用する
     （スコアのスケールはハイパーパラメータでほぼ決まるため流用できる）

使い方:
    python train_gbm.py --train-end 2026-04-30 --out ../worker/src/model.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import lightgbm as lgb

import features as F
import train as T
import model_io

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
    label_gain=[0, 1],
)
NUM_ROUNDS = 150  # 150本×31葉 ≈ gzip 0.3MB。250本×63葉との差は誤差程度だった


def build(races, priors):
    X, o, _ = T.build_matrices(races, priors)
    return X, o


def flatten(X, order):
    n = X.shape[0]
    Xf = X.reshape(n * 6, X.shape[2])
    y = np.zeros(n * 6)
    y[np.arange(n) * 6 + order[:, 0]] = 1
    return Xf, y


def fit_gbm(X, order):
    Xf, y = flatten(X, order)
    return lgb.train(PARAMS, lgb.Dataset(Xf, y, group=[6] * X.shape[0]),
                     num_boost_round=NUM_ROUNDS)


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
    """u = t * score の t を、1着のLogLoss最小で選ぶ（黄金分割まがいのグリッド）。"""
    best_t, best_ll = 1.0, float("inf")
    for t in np.arange(0.5, 2.01, 0.05):
        ll = winner_logloss(t * S, order)
        if ll < best_ll:
            best_t, best_ll = float(t), ll
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

    # ---- 温度・段階指数のキャリブレーション用に末尾3か月を val に ----
    val_start = (dt.date.fromisoformat(a.train_end) - dt.timedelta(days=90)).isoformat()
    sub = [r for r in train_races if r["date"] < val_start]
    val = [r for r in train_races if r["date"] >= val_start]
    print(f"学習 {len(train_races)} レース（うちキャリブレーション用 val {len(val)}）")

    Xs, os_ = build(sub, priors)
    Xv, ov = build(val, priors)

    print("補助モデルを学習（キャリブレーション用）...")
    aux = fit_gbm(Xs, os_)
    Sv = scores_matrix(aux, Xv)
    temp, val_ll = calibrate_temperature(Sv, ov)
    print(f"  温度 t = {temp}（val LogLoss {val_ll:.4f}）")

    exponents = T.fit_stage_exponents_from_V(temp * Sv, ov)

    # ---- 本番モデルは全学習期間で ----
    print("本番モデルを学習...")
    Xtr, otr = build(train_races, priors)
    booster = fit_gbm(Xtr, otr)

    metrics = {"train": evaluate(temp * scores_matrix(booster, Xtr), otr, "train")}
    if len(test_races) > 100:
        Xte, ote = build(test_races, priors)
        metrics["test"] = evaluate(temp * scores_matrix(booster, Xte), ote, "test")

    # ---- 純Python推論器とのパリティ検証（TS移植の正解仕様） ----
    trees = model_io.compact_trees(booster.dump_model())
    rng = np.random.default_rng(0)
    sample = Xtr[rng.integers(0, Xtr.shape[0], 200)].reshape(-1, Xtr.shape[2])
    expect = booster.predict(sample)
    got = np.array([model_io.eval_trees(trees, row) for row in sample])
    err = float(np.abs(expect - got).max())
    assert err < 1e-6, f"純Python推論器がlightgbmと一致しません (max err {err})"
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
