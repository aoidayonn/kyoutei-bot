"""Plackett-Luce モデルの最尤推定。

3連単を当てたいので、「1着だけ」ではなく「1〜3着の並び全体」の尤度を最大化する。

対数尤度:
    LL = Σ_races Σ_{k=1..3} [ v_{σ(k)} − log Σ_{j ∈ 残り} exp(v_j) ]

勾配:
    ∂LL/∂w = Σ [ x_{σ(k)} − Σ_{j∈残り} p_j · x_j ]

使い方:
    python train.py --train-end 2026-03-31 --out ../worker/src/model.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import features as F

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"


# ------------------------------------------------------------------ データ読み込み

def load_races(con, start=None, end=None):
    """(race_meta, entries) のリストを返す。1〜3着が確定しているレースのみ。"""
    q = """
        SELECT r.race_id, r.date, r.jcd, r.rno, r.wind_speed, r.wave_height,
               r.trifecta, r.trifecta_payout, r.trifecta_pop,
               e.lane, e.racer_class, e.win_rate_national, e.top2_national,
               e.win_rate_local, e.top2_local, e.motor_top2, e.boat_top2,
               e.age, e.weight, e.exhibition_time, e.rank, e.racer_id
        FROM races r JOIN entries e ON e.race_id = r.race_id
        WHERE r.trifecta IS NOT NULL
    """
    params = []
    if start:
        q += " AND r.date >= ?"
        params.append(start)
    if end:
        q += " AND r.date <= ?"
        params.append(end)
    q += " ORDER BY r.date, r.jcd, r.rno, e.lane"

    grouped = defaultdict(list)
    meta = {}
    for row in con.execute(q, params):
        rid = row[0]
        meta[rid] = dict(race_id=rid, date=row[1], jcd=row[2], rno=row[3],
                         wind_speed=row[4], wave_height=row[5],
                         trifecta=row[6], payout=row[7], trifecta_pop=row[8])
        grouped[rid].append(dict(
            lane=row[9], racer_class=row[10],
            win_rate_national=row[11], top2_national=row[12],
            win_rate_local=row[13], top2_local=row[14],
            motor_top2=row[15], boat_top2=row[16],
            age=row[17], weight=row[18], ex_time=row[19], rank=row[20],
            racer_id=row[21],
        ))

    out = []
    for rid, entries in grouped.items():
        if len(entries) != 6:
            continue
        m = meta[rid]
        m["entries"] = sorted(entries, key=lambda e: e["lane"])
        out.append(m)
    return out


def compute_lane_prior(races) -> dict[str, float]:
    """場×枠ごとの1着率を対数オッズにした事前値（学習期間のみから算出）。"""
    win = defaultdict(int)
    cnt = defaultdict(int)
    for r in races:
        first = int(r["trifecta"].split("-")[0])
        for lane in range(1, 7):
            cnt[(r["jcd"], lane)] += 1
        win[(r["jcd"], first)] += 1

    prior = {}
    for key, n in cnt.items():
        # ラプラス平滑化（全国平均の1/6を事前分布として弱く混ぜる）
        p = (win[key] + 1.0) / (n + 6.0)
        prior[f"{key[0]}-{key[1]}"] = math.log(p / (1 - p))
    return prior


def compute_racer_lane(races, min_races=5, shrink=15.0) -> dict[str, float]:
    """選手×枠の「平均からの上振れ量」を縮小推定する。

    「1枠でも弱い選手」「5枠でも強い選手」をモデルに教えるための特徴量。
    サンプルが少ない選手が極端な値を取らないよう、その枠の全体平均に向けて縮小する
    （ベイズ的な縮小推定）。学習期間のデータだけから作るので未来の情報は入らない。
    """
    win = defaultdict(int)
    cnt = defaultdict(int)
    lane_win = defaultdict(int)
    lane_cnt = defaultdict(int)

    for r in races:
        first = int(r["trifecta"].split("-")[0])
        for e in r["entries"]:
            rid, lane = e.get("racer_id"), e["lane"]
            lane_cnt[lane] += 1
            if lane == first:
                lane_win[lane] += 1
            if rid:
                cnt[(rid, lane)] += 1
                if lane == first:
                    win[(rid, lane)] += 1

    base = {lane: lane_win[lane] / lane_cnt[lane] for lane in lane_cnt}

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    table = {}
    for (rid, lane), n in cnt.items():
        if n < min_races:
            continue
        b = base[lane]
        p = (win[(rid, lane)] + shrink * b) / (n + shrink)
        edge = logit(p) - logit(b)
        if abs(edge) >= 0.01:
            table[f"{rid}-{lane}"] = round(edge, 4)
    return table


def annotate_racer_lane_expanding(races, min_races=5, shrink=15.0):
    """学習用: 各レースの選手×枠エッジを「そのレースより前の結果」だけから計算し、
    e["racer_lane_edge"] に埋め込む（features.py が優先して使う）。

    compute_racer_lane は学習期間全体の1着実績からテーブルを作るため、
    そのまま学習の特徴量に使うと「そのレース自身の結果」が説明変数に混ざる
    （ターゲットエンコーディングのインサンプルリーク）。実測では選手×枠の
    判別力が約26%水増しされ、重みが過大に学習されていた。

    推論用のテーブル（model.json に載せる方）は従来どおり全期間で作ってよい。
    推論時点では学習期間ぜんぶが「過去」だから。
    """
    win = defaultdict(int)
    cnt = defaultdict(int)
    lane_win = defaultdict(int)
    lane_cnt = defaultdict(int)

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    for r in sorted(races, key=lambda x: (x["date"], x["jcd"], x["rno"])):
        first = int(r["trifecta"].split("-")[0])

        # まず「ここまでの実績」で特徴量を埋める
        for e in r["entries"]:
            rid, lane = e.get("racer_id"), e["lane"]
            n = cnt[(rid, lane)] if rid else 0
            if rid and n >= min_races and lane_cnt[lane] > 0:
                b = lane_win[lane] / lane_cnt[lane]
                p = (win[(rid, lane)] + shrink * b) / (n + shrink)
                e["racer_lane_edge"] = round(logit(p) - logit(b), 4)
            else:
                e["racer_lane_edge"] = 0.0

        # その後にこのレースの結果を実績へ反映する
        for e in r["entries"]:
            rid, lane = e.get("racer_id"), e["lane"]
            lane_cnt[lane] += 1
            if lane == first:
                lane_win[lane] += 1
            if rid:
                cnt[(rid, lane)] += 1
                if lane == first:
                    win[(rid, lane)] += 1


def exclude_nonstarter_races(races):
    """展示タイムの無い艇（=欠場・出走取消）を含むレースを学習から外す。

    Kファイルで展示タイムが空の艇は「そのレースを走らなかった艇」であり、
    ほぼ確実に勝たない。これを ex_missing 特徴量として学習すると
    「結果を説明変数にする」完全分離が起き、重みが -5 台まで暴走していた。
    本番では ex_missing は「直前情報が未公開」という別の意味になるため、
    学習データから欠場レースごと取り除いて特徴量を無害化する。
    """
    kept = [
        r for r in races
        if all(
            e.get("ex_time") is not None and 5.0 <= float(e["ex_time"]) <= 9.0
            for e in r["entries"]
        )
    ]
    dropped = len(races) - len(kept)
    if dropped:
        print(f"  欠場艇を含む {dropped:,} レースを学習から除外（完全分離の防止）")
    return kept


# ------------------------------------------------------------------ 行列化

def build_matrices(races, priors):
    """X: (N, 6, F) 、 order: (N, 3) 各着順の艇インデックス(0..5)"""
    X, order, keep = [], [], []
    for r in races:
        try:
            first, second, third = (int(x) - 1 for x in r["trifecta"].split("-"))
        except ValueError:
            continue
        rows = F.build_features(r, priors)
        X.append(rows)
        order.append([first, second, third])
        keep.append(r)
    return np.array(X, dtype=np.float64), np.array(order, dtype=np.int64), keep


# ------------------------------------------------------------------ 最適化

def neg_loglik(w, X, order, l2):
    """負の対数尤度とその勾配。"""
    N = X.shape[0]
    V = X @ w                          # (N, 6)
    grad = np.zeros_like(w)
    ll = 0.0
    mask = np.ones((N, 6), dtype=bool)
    idx = np.arange(N)

    for k in range(3):
        chosen = order[:, k]
        Vm = np.where(mask, V, -np.inf)
        mx = Vm.max(axis=1, keepdims=True)
        ex = np.where(mask, np.exp(Vm - mx), 0.0)
        denom = ex.sum(axis=1, keepdims=True)
        ll += float((V[idx, chosen] - (mx[:, 0] + np.log(denom[:, 0]))).sum())

        p = ex / denom                                     # (N, 6)
        exp_x = np.einsum("nj,njf->nf", p, X)              # 期待特徴量
        grad += X[idx, chosen] .sum(axis=0) - exp_x.sum(axis=0)

        mask[idx, chosen] = False

    # 正則化はレース数で割らない。以前は ll と一緒に /N されていたため
    # 実効強度が l2/N（N=15万なら 6e-6）となり、事実上の無正則化だった。
    # これが完全分離した特徴量の重みを -5 台まで暴走させていた。
    return -ll / N + l2 * float(w @ w), -grad / N + 2 * l2 * w


def fit_stage_exponents(w, X, order, verbose=True):
    """2着・3着の「効きの強さ」を別パラメータで推定する。

    素の Plackett-Luce は「1着を除いた残りの中でも強さの比が同じ」と仮定するが、
    現実のボートレースはそうならない。1着が決まった後の2・3着争いは
    もっと横並びに近い。この仮定のズレを放置すると穴目の確率が過大評価され、
    「期待値が高い買い目」が万券ばかりになってしまう。

    そこで各段階に指数 α, β を入れる:
        P(j が2着) ∝ exp(α · v_j)
        P(k が3着) ∝ exp(β · v_k)
    α, β < 1 なら「2着以降は差がつきにくい」ことを意味する。
    """
    V = X @ w
    N = X.shape[0]
    idx = np.arange(N)

    def stage_ll(power, stage):
        mask = np.ones((N, 6), dtype=bool)
        for k in range(stage):
            mask[idx, order[:, k]] = False
        chosen = order[:, stage]
        Vp = power * V
        Vm = np.where(mask, Vp, -np.inf)
        mx = Vm.max(axis=1, keepdims=True)
        ex = np.where(mask, np.exp(Vm - mx), 0.0)
        return float((Vp[idx, chosen] - (mx[:, 0] + np.log(ex.sum(axis=1)))).mean())

    exponents = [1.0]
    for stage in (1, 2):
        best, best_ll = 1.0, -1e18
        for p in np.arange(0.30, 1.51, 0.05):
            ll = stage_ll(p, stage)
            if ll > best_ll:
                best, best_ll = float(p), ll
        # 粗いグリッドの周りを細かく探す
        for p in np.arange(best - 0.05, best + 0.051, 0.01):
            ll = stage_ll(float(p), stage)
            if ll > best_ll:
                best, best_ll = float(p), ll
        exponents.append(round(best, 3))

    if verbose:
        print(f"  段階指数: 1着=1.0  2着={exponents[1]}  3着={exponents[2]}")
    return exponents


def fit(X, order, l2=1.0, verbose=True):
    w0 = np.zeros(X.shape[2])
    res = minimize(neg_loglik, w0, args=(X, order, l2), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500})
    if verbose:
        print(f"  収束: {res.success}  反復 {res.nit}  平均NLL {res.fun:.4f}")
    return res.x


# ------------------------------------------------------------------ 評価

def evaluate(w, X, order, label=""):
    V = X @ w
    N = X.shape[0]
    ex = np.exp(V - V.max(axis=1, keepdims=True))
    p1 = ex / ex.sum(axis=1, keepdims=True)

    top1 = p1.argmax(axis=1)
    hit1 = float((top1 == order[:, 0]).mean())

    # 1着のlog loss（人気順ではなくモデルの確率）
    logloss = float(-np.log(np.clip(p1[np.arange(N), order[:, 0]], 1e-12, None)).mean())

    # ベースライン: 常に1号艇
    base = float((order[:, 0] == 0).mean())
    print(f"  [{label}] N={N}  1着的中率 {hit1:.3%} (1号艇固定 {base:.3%})  LogLoss {logloss:.4f}")
    return dict(n=N, win_accuracy=hit1, baseline_lane1=base, logloss=logloss)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-end", required=True, help="学習に使う最終日 YYYY-MM-DD")
    p.add_argument("--train-start", default=None)
    p.add_argument("--l2", type=float, default=0.0001,
                   help="L2正則化。平均対数尤度に対する係数（データ量に依存しない尺度）")
    p.add_argument("--db", default=str(DB))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                        / "worker" / "src" / "model.json"))
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    train = load_races(con, a.train_start, a.train_end)
    test = load_races(con, _next_day(a.train_end), None)
    con.close()

    print(f"学習 {len(train)} レース / 検証 {len(test)} レース")
    if len(train) < 500:
        raise SystemExit("学習データが少なすぎます。download.py と build_db.py を先に実行してください。")

    train = exclude_nonstarter_races(train)

    lane_prior = compute_lane_prior(train)
    # 推論用テーブルは全学習期間から（推論時には全部が過去）
    racer_lane = compute_racer_lane(train)
    # 学習用の特徴量は各レースより前の実績のみから（リーク防止）
    annotate_racer_lane_expanding(train)
    priors = {"lane_prior": lane_prior, "racer_lane": racer_lane}
    print(f"選手×枠テーブル: {len(racer_lane)} 件")

    Xtr, otr, _ = build_matrices(train, priors)
    print("学習中...")
    w = fit(Xtr, otr, a.l2)
    exponents = fit_stage_exponents(w, Xtr, otr)

    metrics = {"train": evaluate(w, Xtr, otr, "train")}
    if len(test) > 100:
        Xte, ote, _ = build_matrices(test, priors)
        metrics["test"] = evaluate(w, Xte, ote, "test")

    print("\n重み（絶対値の大きい順）:")
    for name, val in sorted(zip(F.FEATURE_NAMES, w), key=lambda t: -abs(t[1]))[:12]:
        print(f"  {name:<22} {val:+.4f}")

    model = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_period": [a.train_start or "all", a.train_end],
        "n_train_races": len(train),
        "feature_names": F.FEATURE_NAMES,
        "weights": [round(float(x), 6) for x in w],
        "lane_prior": {k: round(v, 6) for k, v in lane_prior.items()},
        "racer_lane": racer_lane,
        "stage_exponents": exponents,
        "metrics": metrics,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nモデルを書き出しました -> {out}")


def _next_day(d: str) -> str:
    import datetime as _dt
    return (_dt.date.fromisoformat(d) + _dt.timedelta(days=1)).isoformat()


if __name__ == "__main__":
    main()
