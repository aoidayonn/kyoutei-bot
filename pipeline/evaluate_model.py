"""モデルを指定期間で評価する（採用ゲート用の対称プロトコル）。

なぜ「対称」か
--------------
旧実装は model.json に記録済みの指標と新モデルの指標を比べていたが、
2つの罠があった:
  1. 検証期間が違う（週次で窓がずれる）→ 期間差を比べてしまう
  2. 温度キャリブレーションの有無・当たり外れが混ざる。実測では
     LogLoss改善0.036のうち0.025が温度の産物で、真の判別力差は0.012だった

そこで新旧**両方**をこのスクリプトで同じ期間・同じ手順で測る:
  - 生スコアに対し温度 t を評価窓で再推定（両モデル同条件）
  - 段階指数 α,β も評価窓で再推定
  - 主指標は商品そのものである **3連単のNLL**（1〜3着の完全対数尤度）

    python evaluate_model.py --model ../worker/src/model.json \
                             --start 2026-05-01 --out /tmp/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

import train as T
import model_io
import stage_calib

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"


def trifecta_nll(V, order):
    """1〜3着のPL対数尤度（2・3着は段階指数を織り込み済みのVを渡すこと…ではなく
    ここでは指数を含めて計算する）。V: (N,6) 温度適用済み生効用。"""
    raise NotImplementedError  # 下の nll3 を使う


def nll3(V, order, exps):
    n = V.shape[0]
    idx = np.arange(n)
    total = 0.0
    for k, p in enumerate(exps):
        mask = np.ones((n, 6), dtype=bool)
        for j in range(k):
            mask[idx, order[:, j]] = False
        Vp = p * V
        Vm = np.where(mask, Vp, -np.inf)
        mx = Vm.max(axis=1, keepdims=True)
        ex = np.where(mask, np.exp(Vm - mx), 0.0)
        total += float(-(Vp[idx, order[:, k]] - (mx[:, 0] + np.log(ex.sum(axis=1)))).mean())
    return total


def calibrate_temperature(S, order):
    def winner_ll(t):
        V = t * S
        m = V.max(axis=1, keepdims=True)
        p = np.exp(V - m)
        p /= p.sum(axis=1, keepdims=True)
        idx = np.arange(V.shape[0])
        return float(-np.log(np.clip(p[idx, order[:, 0]], 1e-12, None)).mean())

    best_t, best = 1.0, float("inf")
    for t in np.arange(0.4, 2.51, 0.05):
        ll = winner_ll(float(t))
        if ll < best:
            best_t, best = float(t), ll
    if best_t <= 0.45 or best_t >= 2.45:
        print(f"⚠️  温度がグリッド端に張り付いています (t={best_t})")
    for t in np.arange(best_t - 0.05, best_t + 0.051, 0.01):
        ll = winner_ll(float(t))
        if ll < best:
            best_t, best = float(t), ll
    return round(best_t, 3), best


def symmetric_metrics(m: "model_io.Model", races):
    """モデルを「出荷された状態のまま」（焼き込み済みの温度・較正）で測る。

    かつては評価窓で温度・較正を再推定していた。理由は「初期の線形モデルが
    温度を持たず、比較が非対称になる」ため。現在は新旧どちらの model.json も
    温度と段階較正一式を焼き込んで出荷されるので、**実際にデプロイされる
    アーティファクト同士**を同じ窓で比べるのが最も公平で、しかも
    較正の質の改善（val→out-of-fold化など）もゲートに正しく反映される。

    再推定プロトコルだと較正はゲート側で毎回作り直されるため、
    「較正の作り方」を良くしても差がゼロに見えてしまう。

    較正を持たない旧形式モデルが来た場合だけ、後方互換として
    評価窓の前半で較正を推定する（後半で採点）。
    """
    X, order, _ = T.build_matrices(races, m.priors)
    S = np.array([m.raw_utilities(rows) for rows in X])
    Xa = np.asarray(X, dtype=float)

    if m.stage_calib:
        # 出荷物そのまま。評価窓のどこにも当てはめを行わない
        V = getattr(m, "temperature", 1.0) * S
        osc = order
        Xs = Xa
        calib = m.stage_calib
        exps = list(m.stage_exponents)
        t = getattr(m, "temperature", 1.0)
    else:
        # 後方互換: 前半で較正、後半で採点
        n = S.shape[0]
        half = n // 2
        t, _ = calibrate_temperature(S[:half], order[:half])
        exps = T.fit_stage_exponents_from_V(t * S[:half], order[:half], verbose=False)
        calib = stage_calib.fit(t * S[:half], Xa[:half], order[:half], verbose=False)
        V = t * S[half:]
        osc = order[half:]
        Xs = Xa[half:]

    # 1着の指標も本番と同じ較正（firstブロック込み）で測る
    sc1 = stage_calib.first_scores(V, Xs, calib)
    mx = sc1.max(axis=1, keepdims=True)
    prob = np.exp(sc1 - mx)
    prob /= prob.sum(axis=1, keepdims=True)
    idx = np.arange(V.shape[0])
    return dict(
        n=int(V.shape[0]),
        win_accuracy=float((prob.argmax(axis=1) == osc[:, 0]).mean()),
        baseline_lane1=float((osc[:, 0] == 0).mean()),
        logloss=float(-np.log(np.clip(prob[idx, osc[:, 0]], 1e-12, None)).mean()),
        # 商品指標。較正あり（本番と同じ展開）を主指標にする。
        trifecta_nll=stage_calib.trifecta_nll(V, Xs, osc, calib),
        # 参考: 枠ペアを使わない旧方式でも測っておく
        trifecta_nll_scalar=nll3(V, osc, exps),
        refit_temperature=t,
        refit_exponents=exps,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--db", default=str(DB))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    model_path = Path(a.model)
    if not model_path.exists():
        print(f"{a.model} がありません。空の指標を書き出します。")
        Path(a.out).write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return

    m = model_io.load_model(model_path)  # feature_names 照合込み

    con = sqlite3.connect(a.db)
    races = T.load_races(con, a.start, a.end)
    con.close()
    if len(races) < 100:
        print(f"検証データが {len(races)} レースしかありません。")
        Path(a.out).write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return

    metrics = symmetric_metrics(m, races)
    print(f"  [{m.type} @ {a.start}〜] N={metrics['n']}  "
          f"3連単NLL {metrics['trifecta_nll']:.4f}  "
          f"1着LL {metrics['logloss']:.4f}  的中率 {metrics['win_accuracy']:.3%}")

    Path(a.out).write_text(
        json.dumps({"metrics": {"test": metrics}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"書き出しました -> {a.out}")


if __name__ == "__main__":
    main()
