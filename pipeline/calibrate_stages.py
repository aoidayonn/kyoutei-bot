"""既存の model.json に、2・3着の較正（枠ペア相互作用）を後から足す／貼り直す。

再学習は不要。木はそのままで、展開のしかただけを直近データに合わせる。

    python calibrate_stages.py --start 2026-05-01

手順:
  1. 指定期間を前半・後半に割り、**前半だけで**較正パラメータを推定して
     後半で採点する（改善が本物かの確認。ここで悪化したら書き込まない）
  2. 確認が取れたら期間全体で推定し直して model.json に書き込む

注意: 使った期間はもう「較正に使ったデータ」なので、
その期間に対する backtest.py の数字はインサンプル寄りになる。
週次の再学習が回れば、学習期間内の val で較正する本来の流れに戻る。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

import train as T
import model_io
import stage_calib as SC

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"
PAYOUT_RATE = 0.75


def scalar_nll(V, order, exponents):
    n = V.shape[0]
    idx = np.arange(n)
    mask = np.ones((n, 6), dtype=bool)
    tot = 0.0
    for k, p in enumerate(exponents):
        nll, _ = SC._softmax_nll(p * V, mask, order[:, k])
        tot += nll
        mask = mask.copy()
        mask[idx, order[:, k]] = False
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent
                                           / "worker" / "src" / "model.json"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--dry-run", action="store_true", help="測るだけで書き込まない")
    a = ap.parse_args()

    path = Path(a.model)
    m = model_io.load_model(path)

    tp = m.data.get("train_period")
    if tp and str(a.start) <= str(tp[1]):
        raise SystemExit(
            f"較正期間 {a.start} が学習期間（〜{tp[1]}）と重なっています。\n"
            "インサンプルの効用で較正すると分布の広がりを読み違えます。"
        )

    con = sqlite3.connect(a.db)
    races = T.load_races(con, a.start, a.end)
    con.close()
    if len(races) < 2000:
        raise SystemExit(f"レースが {len(races)} 件しかありません（2000件以上必要）")

    X, order, _ = T.build_matrices(races, m.priors)
    Xa = np.asarray(X, dtype=float)
    V = np.array([m.utilities(rows) for rows in X])
    n = len(races)
    half = n // 2

    print(f"対象 {n:,} レース（{a.start} 〜 {a.end or '最新'}）")
    print(f"\n[1] 前半 {half:,} で推定 → 後半 {n-half:,} で採点（改善が本物かの確認）")
    calib_half = SC.fit(V[:half], Xa[:half], order[:half], verbose=False)
    old = scalar_nll(V[half:], order[half:], m.stage_exponents)
    new = SC.trifecta_nll(V[half:], Xa[half:], order[half:], calib_half)

    payout = np.array([r.get("payout") or 0 for r in races], dtype=float)
    ok = payout[half:] > 100
    mkt = float(-np.log(PAYOUT_RATE / (payout[half:][ok] / 100.0)).mean())

    print(f"    市場                 {mkt:.4f}")
    print(f"    現行（段階指数のみ）    {old:.4f}   市場との差 {old-mkt:+.4f}")
    print(f"    枠ペア較正あり         {new:.4f}   市場との差 {new-mkt:+.4f}")
    print(f"    改善 {old-new:+.4f}")

    if new >= old:
        raise SystemExit("\n改善しなかったので書き込みません。")

    if a.dry_run:
        print("\n--dry-run なので書き込みません。")
        return

    print(f"\n[2] 期間全体 {n:,} で推定し直して書き込む")
    calib = SC.fit(V, Xa, order)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stage_calib"] = calib
    data["stage_calib_period"] = [a.start, a.end or str(races[-1]["date"])]
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"書き込みました -> {path}  ({path.stat().st_size/1e6:.1f}MB)")
    print("\n次に必ず: python export_fixture.py && (cd ../worker && npm test)")


if __name__ == "__main__":
    main()
