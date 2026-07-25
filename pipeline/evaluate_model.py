"""既存のモデルを、指定した期間で評価し直す。

なぜ必要か
----------
model.json に記録されている指標は「そのモデルを学習したときの検証期間」で
測ったもの。週次で再学習すると検証期間が毎回ずれるため、
記録済みの数字と新しい数字をそのまま比べると、
モデルの良し悪しではなく**期間の違い**を比べてしまう。

そこで採用判定の前に、現行モデルを新しい検証期間で測り直して、
同じ土俵で比較できるようにする。

    python evaluate_model.py --model ../worker/src/model.json \
                             --start 2026-04-26 --out /tmp/current-metrics.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

import train as T

DB = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--start", required=True, help="検証期間の開始日 YYYY-MM-DD")
    p.add_argument("--end", default=None)
    p.add_argument("--db", default=str(DB))
    p.add_argument("--out", required=True)
    a = p.parse_args()

    model_path = Path(a.model)
    if not model_path.exists():
        print(f"{a.model} がありません。比較対象なしとして空の指標を書き出します。")
        Path(a.out).write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return

    model = json.loads(model_path.read_text(encoding="utf-8"))
    w = np.array(model["weights"])
    priors = {
        "lane_prior": model.get("lane_prior", {}),
        "racer_lane": model.get("racer_lane", {}),
    }

    con = sqlite3.connect(a.db)
    races = T.load_races(con, a.start, a.end)
    con.close()

    if len(races) < 100:
        print(f"検証データが {len(races)} レースしかありません。比較を諦めます。")
        Path(a.out).write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        return

    X, order, _ = T.build_matrices(races, priors)
    metrics = T.evaluate(w, X, order, f"現行モデル @ {a.start}〜")

    Path(a.out).write_text(
        json.dumps({"metrics": {"test": metrics}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"書き出しました -> {a.out}")


if __name__ == "__main__":
    main()
