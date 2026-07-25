"""TypeScript 側とのパリティテスト用に、特徴量の期待値を書き出す。

    python export_fixture.py

worker/test/fixture.json が生成される。
features.py を変更したら必ず再実行し、`npm test` を通すこと。
"""
from __future__ import annotations

import json
from pathlib import Path

import features as F

OUT = Path(__file__).resolve().parent.parent / "worker" / "test" / "fixture.json"

# 実データに近い、決め打ちのサンプルレース
SAMPLE_RACE = {
    "jcd": 22,
    "wind_speed": 3,
    "wave_height": 3,
    "entries": [
        dict(lane=1, racer_id=5009, racer_class="A2", win_rate_national=5.42,
             top2_national=36.00, win_rate_local=5.15, top2_local=37.70,
             motor_top2=44.44, boat_top2=34.62, age=29, weight=57, ex_time=6.92),
        dict(lane=2, racer_id=5155, racer_class="A2", win_rate_national=6.12,
             top2_national=43.86, win_rate_local=6.21, top2_local=47.37,
             motor_top2=44.16, boat_top2=31.58, age=29, weight=45, ex_time=6.86),
        dict(lane=3, racer_id=4966, racer_class="B1", win_rate_national=5.09,
             top2_national=34.00, win_rate_local=6.03, top2_local=37.93,
             motor_top2=37.04, boat_top2=38.27, age=28, weight=54, ex_time=6.93),
        dict(lane=4, racer_id=5145, racer_class="A2", win_rate_national=6.10,
             top2_national=48.00, win_rate_local=5.82, top2_local=54.55,
             motor_top2=43.48, boat_top2=32.00, age=23, weight=55, ex_time=6.87),
        dict(lane=5, racer_id=5032, racer_class="A2", win_rate_national=5.33,
             top2_national=33.63, win_rate_local=0.0, top2_local=0.0,
             motor_top2=31.11, boat_top2=39.47, age=28, weight=52, ex_time=6.98),
        dict(lane=6, racer_id=5303, racer_class="B2", win_rate_national=5.70,
             top2_national=30.86, win_rate_local=4.68, top2_local=22.97,
             motor_top2=36.19, boat_top2=38.57, age=23, weight=51, ex_time=None),
    ],
}

SAMPLE_PRIORS = {
    "lane_prior": {f"22-{i}": round(1.0 - 0.5 * i, 4) for i in range(1, 7)},
    "racer_lane": {"5009-1": 0.25, "5032-5": -0.4},
}


def main():
    rows = F.build_features(SAMPLE_RACE, SAMPLE_PRIORS)

    # 本番モデルがGBMなら、同じ入力に対する期待効用も出力する。
    # TS側の木トラバーサル（gbm.ts）がPython実装と一致することをテストで固定する。
    expected_utilities = None
    model_path = OUT.parent.parent / "src" / "model.json"
    if model_path.exists():
        import model_io
        try:
            m = model_io.load_model(model_path)
            model_rows = F.build_features(SAMPLE_RACE, m.priors)
            expected_utilities = [round(u, 10) for u in m.utilities(model_rows)]
        except SystemExit:
            pass  # 特徴量不一致のときはこの後の再学習で直る
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "feature_names": F.FEATURE_NAMES,
                "race": SAMPLE_RACE,
                "priors": SAMPLE_PRIORS,
                "expected": [[round(x, 10) for x in row] for row in rows],
                "expected_utilities": expected_utilities,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"書き出しました -> {OUT}  ({len(F.FEATURE_NAMES)} 特徴量 × 6艇)")


if __name__ == "__main__":
    main()
