"""モデルの読み込みと推論の共通実装。

model.json は2形式を受け付ける:
  - 線形:      {"weights": [...], ...}
  - LightGBM:  {"model_type": "lightgbm_rank", "trees": [...], "temperature": t, ...}

LightGBM の木は依存ライブラリなしの純Pythonで辿る。
これが worker/src/gbm.ts の**唯一の正解仕様**であり、
export_fixture.py がこの実装で期待値を吐き、TS側はそれと一致することを
テストで強制される。

木のフォーマット（dump_model から compact_trees() で変換）:
  tree = {"f": [特徴番号...], "t": [しきい値...], "l": [左子...], "r": [右子...],
          "v": [葉の値...]}
  - ノード i が内部ノードのとき: 特徴 f[i] の値 <= t[i] なら l[i]、そうでなければ r[i] へ
  - 子の値が負のとき: ~child が葉番号（v[~child] が出力への加算値）
  - 特徴量に NaN は来ない前提（features.py が既定値で埋める）
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import features as F


# ---------------------------------------------------------------- 木の変換

def compact_trees(dump: dict) -> list[dict]:
    """lightgbm の dump_model() 出力を、上記のフラット形式へ変換する。"""
    out = []
    for tree_info in dump["tree_info"]:
        f, t, l, r, v = [], [], [], [], []

        def walk(node) -> int:
            if "leaf_value" in node:
                v.append(float(node["leaf_value"]))
                return ~(len(v) - 1)
            # 数値分割(<=)のみ対応。カテゴリ分割が来たら黙って壊れる前に止める
            dt = node.get("decision_type")
            if dt not in (None, "<="):
                raise SystemExit(f"未対応の decision_type: {dt}（categorical_feature は使わないこと）")
            i = len(f)
            f.append(int(node["split_feature"]))
            t.append(float(node["threshold"]))
            l.append(0)
            r.append(0)
            l[i] = walk(node["left_child"])
            r[i] = walk(node["right_child"])
            return i

        walk(tree_info["tree_structure"])
        # しきい値・葉値は丸めない。丸めると境界ちょうどの値で分岐が反転し、
        # lightgbm本体と結果がズレる（実測で最大0.25の誤差が出た）。
        out.append({"f": f, "t": t, "l": l, "r": r, "v": v})
    return out


def eval_tree(tree: dict, x) -> float:
    """1本の木を辿る。"""
    f, t, l, r, v = tree["f"], tree["t"], tree["l"], tree["r"], tree["v"]
    if not f:  # 分割なし（葉1枚）
        return v[0]
    i = 0
    while True:
        i = l[i] if x[f[i]] <= t[i] else r[i]
        if i < 0:
            return v[~i]


def eval_trees(trees: list[dict], x) -> float:
    return sum(eval_tree(tr, x) for tr in trees)


# ---------------------------------------------------------------- モデル

class Model:
    """線形/GBM を同じ顔で扱う。"""

    def __init__(self, data: dict):
        self.data = data
        self.type = data.get("model_type", "linear")
        names = data.get("feature_names")
        if names != F.FEATURE_NAMES:
            raise SystemExit(
                "model.json の feature_names が features.py と一致しません。再学習してください"
            )
        self.priors = {
            "lane_prior": data.get("lane_prior", {}),
            "racer_lane": data.get("racer_lane", {}),
        }
        self.stage_exponents = tuple(data.get("stage_exponents", [1.0, 1.0, 1.0]))
        if self.type == "linear":
            self.w = data["weights"]
        elif self.type == "lightgbm_rank":
            self.trees = data["trees"]
            self.temperature = float(data.get("temperature", 1.0))
        else:
            raise SystemExit(f"未知の model_type: {self.type}")

    def raw_utilities(self, rows) -> list[float]:
        """温度を掛ける前の生スコア（キャリブレーション再推定用）。"""
        if self.type == "linear":
            return [sum(x * w for x, w in zip(row, self.w)) for row in rows]
        return [eval_trees(self.trees, row) for row in rows]

    def utilities(self, rows) -> list[float]:
        """1レース分（6艇の特徴量行列）→ 効用ベクトル（デプロイ時と同じ値）。"""
        if self.type == "linear":
            return self.raw_utilities(rows)
        return [self.temperature * u for u in self.raw_utilities(rows)]

    def win_probs(self, rows) -> list[float]:
        u = self.utilities(rows)
        m = max(u)
        e = [math.exp(x - m) for x in u]
        s = sum(e)
        return [x / s for x in e]


def load_model(path) -> Model:
    return Model(json.loads(Path(path).read_text(encoding="utf-8")))
