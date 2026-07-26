"""特徴量の定義。

重要: この特徴量の並びは worker/src/features.ts と**完全に一致**させること。
     ズレると学習した重みが意味をなさなくなる。

設計上の注意
------------
* Plackett-Luce / softmax はレース内の定数シフトに不変なので、
  「レース内平均との差」を明示的に作る必要はない（自動的に相殺される）。
* スタートタイミング(実際のST)やレースタイムは**結果**なので特徴量に入れない（リーク）。
* 展示タイムは締切前に公開されるので使ってよい。
"""
from __future__ import annotations

FEATURE_NAMES = [
    # 枠番ダミー（1〜6）
    "lane_1", "lane_2", "lane_3", "lane_4", "lane_5", "lane_6",
    # 場×枠の1着率から作った事前ロジット
    "lane_prior",
    # この選手がこの枠で平均よりどれだけ強いか（過去実績を縮小推定したもの）
    # 「1枠でも弱い選手」「5枠でも強い選手」を直接表現する
    "racer_lane_edge",
    # 選手能力
    "win_rate_national", "top2_national",
    "win_rate_local", "top2_local",
    # 機材
    "motor_top2", "boat_top2",
    # 級別（B2 が基準）
    "class_A1", "class_A2", "class_B1",
    # 属性
    "age", "weight",
    # 直前情報
    "ex_time",          # 展示タイム（速いほど大きくなるよう符号反転）
    "ex_missing",       # 展示タイム欠損フラグ
    # 気象 × 枠 の交互作用
    "wind_x_lane1", "wind_x_lane2", "wind_x_lane3",
    "wind_x_lane4", "wind_x_lane5", "wind_x_lane6",
    "wave_x_lane1", "wave_x_lane2", "wave_x_lane3",
    "wave_x_lane4", "wave_x_lane5", "wave_x_lane6",
    # 選手の実力 × 枠 の交互作用
    # 「1枠の弱い選手」「5枠の強い選手」を表現するための最重要ブロック。
    # これがないと枠の効果と選手の効果が足し算になってしまう。
    "wr_x_lane1", "wr_x_lane2", "wr_x_lane3",
    "wr_x_lane4", "wr_x_lane5", "wr_x_lane6",
    # モーター × 枠（アウトほど機力差が出る）
    "motor_x_lane1", "motor_x_lane2", "motor_x_lane3",
    "motor_x_lane4", "motor_x_lane5", "motor_x_lane6",
    # 展示タイム × 枠
    "ex_x_lane1", "ex_x_lane2", "ex_x_lane3",
    "ex_x_lane4", "ex_x_lane5", "ex_x_lane6",
    # 当節成績（同じ節のこれまでの走り）。市場が重視するのにモデルに無かった情報。
    # 出走表ページの「今節成績」グリッドから取れる。
    # 実測: 3連単NLL -0.016〜-0.021（2窓で再現）
    "setsu_n",         # 当節でここまで走ったレース数
    "setsu_wins",      # うち1着の数
    "setsu_avg_rank",  # 平均着順（走っていなければ 3.5）
    # レース番号。12R=優勝戦など番組編成の情報を持つ（gain比 1.4%）
    "race_no",
    # 注意: 当節の実ST平均も試したが不採用。実験ハーネス（学習〜2026-01）では
    # 2窓とも -0.002台の改善だったのに、本番構成（学習〜2026-04）の対称ゲートでは
    # +0.0012 の悪化で再現しなかった。3回中1回失敗する程度の証拠では入れない。
    # 配管（annotate_setsu の setsu_avg_st / scrape.ts の ST 解析）は残してある。
]

N_FEATURES = len(FEATURE_NAMES)

# 欠損時に使う全国平均的な既定値
DEFAULTS = {
    "win_rate_national": 5.5,
    "top2_national": 35.0,
    "win_rate_local": 5.5,
    "top2_local": 35.0,
    "motor_top2": 35.0,
    "boat_top2": 35.0,
    "age": 35.0,
    "weight": 52.0,
    "ex_time": 6.85,
}


# 当地未走などで 0.00 が入る項目。欠損として扱う。
# top2 系を外していた頃は、当地未走の選手（全体の約1割）に
# 「当地勝率=平均5.5、当地2連率=0%」という矛盾した組が渡っていた。
ZERO_IS_MISSING = (
    "win_rate_national", "top2_national",
    "win_rate_local", "top2_local",
    "motor_top2", "boat_top2",
)

# FEATURE_NAMES -> 添字。以前はレース×艇ごとに辞書を作り直していた（100万回）
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _num(v, key):
    if v is None:
        return DEFAULTS[key], True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULTS[key], True
    if f == 0.0 and key in ZERO_IS_MISSING:
        return DEFAULTS[key], True
    return f, False


def build_features(race: dict, priors) -> list[list[float]]:
    """1レース分（6艇）の特徴量行列を返す。

    race = {
        "jcd": int,
        "wind_speed": int | None,
        "wave_height": int | None,
        "entries": [ {lane, racer_id, win_rate_national, ..., ex_time}, ... ]  # 6件
    }
    priors = {
        "lane_prior": {"24-1": 1.23, ...},    # 場-枠 -> 事前ロジット
        "racer_lane": {"4320-1": 0.42, ...},  # 選手-枠 -> 平均との差
    }
    """
    import math as _math

    def _finite(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if _math.isfinite(f) else 0.0

    lane_prior = priors.get("lane_prior", {})
    racer_lane = priors.get("racer_lane", {})
    # NaN は `or 0` をすり抜ける（truthy）。TS側と同じく完全に遮断する
    wind = _finite(race.get("wind_speed"))
    wave = _finite(race.get("wave_height"))
    jcd = race["jcd"]

    # 展示タイムはレース内の相対で効く。平均を基準に符号反転（速い=大きい）
    ex_vals = []
    for e in race["entries"]:
        v, miss = _num(e.get("ex_time"), "ex_time")
        # 妥当範囲外（0.00 や列ずれの混入）は欠損扱い。
        # 異常値を1つ通すと ex_mean ごと歪んでレース全体の確率が崩壊する
        if not miss and not (5.0 <= v <= 9.0):
            miss = True
        ex_vals.append(None if miss else v)
    present = [v for v in ex_vals if v is not None]
    ex_mean = sum(present) / len(present) if present else DEFAULTS["ex_time"]

    rows = []
    for e, exv in zip(race["entries"], ex_vals):
        lane = int(e["lane"])
        row = [0.0] * N_FEATURES
        idx = _IDX

        row[idx[f"lane_{lane}"]] = 1.0
        row[idx["lane_prior"]] = lane_prior.get(f"{jcd}-{lane}", 0.0)
        rid = e.get("racer_id")
        # 学習時は train.py が「そのレースより前の実績だけ」から計算した値を
        # e["racer_lane_edge"] に埋め込む（インサンプルリーク防止）。
        # 推論時は override が無いので model.json のテーブルを引く。
        override = e.get("racer_lane_edge")
        if override is not None:
            row[idx["racer_lane_edge"]] = override
        else:
            row[idx["racer_lane_edge"]] = racer_lane.get(f"{rid}-{lane}", 0.0) if rid else 0.0

        for key in ("win_rate_national", "top2_national", "win_rate_local",
                    "top2_local", "motor_top2", "boat_top2", "age", "weight"):
            v, _ = _num(e.get(key), key)
            # 2連対率系は 0〜1 にスケール
            row[idx[key]] = v / 100.0 if key.startswith("top2") or key.endswith("_top2") else v / 10.0

        cls = (e.get("racer_class") or "B2").strip().upper()
        if cls in ("A1", "A2", "B1"):
            row[idx[f"class_{cls}"]] = 1.0

        ex_rel = 0.0
        if exv is None:
            row[idx["ex_time"]] = 0.0
            row[idx["ex_missing"]] = 1.0
        else:
            # 展示が平均より速い(小さい)ほど正の値。0.1秒差 = 1.0 のスケール。
            # ±0.3秒相当でクリップし、異常値がレースを支配しないようにする
            ex_rel = max(-3.0, min(3.0, (ex_mean - exv) * 10.0))
            row[idx["ex_time"]] = ex_rel

        row[idx[f"wind_x_lane{lane}"]] = wind / 10.0
        row[idx[f"wave_x_lane{lane}"]] = wave / 10.0

        # 交互作用。全国勝率は 5.5 を中心に振ることで枠の主効果と切り分ける
        wr, _ = _num(e.get("win_rate_national"), "win_rate_national")
        mt, _ = _num(e.get("motor_top2"), "motor_top2")
        row[idx[f"wr_x_lane{lane}"]] = (wr - 5.5) / 2.0
        row[idx[f"motor_x_lane{lane}"]] = (mt - 35.0) / 20.0
        row[idx[f"ex_x_lane{lane}"]] = ex_rel

        # 当節成績。学習時は train.annotate_setsu() が埋め、
        # 推論時は scrape.ts が今節成績グリッドから読む。無ければ「節の初走」扱い。
        n = _finite(e.get("setsu_n"))
        row[idx["setsu_n"]] = n
        row[idx["setsu_wins"]] = _finite(e.get("setsu_wins"))
        # TS側と同一: n>0 かつ有限値のときだけ平均着順、それ以外は 3.5
        ar = e.get("setsu_avg_rank")
        try:
            arf = float(ar) if ar is not None else None
            if arf is not None and not _math.isfinite(arf):
                arf = None
        except (TypeError, ValueError):
            arf = None
        row[idx["setsu_avg_rank"]] = arf if (n > 0 and arf is not None) else 3.5
        row[idx["race_no"]] = _finite(race.get("rno"))

        # 最終防衛線: どの経路から来た NaN/Inf もモデルに渡さない（TS側と同一）
        rows.append([x if _math.isfinite(x) else 0.0 for x in row])

    return rows
