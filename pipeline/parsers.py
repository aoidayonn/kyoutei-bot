"""番組表(B)・競走成績(K)の固定長テキストパーサ。

ファイルは CP932(Shift_JIS)。日本語が2バイトなので、必ず**バイト位置**で切り出す。
文字位置で切るとズレるので注意。

B(番組表) 出走者行のバイト割り付け（実データで確認済み）:
    [ 0: 1] 艇番            "1"
    [ 2: 6] 登録番号        "5009"
    [ 6:14] 選手名(全角4字) "大賀龍之"
    [14:16] 年齢            "29"
    [16:20] 支部(全角2字)   "福岡"
    [20:22] 体重            "57"
    [22:24] 級別            "A2"
    [24:29] 全国勝率        " 5.42"
    [29:35] 全国2連対率     " 36.00"
    [35:40] 当地勝率        " 5.15"
    [40:46] 当地2連対率     " 37.70"
    [46:49] モーター番号    " 56"
    [49:55] モーター2連対率 " 44.44"
    [55:58] ボート番号      "170"
    [58:64] ボート2連対率   " 34.62"
    [64:  ] 今節成績・早見

K(競走成績) 着順行のバイト割り付け:
    [ 2: 4] 着順   "01" / "F " / "K0" など
    [ 6: 7] 艇番   "1"
    [ 8:12] 登録番号
    [13:29] 選手名(全角8字)
    [29:32] モーター番号
    [34:37] ボート番号
    [39:43] 展示タイム
    [46:47] 進入コース
    [50:55] スタートタイミング  " 0.09" / "F0.01" / "L0.05"（F/Lはゼロ入り5バイト）
    [59:66] レースタイム
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- 共通ユーティリティ

_ZEN = (
    "０１２３４５６７８９：．－"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
)
_HAN = (
    "0123456789:.-"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
_Z2H = str.maketrans(_ZEN, _HAN)


def z2h(s: str) -> str:
    """全角英数記号を半角へ。"""
    return s.translate(_Z2H)


def _f(raw: str):
    """浮動小数として読む。空白や '.' のみなら None."""
    s = raw.strip()
    if not s or s in (".", "-", "0.00"):
        return None if s in ("", ".", "-") else 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _i(raw: str):
    s = raw.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        # "--5" のような値は lstrip+isdigit の判定をすり抜けて int() で落ち、
        # 上位の except がエントリを丸ごと捨てていた。欠損として扱う
        return None


def _cut(bs: bytes, start: int, end: int) -> str:
    return bs[start:end].decode("cp932", errors="replace")


def _clean_name(s: str) -> str:
    """'大　賀　　龍之介' -> '大賀龍之介'"""
    return s.replace("　", "").strip()


# ---------------------------------------------------------------- データ構造


@dataclass
class BEntry:
    lane: int
    racer_id: int
    racer_name: str
    age: int | None
    branch: str
    weight: int | None
    racer_class: str
    win_rate_national: float | None
    top2_national: float | None
    win_rate_local: float | None
    top2_local: float | None
    motor_no: int | None
    motor_top2: float | None
    boat_no: int | None
    boat_top2: float | None


@dataclass
class BRace:
    date: str
    jcd: int
    rno: int
    title: str
    distance: int | None
    deadline: str | None
    entries: list[BEntry] = field(default_factory=list)


@dataclass
class KEntry:
    lane: int
    racer_id: int
    racer_name: str
    rank: int | None          # 1〜6。失格・欠場は None
    rank_raw: str
    motor_no: int | None
    boat_no: int | None
    exhibition_time: float | None
    course: int | None        # 実際の進入コース
    start_timing: float | None
    is_flying: bool
    is_late: bool
    race_time: str | None


@dataclass
class KRace:
    date: str
    jcd: int
    rno: int
    title: str
    distance: int | None
    weather: str | None
    wind_dir: str | None
    wind_speed: int | None
    wave_height: int | None
    trifecta: str | None       # "1-4-2"
    trifecta_payout: int | None
    trifecta_pop: int | None   # 人気順
    kimarite: str | None
    entries: list[KEntry] = field(default_factory=list)


# ---------------------------------------------------------------- 番組表(B)

_B_JCD = re.compile(r"^(\d{2})BBGN")
_B_ENTRY = re.compile(r"^[1-6] \d{4}")


def parse_b(path, date: dt.date) -> list[BRace]:
    """番組表ファイルをパースして BRace のリストを返す。"""
    raw = open(path, "rb").read()
    lines = raw.split(b"\r\n") if b"\r\n" in raw else raw.split(b"\n")

    races: list[BRace] = []
    jcd = None
    current: BRace | None = None
    date_s = date.isoformat()

    for bline in lines:
        line = bline.decode("cp932", errors="replace").rstrip()

        m = _B_JCD.match(line)
        if m:
            jcd = int(m.group(1))
            current = None
            continue

        if line.endswith("BEND"):
            jcd = None
            current = None
            continue

        if jcd is None:
            continue

        # レースヘッダ:「　１Ｒ  予選　　　　          Ｈ１８００ｍ  電話投票締切予定１２：１７」
        if "Ｒ" in line and "電話投票" in line:
            h = z2h(line)
            rm = re.search(r"(\d{1,2})R", h)
            if not rm:
                continue
            rno = int(rm.group(1))
            title = re.sub(r"^\s*\d{1,2}R\s*", "", h.split("H")[0]).replace("　", "").strip()
            dm = re.search(r"H\s*(\d{3,4})\s*m", h)
            tm = re.search(r"(\d{1,2}:\d{2})", h)
            current = BRace(
                date=date_s, jcd=jcd, rno=rno, title=title,
                distance=int(dm.group(1)) if dm else None,
                deadline=tm.group(1) if tm else None,
            )
            races.append(current)
            continue

        if current is None or not _B_ENTRY.match(line):
            continue

        e = bline.rstrip()
        try:
            entry = BEntry(
                lane=int(_cut(e, 0, 1)),
                racer_id=int(_cut(e, 2, 6)),
                racer_name=_clean_name(_cut(e, 6, 14)),
                age=_i(_cut(e, 14, 16)),
                branch=_clean_name(_cut(e, 16, 20)),
                weight=_i(_cut(e, 20, 22)),
                racer_class=_cut(e, 22, 24).strip(),
                win_rate_national=_f(_cut(e, 24, 29)),
                top2_national=_f(_cut(e, 29, 35)),
                win_rate_local=_f(_cut(e, 35, 40)),
                top2_local=_f(_cut(e, 40, 46)),
                motor_no=_i(_cut(e, 46, 49)),
                motor_top2=_f(_cut(e, 49, 55)),
                boat_no=_i(_cut(e, 55, 58)),
                boat_top2=_f(_cut(e, 58, 64)),
            )
        except (ValueError, IndexError):
            continue

        if len(current.entries) < 6:
            current.entries.append(entry)

    return [r for r in races if len(r.entries) == 6]


# ---------------------------------------------------------------- 競走成績(K)

_K_JCD = re.compile(r"^(\d{2})KBGN")
_K_HEADER = re.compile(r"^\s+(\d{1,2})R\s+(.*?)\s+H(\d{3,4})m")
_K_ENTRY = re.compile(r"^  (..)  ([1-6]) (\d{4}) ")
_K_TRIFECTA = re.compile(r"３連単\s+([1-6]-[1-6]-[1-6])\s+(\d+)\s+人気\s+(\d+)")


def parse_k(path, date: dt.date) -> list[KRace]:
    """競走成績ファイルをパースして KRace のリストを返す。"""
    raw = open(path, "rb").read()
    lines = raw.split(b"\r\n") if b"\r\n" in raw else raw.split(b"\n")

    races: list[KRace] = []
    jcd = None
    current: KRace | None = None
    date_s = date.isoformat()

    for bline in lines:
        line = bline.decode("cp932", errors="replace").rstrip()

        m = _K_JCD.match(line)
        if m:
            jcd = int(m.group(1))
            current = None
            continue

        if line.endswith("KEND"):
            jcd = None
            current = None
            continue

        if jcd is None:
            continue

        hm = _K_HEADER.match(line)
        if hm:
            rno = int(hm.group(1))
            title = hm.group(2).replace("　", "").strip()
            rest = line[hm.end():]
            wm = re.search(r"風\s*(\S+?)\s*(\d+)m", rest)
            vm = re.search(r"波\s*(\d+)cm", rest)
            weather = rest.split("風")[0].replace("　", "").strip() or None
            current = KRace(
                date=date_s, jcd=jcd, rno=rno, title=title,
                distance=int(hm.group(3)),
                weather=weather,
                wind_dir=wm.group(1).replace("　", "") if wm else None,
                wind_speed=int(wm.group(2)) if wm else None,
                wave_height=int(vm.group(1)) if vm else None,
                trifecta=None, trifecta_payout=None, trifecta_pop=None,
                kimarite=None,
            )
            races.append(current)
            continue

        if current is None:
            continue

        tm = _K_TRIFECTA.search(line)
        if tm:
            current.trifecta = tm.group(1)
            current.trifecta_payout = int(tm.group(2))
            current.trifecta_pop = int(tm.group(3))
            continue

        em = _K_ENTRY.match(line)
        if not em or len(current.entries) >= 6:
            continue

        e = bline.rstrip()
        rank_raw = _cut(e, 2, 4).strip()
        rank = int(rank_raw) if rank_raw.isdigit() and 1 <= int(rank_raw) <= 6 else None

        st_raw = _cut(e, 50, 55).strip()
        is_flying = st_raw.startswith("F")
        is_late = st_raw.startswith("L")
        st = _f(st_raw.lstrip("FL")) if st_raw else None
        if is_flying and st is not None:
            st = -st  # フライングは負のST扱い

        try:
            entry = KEntry(
                lane=int(em.group(2)),
                racer_id=int(em.group(3)),
                racer_name=_clean_name(_cut(e, 13, 29)),
                rank=rank,
                rank_raw=rank_raw,
                motor_no=_i(_cut(e, 29, 32)),
                boat_no=_i(_cut(e, 34, 37)),
                exhibition_time=_f(_cut(e, 39, 43)),
                course=_i(_cut(e, 46, 47)),
                start_timing=st,
                is_flying=is_flying,
                is_late=is_late,
                race_time=(_cut(e, 59, 66).strip() or None),
            )
        except (ValueError, IndexError):
            continue

        current.entries.append(entry)

    # 決まり手はヘッダ行の末尾に入る場合があるが、6艇そろったレースのみ採用
    return [r for r in races if len(r.entries) == 6]


# ---------------------------------------------------------------- CLI（動作確認用）

if __name__ == "__main__":
    import sys
    import json

    kind, path, date = sys.argv[1], sys.argv[2], dt.date.fromisoformat(sys.argv[3])
    parsed = parse_b(path, date) if kind == "B" else parse_k(path, date)
    print(f"races={len(parsed)}")
    print(json.dumps(asdict(parsed[0]), ensure_ascii=False, indent=2))
