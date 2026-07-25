"""パース済みの番組表・競走成績を SQLite に取り込む。

    # 3年分を4プロセスでパース（約3分）
    python build_db.py --start 2023-07-01 --end 2026-07-24 --jobs 4

data/raw/ にダウンロード済みのファイルだけを対象にする（未取得の日はスキップ）。

パースはCPUを使うので複数プロセスに分けられますが、
SQLiteへの書き込みは1プロセスに集約します（並列書き込みは競合するため）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import parsers

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "kyotei.db"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    jcd             INTEGER NOT NULL,
    rno             INTEGER NOT NULL,
    title           TEXT,
    distance        INTEGER,
    deadline        TEXT,
    weather         TEXT,
    wind_dir        TEXT,
    wind_speed      INTEGER,
    wave_height     INTEGER,
    trifecta        TEXT,
    trifecta_payout INTEGER,
    trifecta_pop    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_jcd  ON races(jcd);

CREATE TABLE IF NOT EXISTS entries (
    race_id           TEXT NOT NULL,
    lane              INTEGER NOT NULL,
    racer_id          INTEGER,
    racer_name        TEXT,
    age               INTEGER,
    branch            TEXT,
    weight            INTEGER,
    racer_class       TEXT,
    win_rate_national REAL,
    top2_national     REAL,
    win_rate_local    REAL,
    top2_local        REAL,
    motor_no          INTEGER,
    motor_top2        REAL,
    boat_no           INTEGER,
    boat_top2         REAL,
    rank              INTEGER,
    course            INTEGER,
    exhibition_time   REAL,
    start_timing      REAL,
    is_flying         INTEGER DEFAULT 0,
    PRIMARY KEY (race_id, lane)
);
CREATE INDEX IF NOT EXISTS idx_entries_racer ON entries(racer_id);
"""


def race_id(date: str, jcd: int, rno: int) -> str:
    return f"{date}-{jcd:02d}-{rno:02d}"


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    # 大量INSERTを速くする設定。個人用の再生成可能なDBなので安全性より速度を優先。
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.executescript(SCHEMA)
    return con


# ---------------------------------------------------------------- パース（並列側）

def parse_day(day: dt.date):
    """1日分をパースして、DBに入れる直前の行データを返す。

    ProcessPoolExecutor から呼ばれるので、返り値はピクル可能な素のタプルにする。
    """
    b_path = RAW / "B" / f"B{day:%y%m%d}.TXT"
    k_path = RAW / "K" / f"K{day:%y%m%d}.TXT"
    if not b_path.exists() and not k_path.exists():
        return day, [], []

    b_races = parsers.parse_b(b_path, day) if b_path.exists() else []
    k_races = parsers.parse_k(k_path, day) if k_path.exists() else []

    b_map = {(r.jcd, r.rno): r for r in b_races}
    k_map = {(r.jcd, r.rno): r for r in k_races}

    race_rows, entry_rows = [], []
    date_s = day.isoformat()

    for jcd, rno in sorted(set(b_map) | set(k_map)):
        b, k = b_map.get((jcd, rno)), k_map.get((jcd, rno))
        rid = race_id(date_s, jcd, rno)

        race_rows.append((
            rid, date_s, jcd, rno,
            (b or k).title,
            (k.distance if k else None) or (b.distance if b else None),
            b.deadline if b else None,
            k.weather if k else None,
            k.wind_dir if k else None,
            k.wind_speed if k else None,
            k.wave_height if k else None,
            k.trifecta if k else None,
            k.trifecta_payout if k else None,
            k.trifecta_pop if k else None,
        ))

        b_by_lane = {e.lane: e for e in b.entries} if b else {}
        k_by_lane = {e.lane: e for e in k.entries} if k else {}

        for lane in sorted(set(b_by_lane) | set(k_by_lane)):
            be, ke = b_by_lane.get(lane), k_by_lane.get(lane)
            entry_rows.append((
                rid, lane,
                be.racer_id if be else (ke.racer_id if ke else None),
                be.racer_name if be else (ke.racer_name if ke else None),
                be.age if be else None,
                be.branch if be else None,
                be.weight if be else None,
                be.racer_class if be else None,
                be.win_rate_national if be else None,
                be.top2_national if be else None,
                be.win_rate_local if be else None,
                be.top2_local if be else None,
                be.motor_no if be else (ke.motor_no if ke else None),
                be.motor_top2 if be else None,
                be.boat_no if be else (ke.boat_no if ke else None),
                be.boat_top2 if be else None,
                ke.rank if ke else None,
                ke.course if ke else None,
                ke.exhibition_time if ke else None,
                ke.start_timing if ke else None,
                int(ke.is_flying) if ke else 0,
            ))

    return day, race_rows, entry_rows


# ---------------------------------------------------------------- 書き込み（直列側）

RACE_SQL = """INSERT OR REPLACE INTO races
    (race_id,date,jcd,rno,title,distance,deadline,weather,wind_dir,
     wind_speed,wave_height,trifecta,trifecta_payout,trifecta_pop)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

ENTRY_SQL = """INSERT OR REPLACE INTO entries
    (race_id,lane,racer_id,racer_name,age,branch,weight,racer_class,
     win_rate_national,top2_national,win_rate_local,top2_local,
     motor_no,motor_top2,boat_no,boat_top2,
     rank,course,exhibition_time,start_timing,is_flying)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--db", default=str(DB_PATH),
                   help="出力先。ネットワークドライブや同期フォルダ上では SQLite が "
                        "disk I/O error を出すことがあるため、その場合はローカルパスを指定")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="パースに使うプロセス数（既定: CPU数-1）")
    a = p.parse_args()

    # 日付の妥当性チェックは download.py と共通（「6月31日」などを分かりやすく弾く）
    from download import parse_date

    start = parse_date(a.start, "--start")
    end = parse_date(a.end, "--end") if a.end else start

    days = []
    day = start
    while day <= end:
        days.append(day)
        day += dt.timedelta(days=1)

    con = connect(Path(a.db))
    total_races = 0
    processed = 0

    def write(result):
        nonlocal total_races, processed
        _day, race_rows, entry_rows = result
        if race_rows:
            con.executemany(RACE_SQL, race_rows)
            con.executemany(ENTRY_SQL, entry_rows)
            total_races += len(race_rows)
        processed += 1
        if processed % 100 == 0:
            con.commit()
            print(f"  {processed}/{len(days)} 日  ({total_races:,} レース)", flush=True)

    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for result in ex.map(parse_day, days, chunksize=8):
                write(result)
    else:
        for d in days:
            write(parse_day(d))

    con.commit()
    con.execute("PRAGMA optimize")
    con.close()
    print(f"合計 {total_races:,} レースを取り込みました -> {a.db}")


if __name__ == "__main__":
    main()
