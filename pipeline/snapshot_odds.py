"""締切直前の3連単オッズを保存する。

なぜ必要か
----------
公式は過去のオッズを公開していない。手元にあるのは「的中した組の払戻金」だけなので、
「期待値1.05超の買い目だけを買う」という戦略の回収率をバックテストできない。

そこで**今日から自分でオッズを貯める**。数週間分たまれば、
実際に運用している買い方そのものを検証できるようになる。

使い方（GitHub Actions から1日数回叩く想定）:
    python snapshot_odds.py --db ../data/odds.db
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://www.boatrace.jp/owpc/pc/race"
UA = "kyoutei-bot/1.0 (personal research)"

# 直列だと開催中の全レース(約180)で3分以上かかり、
# 30分おきに回すとGitHub Actionsの無料枠を使い切ってしまう。
WORKERS = 6

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"User-Agent": UA})
    return _local.s

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    race_id      TEXT NOT NULL,   -- "20260725-24-12"
    captured_at  TEXT NOT NULL,
    minutes_left INTEGER,         -- 締切まで何分か（分からなければ NULL）
    odds_json    TEXT NOT NULL,   -- {"1-2-3": 13.9, ...}
    PRIMARY KEY (race_id, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_odds_race ON odds_snapshots(race_id);
"""


def parse_odds3t(html: str) -> dict[str, float]:
    """worker/src/scrape.ts の parseOdds3t と同じロジック。

    class="oddsPoint" のセルを文書順に120個拾えば並びは決定論的に決まる:
        index = row*6 + col, 1着 = col+1,
        2着 = (1着以外の昇順)[row//4], 3着 = (1着2着以外の昇順)[row%4]
    """
    values = re.findall(r'class="oddsPoint[^"]*"[^>]*>([^<]*)<', html)
    if len(values) < 120:
        return {}

    odds = {}
    for row in range(20):
        for col in range(6):
            first = col + 1
            rest = [x for x in range(1, 7) if x != first]
            second = rest[row // 4]
            rest2 = [x for x in rest if x != second]
            third = rest2[row % 4]
            try:
                v = float(values[row * 6 + col].strip())
            except ValueError:
                continue
            odds[f"{first}-{second}-{third}"] = v
    return odds


def today_races(hd: str) -> list[tuple[int, int]]:
    """本日開催中の (jcd, rno) を列挙する。"""
    res = _session().get(f"{BASE}/index?hd={hd}", timeout=30)
    jcds = sorted({int(m) for m in re.findall(r"jcd=(\d{2})", res.text)})
    return [(j, r) for j in jcds if 1 <= j <= 24 for r in range(1, 13)]


def fetch_one(hd: str, jcd: int, rno: int):
    """1レース分のオッズを取る。未発売・締切済みなら None。"""
    url = f"{BASE}/odds3t?rno={rno}&jcd={jcd:02d}&hd={hd}"
    try:
        res = _session().get(url, timeout=30)
    except requests.RequestException:
        return None
    odds = parse_odds3t(res.text)
    return odds if len(odds) >= 100 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(Path(__file__).resolve().parent.parent
                                       / "data" / "odds.db"))
    p.add_argument("--hd", default=None, help="YYYYMMDD（省略時は今日）")
    p.add_argument("--jcd", type=int, default=None, help="特定の場だけ取る")
    p.add_argument("--workers", type=int, default=WORKERS)
    a = p.parse_args()

    hd = a.hd or (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y%m%d")

    db = Path(a.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    targets = today_races(hd)
    if a.jcd:
        targets = [t for t in targets if t[0] == a.jcd]
    if not targets:
        print("本日は開催がありません")
        return

    now = dt.datetime.utcnow().isoformat()
    rows = []

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = ex.map(lambda t: (t, fetch_one(hd, *t)), targets)
        for (jcd, rno), odds in results:
            if odds:
                rows.append((
                    f"{hd}-{jcd:02d}-{rno:02d}", now, None, json.dumps(odds),
                ))

    # 締切まで何分だったかは、番組表側の deadline と captured_at を突き合わせれば
    # 後から計算できる。ここで余計なリクエストを増やさない。
    con.executemany(
        "INSERT OR REPLACE INTO odds_snapshots"
        " (race_id, captured_at, minutes_left, odds_json) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()

    total, = con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()
    con.close()
    print(f"{len(rows)} / {len(targets)} レースのオッズを保存（累計 {total:,} 件）-> {db}")


if __name__ == "__main__":
    main()
