"""公式サイトから番組表(B)・競走成績(K)のLZHをダウンロードして解凍する。

公式ダウンロードサイト:
  番組表   https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh
  競走成績 https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh

使い方:
    # 3年分を8並列で取得（約20分）
    python download.py --start 2023-07-01 --end 2026-07-24 --workers 8

    # 前日分だけ
    python download.py --start 2026-07-24

取得済みのファイルは自動でスキップされるので、途中で止めて再開できます。

並列数について
--------------
1リクエストあたりのサーバ応答が5〜10秒かかるため、直列だと非常に遅くなります。
並列化は「待ち時間を重ねる」だけなのでサーバへの瞬間的な負荷は上がりますが、
既定の8並列なら実効レートは毎秒1〜2リクエスト程度で、
ブラウザで普通に閲覧するのと大差ありません。
公営競技の公式サイトなので、必要以上に上げないでください（上限32）。
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import io
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lhafile
import requests

BASE = "https://www1.mbrace.or.jp/od2"
UA = "kyoutei-bot/1.0 (personal research)"
MAX_WORKERS = 32

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_local = threading.local()
_print_lock = threading.Lock()


def _session() -> requests.Session:
    """スレッドごとにセッションを使い回してTCP接続を節約する。"""
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        _local.session = s
    return _local.session


def url_for(kind: str, day: dt.date) -> str:
    """kind は 'B'(番組表) or 'K'(競走成績)."""
    assert kind in ("B", "K")
    return f"{BASE}/{kind}/{day:%Y%m}/{kind.lower()}{day:%y%m%d}.lzh"


def local_path(kind: str, day: dt.date) -> Path:
    return RAW_DIR / kind / f"{kind}{day:%y%m%d}.TXT"


def fetch_day(kind: str, day: dt.date, force: bool = False, retries: int = 3):
    """1日分をダウンロード・解凍して保存。

    戻り値: "ok" | "skip"（既存） | "missing"（開催なし等） | "error"
    """
    out = local_path(kind, day)
    if out.exists() and not force:
        return "skip"

    url = url_for(kind, day)
    last_error = None

    for attempt in range(retries):
        try:
            res = _session().get(url, timeout=60)
        except requests.RequestException as e:
            last_error = e
            # 指数バックオフ + ゆらぎ（同時に再試行が集中しないように）
            time.sleep((2 ** attempt) + random.random())
            continue

        if res.status_code == 404:
            return "missing"
        if res.status_code != 200 or len(res.content) < 100:
            last_error = f"HTTP {res.status_code} ({len(res.content)}B)"
            time.sleep((2 ** attempt) + random.random())
            continue

        try:
            archive = lhafile.Lhafile(io.BytesIO(res.content))
            members = archive.infolist()
            if not members:
                return "missing"
            data = archive.read(members[0].filename)
        except Exception as e:  # noqa: BLE001
            last_error = f"解凍失敗 {e}"
            break

        out.parent.mkdir(parents=True, exist_ok=True)
        # 途中で中断されても壊れたファイルが残らないよう、一時ファイル経由で置く
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(out)
        return "ok"

    with _print_lock:
        print(f"  [失敗] {url}: {last_error}", file=sys.stderr)
    return "error"


def fetch_range(start: dt.date, end: dt.date, kinds=("B", "K"),
                force=False, workers=8, retries=3):
    """start〜end(両端含む)を並列でダウンロードする。"""
    workers = max(1, min(workers, MAX_WORKERS))

    jobs = []
    day = start
    while day <= end:
        for kind in kinds:
            if force or not local_path(kind, day).exists():
                jobs.append((kind, day))
        day += dt.timedelta(days=1)

    if not jobs:
        print("すべて取得済みです。")
        return

    print(f"{len(jobs)} ファイルを {workers} 並列で取得します")
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    started = time.time()
    done = 0

    def run(job):
        nonlocal done
        result = fetch_day(job[0], job[1], force, retries)
        with _print_lock:
            counts[result] += 1
            done += 1
            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - started
                rate = done / elapsed
                remaining = (len(jobs) - done) / rate if rate > 0 else 0
                print(
                    f"  {done}/{len(jobs)}  "
                    f"({rate:.1f} 件/秒, 残り約 {remaining / 60:.1f} 分)",
                    flush=True,
                )
        return result

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, jobs))

    elapsed = time.time() - started
    print(
        f"\n完了: 取得 {counts['ok']} / 開催なし {counts['missing']} / "
        f"失敗 {counts['error']}  ({elapsed / 60:.1f} 分)"
    )
    if counts["error"]:
        print("失敗したファイルは、同じコマンドをもう一度実行すれば再取得されます。")


def parse_date(text: str, label: str) -> dt.date:
    """YYYY-MM-DD を日付にする。存在しない日付は理由を添えて弾く。

    「6月31日」のような実在しない日付を渡したときに
    ValueError: day is out of range for month とだけ出ても原因が分かりにくいので、
    何がおかしいのかをはっきり伝える。
    """
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        pass

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", (text or "").strip())
    if m:
        year, month, day = (int(x) for x in m.groups())
        if 1 <= month <= 12:
            last = calendar.monthrange(year, month)[1]
            if day > last:
                raise SystemExit(
                    f"{label} の日付 {text} は存在しません。"
                    f"{year}年{month}月は{last}日までです（{year}-{month:02d}-{last} では？）"
                )
        else:
            raise SystemExit(f"{label} の月が不正です: {text}")

    raise SystemExit(f"{label} は YYYY-MM-DD 形式で指定してください（受け取った値: {text}）")


def main():
    p = argparse.ArgumentParser(
        description="番組表・競走成績のダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD（省略時は start と同じ）")
    p.add_argument("--kinds", default="BK", help="B / K / BK")
    p.add_argument("--workers", type=int, default=8,
                   help=f"並列数（既定8、上限{MAX_WORKERS}）")
    p.add_argument("--retries", type=int, default=3, help="失敗時の再試行回数")
    p.add_argument("--force", action="store_true", help="既存ファイルも再取得")
    a = p.parse_args()

    start = parse_date(a.start, "--start")
    end = parse_date(a.end, "--end") if a.end else start
    if end < start:
        raise SystemExit(f"--end ({end}) が --start ({start}) より前になっています")

    fetch_range(start, end, tuple(a.kinds), a.force, a.workers, a.retries)


if __name__ == "__main__":
    main()
