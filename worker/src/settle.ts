/**
 * 予想の答え合わせ。
 *
 * predictions テーブルに保存した予想へ、実際のレース結果と払戻金を書き戻す。
 * これがないと「予想は貯まるが検証できない」状態になる。
 *
 * Cron Trigger から1日2回呼ばれる。
 */

import { fetchResult } from "./scrape";

/**
 * Workers Free プランの1リクエストあたりのサブリクエスト上限は50。
 * 結果ページ1件につき1リクエストなので、余裕を見てこの数で打ち切る。
 * 残りは次回の実行で処理される。
 */
const MAX_PER_RUN = 40;

export interface SettleResult {
  checked: number;
  settled: number;
  pending: number;
}

export async function settlePredictions(db: D1Database): Promise<SettleResult> {
  // 当日のレースはまだ結果が出ていない可能性があるので、前日以前を対象にする。
  // ただし当日ぶんも夜には確定するので、hd <= 今日 で拾って
  // 結果が取れなければ次回に回す。
  const today = new Date(Date.now() + 9 * 3600 * 1000).toISOString().slice(0, 10).replace(/-/g, "");

  const { results } = await db
    .prepare(
      `SELECT race_id, hd, jcd, rno FROM predictions
        WHERE settled_at IS NULL AND hd <= ?
        ORDER BY hd ASC, jcd ASC, rno ASC
        LIMIT ?`,
    )
    .bind(today, MAX_PER_RUN)
    .all<{ race_id: string; hd: string; jcd: number; rno: number }>();

  const rows = results ?? [];
  let settled = 0;

  for (const row of rows) {
    const result = await fetchResult(row.jcd, row.rno, row.hd);
    if (!result.trifecta) continue; // まだ結果が出ていない、または中止

    await db
      .prepare(
        `UPDATE predictions
            SET actual = ?, payout = ?, settled_at = ?
          WHERE race_id = ?`,
      )
      .bind(result.trifecta, result.payout, new Date().toISOString(), row.race_id)
      .run();
    settled += 1;
  }

  const pendingRow = await db
    .prepare("SELECT COUNT(*) AS n FROM predictions WHERE settled_at IS NULL")
    .first<{ n: number }>();

  return {
    checked: rows.length,
    settled,
    pending: pendingRow?.n ?? 0,
  };
}
