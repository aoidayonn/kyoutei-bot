/**
 * 予想の答え合わせ。
 *
 * predictions テーブルに保存した予想へ、実際のレース結果と払戻金を書き戻す。
 * これがないと「予想は貯まるが検証できない」状態になる。
 *
 * Cron Trigger から1日2回呼ばれる。
 */

import { fetchResult } from "./scrape";
import { todayJst } from "./time";

/**
 * 1回の実行で処理する最大レース数。
 *
 * Workers Free のサブリクエスト上限は 50/リクエスト。
 * 1レースの答え合わせは 結果ページfetch 1 + D1 UPDATE 1 = 2消費するので、
 * SELECT/COUNT/ALTER のぶんも含めて 15件 × 2 + 5 ≒ 35 に抑える
 * （以前は40件設定で、上限に食い込んで途中で打ち切られる可能性があった）。
 */
const MAX_PER_RUN = 15;

/**
 * 結果の取得を諦めるまでの試行回数。
 *
 * 以前は失敗した行を無期限にリトライし続ける設計だった。
 * 中止レースやページ構造の変化で結果が永久に取れない行は
 * `hd ASC` の並びで毎回先頭に選ばれ続けるため、そのような行が
 * MAX_PER_RUN 件たまった時点で新しいレースの答え合わせが完全に止まる。
 * 一定回数で諦めて後ろに回すことで、詰まりを防ぐ。
 */
const MAX_ATTEMPTS = 8;

export interface SettleResult {
  checked: number;
  settled: number;
  failed: number;
  pending: number;
  abandoned: number;
}

/** 既存DBに attempts 列がなければ足す（一度だけ成功し、以降は例外を握る）。 */
async function ensureSchema(db: D1Database): Promise<void> {
  try {
    await db
      .prepare("ALTER TABLE predictions ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
      .run();
  } catch {
    // 既に列がある
  }
}

export async function settlePredictions(db: D1Database): Promise<SettleResult> {
  await ensureSchema(db);

  const today = todayJst();

  const { results } = await db
    .prepare(
      `SELECT race_id, hd, jcd, rno FROM predictions
        WHERE settled_at IS NULL AND hd <= ? AND attempts < ?
        ORDER BY hd ASC, jcd ASC, rno ASC
        LIMIT ?`,
    )
    .bind(today, MAX_ATTEMPTS, MAX_PER_RUN)
    .all<{ race_id: string; hd: string; jcd: number; rno: number }>();

  const rows = results ?? [];
  let settled = 0;
  let failed = 0;

  for (const row of rows) {
    const result = await fetchResult(row.jcd, row.rno, row.hd);

    if (result.trifecta) {
      await db
        .prepare(
          `UPDATE predictions
              SET actual = ?, payout = ?, settled_at = ?
            WHERE race_id = ?`,
        )
        .bind(result.trifecta, result.payout, new Date().toISOString(), row.race_id)
        .run();
      settled += 1;
    } else {
      // 当日ぶんは夜にならないと結果が出ないので、試行回数は数えない。
      // 前日以前で取れないのは中止・不成立・ページ変更のどれかなので数える。
      if (row.hd < today) {
        await db
          .prepare("UPDATE predictions SET attempts = attempts + 1 WHERE race_id = ?")
          .bind(row.race_id)
          .run();
      }
      failed += 1;
    }
  }

  const pendingRow = await db
    .prepare(
      "SELECT COUNT(*) AS n FROM predictions WHERE settled_at IS NULL AND attempts < ?",
    )
    .bind(MAX_ATTEMPTS)
    .first<{ n: number }>();
  const abandonedRow = await db
    .prepare(
      "SELECT COUNT(*) AS n FROM predictions WHERE settled_at IS NULL AND attempts >= ?",
    )
    .bind(MAX_ATTEMPTS)
    .first<{ n: number }>();

  return {
    checked: rows.length,
    settled,
    failed,
    pending: pendingRow?.n ?? 0,
    abandoned: abandonedRow?.n ?? 0,
  };
}
