/** 日時ユーティリティ。index.ts と settle.ts で重複実装していたものを共通化。 */

/** 日本時間の今日を YYYYMMDD で返す。 */
export function todayJst(): string {
  const now = new Date(Date.now() + 9 * 3600 * 1000);
  return now.toISOString().slice(0, 10).replace(/-/g, "");
}
