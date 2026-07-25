/**
 * テスト用のHTMLフィクスチャを取り直す。
 *
 *   node scripts/fetch-fixtures.mjs <jcd> <rno> <hd>
 *   node scripts/fetch-fixtures.mjs 22 1 20260720
 *
 * boatrace.jp のマークアップが変わってスクレイパーが壊れたときは、
 * まずこれで最新のHTMLを取得し、test/scrape.test.mjs を見ながら直す。
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, "..", "test", "fixtures");

const [jcd, rno, hd] = process.argv.slice(2);
if (!jcd || !rno || !hd) {
  console.error("使い方: node scripts/fetch-fixtures.mjs <jcd> <rno> <hd>");
  console.error("例:     node scripts/fetch-fixtures.mjs 22 1 20260720");
  process.exit(1);
}

const pages = ["racelist", "beforeinfo", "odds3t"];
mkdirSync(outDir, { recursive: true });

for (const page of pages) {
  const url =
    `https://www.boatrace.jp/owpc/pc/race/${page}` +
    `?rno=${rno}&jcd=${String(jcd).padStart(2, "0")}&hd=${hd}`;

  const res = await fetch(url, {
    headers: { "User-Agent": "kyoutei-bot/1.0 (fixture updater)" },
  });
  if (!res.ok) {
    console.error(`${page}: ${res.status}`);
    continue;
  }
  const html = await res.text();
  writeFileSync(join(outDir, `${page}.html`), html);
  console.log(`${page}.html (${html.length} bytes)`);

  // 公式サイトに負荷をかけない
  await new Promise((r) => setTimeout(r, 1500));
}

console.log("\n取得しました。test/scrape.test.mjs の期待値も更新してください。");
