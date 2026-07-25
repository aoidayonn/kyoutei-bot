/** ボートレース場コードと表記ゆれの吸収（pipeline/stadiums.py のミラー）。 */

export const STADIUMS: Record<number, string> = {
  1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
  7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
  13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
  19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
};

const ALIASES: Record<string, number> = {
  きりゅう: 1, キリュウ: 1,
  とだ: 2, トダ: 2,
  えどがわ: 3, エドガワ: 3, 江戸: 3,
  へいわじま: 4, ヘイワジマ: 4, 平和: 4,
  たまがわ: 5, タマガワ: 5, 多摩: 5,
  はまなこ: 6, ハマナコ: 6, 浜名: 6,
  がまごおり: 7, ガマゴオリ: 7,
  とこなめ: 8, トコナメ: 8,
  つ: 9, ツ: 9,
  みくに: 10, ミクニ: 10,
  ビワコ: 11, びわ湖: 11, 琵琶湖: 11, びわこ: 11,
  すみのえ: 12, スミノエ: 12,
  あまがさき: 13, アマガサキ: 13, 尼: 13,
  なると: 14, ナルト: 14,
  まるがめ: 15, マルガメ: 15,
  こじま: 16, コジマ: 16,
  みやじま: 17, ミヤジマ: 17,
  とくやま: 18, トクヤマ: 18,
  しものせき: 19, シモノセキ: 19,
  わかまつ: 20, ワカマツ: 20,
  あしや: 21, アシヤ: 21,
  ふくおか: 22, フクオカ: 22,
  からつ: 23, カラツ: 23,
  おおむら: 24, オオムラ: 24,
};

const NAME_TO_JCD: Record<string, number> = Object.fromEntries(
  Object.entries(STADIUMS).map(([k, v]) => [v, Number(k)]),
);

export function stadiumName(jcd: number): string {
  return STADIUMS[jcd] ?? `不明(${jcd})`;
}

/** レース場名・かな・場コードから jcd を解決する。 */
export function resolveStadium(text: string): number | null {
  let t = text
    .trim()
    .replace(/[　\s]/g, "")
    .replace(/ボートレース|競艇/g, "")
    .replace(/場$/, "");

  if (/^\d+$/.test(t)) {
    const n = parseInt(t, 10);
    return STADIUMS[n] ? n : null;
  }
  if (NAME_TO_JCD[t] !== undefined) return NAME_TO_JCD[t];
  if (ALIASES[t] !== undefined) return ALIASES[t];

  for (const [name, jcd] of Object.entries(NAME_TO_JCD)) {
    if (t.includes(name) || name.includes(t)) return jcd;
  }
  return null;
}

/**
 * 「大村 12」「24 12」「おおむら12R」などを (jcd, rno) に分解する。
 * 末尾の数字をレース番号、それ以外をレース場として扱う。
 */
export function parseCommand(
  raw: string,
): { jcd: number; rno: number } | { error: string } {
  const text = raw.trim().replace(/[Ｒｒ]/g, "R").replace(/R$/i, "").trim();

  // 「24 12」のように数字が2つ並ぶ場合
  const twoNums = text.match(/^(\d{1,2})\s*[\s\-/]\s*(\d{1,2})$/);
  if (twoNums) {
    const jcd = parseInt(twoNums[1], 10);
    const rno = parseInt(twoNums[2], 10);
    if (STADIUMS[jcd] && rno >= 1 && rno <= 12) return { jcd, rno };
  }

  const m = text.match(/^(.+?)\s*(\d{1,2})$/);
  if (!m) return { error: "レース場とレース番号を送ってください（例：大村 12）" };

  const rno = parseInt(m[2], 10);
  if (rno < 1 || rno > 12) return { error: `レース番号は1〜12です（受け取った値: ${rno}）` };

  const jcd = resolveStadium(m[1]);
  if (jcd === null) return { error: `レース場「${m[1]}」が分かりませんでした` };

  return { jcd, rno };
}
