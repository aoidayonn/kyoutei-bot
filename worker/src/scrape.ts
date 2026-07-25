/**
 * boatrace.jp から出走表・直前情報・3連単オッズを取得する。
 *
 * 公式のオッズAPI/番組表閲覧サービスは2025年3月で終了しているため、
 * 公開ページのHTMLを読む。DOMパーサは使わず、正規表現でセルを抜き出す。
 *
 * 注意: サイトのマークアップが変わると壊れる。壊れた場合は
 *       `npm run probe` で生HTMLを確認して調整すること。
 */

const BASE = "https://www.boatrace.jp/owpc/pc/race";
const UA =
  "Mozilla/5.0 (compatible; kyoutei-bot/1.0; personal use)";

export interface ScrapedEntry {
  lane: number;
  racerId: number | null;
  racerName: string;
  racerClass: string | null;
  age: number | null;
  weight: number | null;
  avgSt: number | null;
  flyingCount: number | null;
  winRateNational: number | null;
  top2National: number | null;
  winRateLocal: number | null;
  top2Local: number | null;
  motorNo: number | null;
  motorTop2: number | null;
  boatNo: number | null;
  boatTop2: number | null;
  exTime: number | null;
  tilt: number | null;
}

export interface ScrapedRace {
  jcd: number;
  rno: number;
  hd: string;
  deadline: string | null;
  entries: ScrapedEntry[];
  weather: string | null;
  windSpeed: number | null;
  waveHeight: number | null;
  temperature: number | null;
  waterTemp: number | null;
  hasBeforeInfo: boolean;
}

// ---------------------------------------------------------------- ユーティリティ

function stripTags(s: string): string {
  return s
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function toNum(s: string | undefined | null): number | null {
  if (!s) return null;
  const m = s.match(/-?\d+(\.\d+)?/);
  return m ? parseFloat(m[0]) : null;
}

function tbodies(html: string): string[] {
  return [...html.matchAll(/<tbody[^>]*>([\s\S]*?)<\/tbody>/g)].map((m) => m[1]);
}

function cells(tbody: string): string[] {
  return [...tbody.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((m) => m[1]);
}

async function get(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, "Accept-Language": "ja" },
    cf: { cacheTtl: 60, cacheEverything: true },
  } as RequestInit);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return await res.text();
}

function q(jcd: number, rno: number, hd: string): string {
  return `?rno=${rno}&jcd=${String(jcd).padStart(2, "0")}&hd=${hd}`;
}

// ---------------------------------------------------------------- 出走表

/**
 * 出走表 tbody は1艇1ブロック。
 *   cell[2] : "5009 / A2 ... 大賀 龍之介 ... 福岡/福岡 29歳/55.1kg"
 *   cell[3] : "F1 L0 0.16"          （F数 / L数 / 平均ST）
 *   cell[4] : "5.42 36.00 52.00"    （全国 勝率 / 2連率 / 3連率）
 *   cell[5] : "5.20 41.18 52.94"    （当地）
 *   cell[6] : "56 44.44 65.43"      （モーター番号 / 2連率 / 3連率）
 *   cell[7] : "170 34.62 58.97"     （ボート番号 / 2連率 / 3連率）
 */
export function parseRacelist(html: string): Partial<ScrapedEntry>[] {
  const out: Partial<ScrapedEntry>[] = [];

  for (const tb of tbodies(html)) {
    const c = cells(tb);
    if (c.length < 8) continue;
    const profile = stripTags(c[2]);
    const idm = profile.match(/(\d{4})\s*\/\s*([AB][12])/);
    if (!idm) continue;

    const nums = (i: number) =>
      stripTags(c[i]).split(/\s+/).map((x) => toNum(x));

    const nat = nums(4);
    const loc = nums(5);
    const motor = nums(6);
    const boat = nums(7);
    const st = nums(3);

    const nameMatch = profile.match(/[0-9]{4}\s*\/\s*[AB][12]\s*(.+?)\s*[^\s]*\/[^\s]*\s*\d+歳/);
    const ageWeight = profile.match(/(\d+)歳\s*\/\s*([\d.]+)kg/);

    out.push({
      lane: out.length + 1,
      racerId: parseInt(idm[1], 10),
      racerClass: idm[2],
      racerName: (nameMatch?.[1] ?? "").replace(/\s+/g, " ").trim(),
      age: ageWeight ? parseInt(ageWeight[1], 10) : null,
      weight: ageWeight ? parseFloat(ageWeight[2]) : null,
      flyingCount: st[0],
      avgSt: st[2] ?? null,
      winRateNational: nat[0],
      top2National: nat[1],
      winRateLocal: loc[0],
      top2Local: loc[1],
      motorNo: motor[0],
      motorTop2: motor[1],
      boatNo: boat[0],
      boatTop2: boat[1],
    });
    if (out.length === 6) break;
  }
  return out;
}

// ---------------------------------------------------------------- 直前情報

/**
 * 直前情報 tbody も1艇1ブロック。
 *   cell[0]=艇番 cell[2]=選手名 cell[3]=体重 cell[4]=展示タイム cell[5]=チルト
 * 気象は weather1 ブロックのテキストから拾う。
 */
export function parseBeforeInfo(html: string) {
  const exhibition: { exTime: number | null; tilt: number | null }[] = [];

  for (const tb of tbodies(html)) {
    const c = cells(tb);
    if (c.length < 6) continue;
    const lane = toNum(stripTags(c[0]));
    if (lane === null || lane < 1 || lane > 6) continue;
    exhibition.push({
      exTime: toNum(stripTags(c[4])),
      tilt: toNum(stripTags(c[5])),
    });
    if (exhibition.length === 6) break;
  }

  const wi = html.indexOf("weather1");
  const block = wi >= 0 ? stripTags(html.slice(wi, wi + 4000)) : "";
  const pick = (re: RegExp) => {
    const m = block.match(re);
    return m ? parseFloat(m[1]) : null;
  };

  return {
    exhibition,
    temperature: pick(/気温\s*([\d.]+)\s*℃/),
    windSpeed: pick(/風速\s*([\d.]+)\s*m/),
    waterTemp: pick(/水温\s*([\d.]+)\s*℃/),
    waveHeight: pick(/波高\s*([\d.]+)\s*cm/),
    weather: (block.match(/(晴|曇り|曇|雨|雪|霧)/) ?? [])[1] ?? null,
  };
}

// ---------------------------------------------------------------- オッズ

/**
 * 3連単オッズは 20行 × 6列（列 = 1着艇）。
 * class="oddsPoint" のセルを文書順に120個拾えば、並びは決定論的に決まる:
 *
 *   index = row * 6 + col
 *   1着 = col + 1
 *   2着 = (1着以外を昇順に並べたもの)[row / 4]
 *   3着 = (1着・2着以外を昇順に並べたもの)[row % 4]
 */
export function parseOdds3t(html: string): Record<string, number> {
  const values = [...html.matchAll(/class="oddsPoint[^"]*"[^>]*>([^<]*)</g)].map(
    (m) => m[1].trim(),
  );
  const odds: Record<string, number> = {};
  if (values.length < 120) return odds;

  for (let row = 0; row < 20; row++) {
    for (let col = 0; col < 6; col++) {
      const first = col + 1;
      const rest = [1, 2, 3, 4, 5, 6].filter((x) => x !== first);
      const second = rest[Math.floor(row / 4)];
      const rest2 = rest.filter((x) => x !== second);
      const third = rest2[row % 4];
      const raw = values[row * 6 + col];
      const v = parseFloat(raw);
      if (Number.isFinite(v)) odds[`${first}-${second}-${third}`] = v;
    }
  }
  return odds;
}

// ---------------------------------------------------------------- 締切時刻

export function parseDeadline(html: string, rno: number): string | null {
  // 出走表ページ上部のレースタブに締切時刻が入っている
  const m = html.match(/締切予定[^0-9]*(\d{1,2}:\d{2})/);
  if (m) return m[1];
  const tab = html.match(new RegExp(`${rno}R[\\s\\S]{0,200}?(\\d{1,2}:\\d{2})`));
  return tab ? tab[1] : null;
}

// ---------------------------------------------------------------- 公開API

export async function fetchRace(
  jcd: number,
  rno: number,
  hd: string,
): Promise<ScrapedRace> {
  const [racelistHtml, beforeHtml] = await Promise.all([
    get(`${BASE}/racelist${q(jcd, rno, hd)}`),
    get(`${BASE}/beforeinfo${q(jcd, rno, hd)}`).catch(() => ""),
  ]);

  const base = parseRacelist(racelistHtml);
  if (base.length < 6) {
    throw new Error("出走表を取得できませんでした（開催がないか、まだ公開されていません）");
  }

  const before = beforeHtml ? parseBeforeInfo(beforeHtml) : null;
  const hasBeforeInfo = !!before && before.exhibition.some((e) => e.exTime !== null);

  const entries: ScrapedEntry[] = base.map((e, i) => ({
    ...(e as ScrapedEntry),
    lane: i + 1,
    exTime: hasBeforeInfo ? before!.exhibition[i]?.exTime ?? null : null,
    tilt: hasBeforeInfo ? before!.exhibition[i]?.tilt ?? null : null,
  }));

  return {
    jcd,
    rno,
    hd,
    deadline: parseDeadline(racelistHtml, rno),
    entries,
    weather: before?.weather ?? null,
    windSpeed: before?.windSpeed ?? null,
    waveHeight: before?.waveHeight ?? null,
    temperature: before?.temperature ?? null,
    waterTemp: before?.waterTemp ?? null,
    hasBeforeInfo,
  };
}

export async function fetchOdds(
  jcd: number,
  rno: number,
  hd: string,
): Promise<Record<string, number>> {
  try {
    const html = await get(`${BASE}/odds3t${q(jcd, rno, hd)}`);
    return parseOdds3t(html);
  } catch {
    return {};
  }
}

export async function fetchResult(
  jcd: number,
  rno: number,
  hd: string,
): Promise<{ trifecta: string | null; payout: number | null }> {
  try {
    const html = await get(`${BASE}/raceresult${q(jcd, rno, hd)}`);
    const m = html.match(/3連単[\s\S]{0,600}?([1-6])\s*<[\s\S]{0,80}?([1-6])\s*<[\s\S]{0,80}?([1-6])\s*<[\s\S]{0,400}?¥([\d,]+)/);
    if (!m) return { trifecta: null, payout: null };
    return {
      trifecta: `${m[1]}-${m[2]}-${m[3]}`,
      payout: parseInt(m[4].replace(/,/g, ""), 10),
    };
  } catch {
    return { trifecta: null, payout: null };
  }
}
