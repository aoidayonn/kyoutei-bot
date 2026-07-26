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
  /** 当節ここまでのレース数（今節成績グリッドから。着順が付いたものだけ） */
  setsuN: number | null;
  /** うち1着 */
  setsuWins: number | null;
  /** 平均着順（走っていなければ null） */
  setsuAvgRank: number | null;
  /** 当節の実ST平均（F・欠測除く。無ければ null） */
  setsuAvgSt: number | null;
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

async function get(url: string, ttl = 60): Promise<string> {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, "Accept-Language": "ja" },
    // cacheTtl はエラーページまで60秒キャッシュしてしまう
    // （未公開の出走表を見たユーザーが送り直しても60秒間同じ失敗が返る）ので、
    // 成功レスポンスだけキャッシュする。
    cf: {
      cacheEverything: true,
      cacheTtlByStatus: { "200-299": ttl, "300-399": 0, "400-599": 0 },
    },
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
/**
 * 今節成績グリッド（1選手分の tbody）を読む。
 *
 * 構造は4行:
 *   tr[0] … 走った/走る予定の R番号（"5R" 等。艇番色クラス付きのリンク）
 *   tr[1] … 進入コース
 *   tr[2] … ST
 *   tr[3] … 着順（全角数字。Ｆ・転・落など数字以外は着外扱い）
 *
 * 当日の早いレースの結果も、確定するとこのグリッドに載る（実地確認済み）。
 * rno を渡すと「いま見ているレース自身」（R番号が一致する最後の列）を除外する。
 * これをしないと、締切後に照会したとき自分自身の結果が特徴量に混ざる。
 */
export function parseSetsu(
  tb: string,
  rno?: number,
): { setsuN: number; setsuWins: number; setsuAvgRank: number | null; setsuAvgSt: number | null } {
  const rows = [...tb.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)].map((m) => m[1]);
  const empty = { setsuN: 0, setsuWins: 0, setsuAvgRank: null, setsuAvgSt: null };
  if (rows.length < 4) return empty;

  // rowspan 付きセル（艇番・写真・選手情報…）はグリッドの一部ではない
  const gridCells = (row: string) =>
    [...row.matchAll(/<td((?:(?!rowspan)[^>])*)>([\s\S]*?)<\/td>/g)].map((m) => m[2]);

  const rCells = gridCells(rows[0]);
  const stCells = gridCells(rows[rows.length - 2]);
  const finishCells = gridCells(rows[rows.length - 1]);
  if (rCells.length === 0 || rCells.length !== finishCells.length) return empty;

  // 全角数字 → 半角
  const z2h = (s: string) =>
    s.replace(/[０-９]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0));

  const cols = rCells.map((rc, i) => ({
    r: toNum(stripTags(rc)),
    finish: (() => {
      const t = z2h(stripTags(finishCells[i]));
      const m = t.match(/^([1-6])$/);
      return m ? parseInt(m[1], 10) : null;
    })(),
    st: (() => {
      // ".08" の形式。toNum は先頭の "." を読めず 8 になるので専用に解く。
      // F（フライング）は学習側の 0<st<1 フィルタと合わせて除外する
      const t = z2h(stripTags(stCells[i] ?? ""));
      if (/[FfＦ]/.test(t)) return null;
      const m = t.match(/(\d*\.\d+)/);
      if (!m) return null;
      const v = parseFloat(m[1].startsWith(".") ? "0" + m[1] : m[1]);
      return v > 0 && v < 1 ? v : null;
    })(),
  }));

  // いま見ているレース自身を除外（R番号一致の最後の列。
  // 同じR番号を別の日に走った列は残す）
  if (rno) {
    for (let i = cols.length - 1; i >= 0; i--) {
      if (cols[i].r === rno) {
        cols.splice(i, 1);
        break;
      }
    }
  }

  let n = 0;
  let wins = 0;
  let sum = 0;
  let stN = 0;
  let stSum = 0;
  for (const col of cols) {
    if (col.st !== null) {
      stN++;
      stSum += col.st;
    }
    if (col.finish === null) continue; // 未走・F・失格などは数えない（DB側の rank NULL と同じ）
    n++;
    sum += col.finish;
    if (col.finish === 1) wins++;
  }
  return {
    setsuN: n,
    setsuWins: wins,
    setsuAvgRank: n ? sum / n : null,
    setsuAvgSt: stN ? stSum / stN : null,
  };
}

export function parseRacelist(html: string, rno?: number): Partial<ScrapedEntry>[] {
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
    const setsu = parseSetsu(tb, rno);

    out.push({
      ...setsu,
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
  // 艇番で格納する。以前は出現順に push していたため、1艇でも読み飛ばすと
  // 以降の艇の展示タイムが全部1つずれるバグがあった（ページ末尾には
  // 艇番に見える値を持つゴミtbodyも存在する）。
  const exhibition: ({ exTime: number | null; tilt: number | null } | null)[] = [
    null, null, null, null, null, null,
  ];

  for (const tb of tbodies(html)) {
    const c = cells(tb);
    if (c.length < 6) continue;
    const lane = toNum(stripTags(c[0]));
    if (lane === null || !Number.isInteger(lane) || lane < 1 || lane > 6) continue;
    if (exhibition[lane - 1] !== null) continue; // 同じ艇番の再出現はゴミ行

    const ex = toNum(stripTags(c[4]));
    exhibition[lane - 1] = {
      // 展示タイムの妥当範囲は6秒台。列ずれでチルト値(-0.5〜3.0)等を
      // 拾ってしまうと確率計算が暴走するため、範囲外は欠損扱いにする。
      exTime: ex !== null && ex >= 5.0 && ex <= 9.0 ? ex : null,
      tilt: toNum(stripTags(c[5])),
    };
  }

  const wi = html.indexOf("weather1");
  const block = wi >= 0 ? stripTags(html.slice(wi, wi + 4000)) : "";
  const pick = (re: RegExp) => {
    const m = block.match(re);
    if (!m) return null;
    const v = parseFloat(m[1]);
    // "．"だけ等にマッチすると parseFloat が NaN を返す
    return Number.isFinite(v) ? v : null;
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
  // 出走表ページ上部の「電話投票締切予定」行には12レース分の時刻が並ぶ。
  // 以前は最初にマッチした時刻を返していたため、何Rを聞いても常に
  // 1Rの締切が表示されていた（締切済みのレースを買いに行きかねない）。
  const i = html.indexOf("締切予定");
  if (i < 0) return null;
  const end = html.indexOf("</tr>", i);
  const seg = html.slice(i, end > i ? end : i + 3000);
  const times = [...seg.matchAll(/(\d{1,2}:\d{2})/g)].map((m) => m[1]);
  return times.length >= rno ? times[rno - 1] : null;
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

  const base = parseRacelist(racelistHtml, rno);
  if (base.length < 6) {
    throw new Error(
      "出走表を取得できませんでした（開催なし・未公開・欠場で6艇そろっていない、のいずれか）",
    );
  }

  const before = beforeHtml ? parseBeforeInfo(beforeHtml) : null;
  // 全艇そろったときだけ展示タイムを使う（全か無か）。
  // 一部の艇だけ欠損した状態でモデルに渡すと、学習データの欠損が
  // 「不出走」を意味していたせいで、その艇の確率が実力と無関係に潰れる。
  const hasBeforeInfo =
    !!before && before.exhibition.every((e) => e !== null && e.exTime !== null);

  const entries: ScrapedEntry[] = base.map((e, i) => ({
    ...(e as ScrapedEntry),
    lane: i + 1,
    exTime: hasBeforeInfo ? before!.exhibition[i]!.exTime : null,
    tilt: before?.exhibition[i]?.tilt ?? null,
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
    // オッズは締切直前まで動くのでキャッシュは短めにする
    const html = await get(`${BASE}/odds3t${q(jcd, rno, hd)}`, 15);
    return parseOdds3t(html);
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------- レース結果

export interface RaceResult {
  trifecta: string | null; // "1-4-2"
  payout: number | null; // 100円あたりの払戻金
  popularity: number | null; // 何番人気だったか
}

/**
 * 結果ページの払戻金表から3連単の行を読む。
 *
 *   <td rowspan="2">3連単</td>
 *   <td>... <span class="numberSet1_number is-type1">1</span> - 4 - 2 ...</td>
 *   <td><span class="is-payout1">&yen;1,530</span></td>
 *   <td>3</td>            ← 人気
 *
 * 中止・不成立のレースでは払戻が空になるので、その場合は null を返す。
 */
export function parseRaceResult(html: string): RaceResult {
  const empty: RaceResult = { trifecta: null, payout: null, popularity: null };

  const row = html.match(/3連単<\/td>([\s\S]*?)<\/tr>/);
  if (!row) return empty;

  const numbers = [...row[1].matchAll(/numberSet1_number[^>]*>\s*([1-6])\s*</g)].map(
    (m) => m[1],
  );
  if (numbers.length < 3) return empty;

  const payoutMatch = row[1].match(/is-payout1[^>]*>\s*(?:&yen;|¥)([\d,]+)/);
  const popMatch = row[1].match(/<td>\s*(\d+)\s*<\/td>\s*$/);

  return {
    trifecta: numbers.slice(0, 3).join("-"),
    payout: payoutMatch ? parseInt(payoutMatch[1].replace(/,/g, ""), 10) : null,
    popularity: popMatch ? parseInt(popMatch[1], 10) : null,
  };
}

export async function fetchResult(
  jcd: number,
  rno: number,
  hd: string,
): Promise<RaceResult> {
  try {
    return parseRaceResult(await get(`${BASE}/raceresult${q(jcd, rno, hd)}`));
  } catch {
    return { trifecta: null, payout: null, popularity: null };
  }
}
