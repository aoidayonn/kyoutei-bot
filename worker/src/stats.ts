/**
 * 実運用成績の集計。
 *
 * バックテストの数字は過去データ上のもので、実際に運用したときの成績とは別物。
 * 「モデルの言うことが本当に当たっているか」を測れるのはここだけなので、
 * 予想を出すたびに D1 へ記録し、後から結果を突き合わせている。
 *
 * D1 に依存しない純粋な関数にしてあるのはテストのため。
 */

export interface PredictionRow {
  race_id: string;
  hd: string;
  jcd: number;
  rno: number;
  verdict: string; // "buy" | "skip"
  picks_json: string; // {byEv:[...], byProb:[...]}
  win_probs_json: string;
  actual: string | null; // "1-4-2"
  payout: number | null; // 100円あたりの払戻金
}

export interface Summary {
  /** 予想を出したレース数（結果が確定しているもの） */
  settled: number;
  /** うち「買い」判定だったレース数 */
  buyRaces: number;
  /** 見送り判定だったレース数 */
  skipRaces: number;
  /** 推奨買い目の総点数 */
  totalPicks: number;
  /** 推奨が的中したレース数 */
  hitRaces: number;
  /** 投資額（100円 × 点数） */
  invested: number;
  /** 払戻総額 */
  returned: number;
  /** 回収率。投資0なら null */
  roi: number | null;
  /** レース的中率。買ったレースが0なら null */
  hitRate: number | null;
  /** 「的中率トップ1点」だけ買っていた場合の的中率（比較用） */
  topPickHitRate: number | null;
  /** 見送ったレースのうち、実際に的中率トップが当たっていた割合（見送りの妥当性） */
  skipWouldHaveHit: number | null;
}

interface Pick {
  combo: string;
  prob: number;
  odds: number | null;
  ev: number | null;
}

function parsePicks(json: string): { byEv: Pick[]; byProb: Pick[] } {
  try {
    const o = JSON.parse(json);
    return { byEv: o.byEv ?? [], byProb: o.byProb ?? [] };
  } catch {
    return { byEv: [], byProb: [] };
  }
}

export function summarize(rows: PredictionRow[]): Summary {
  let settled = 0;
  let buyRaces = 0;
  let skipRaces = 0;
  let totalPicks = 0;
  let hitRaces = 0;
  let invested = 0;
  let returned = 0;
  let topPickTried = 0;
  let topPickHit = 0;
  let skipTried = 0;
  let skipHit = 0;

  for (const r of rows) {
    if (!r.actual) continue; // 未確定
    settled += 1;

    const { byEv, byProb } = parsePicks(r.picks_json);

    // 比較用: 常に「的中率トップ1点」だけ買っていたら
    if (byProb.length) {
      topPickTried += 1;
      if (byProb[0].combo === r.actual) topPickHit += 1;
    }

    if (byEv.length === 0) {
      skipRaces += 1;
      // 見送り判断が妥当だったか（本命が当たっていたなら見送りは損）
      if (byProb.length) {
        skipTried += 1;
        if (byProb[0].combo === r.actual) skipHit += 1;
      }
      continue;
    }

    buyRaces += 1;
    totalPicks += byEv.length;
    invested += 100 * byEv.length;

    const hit = byEv.find((p) => p.combo === r.actual);
    if (hit) {
      hitRaces += 1;
      // 払戻金が取れていない場合は、予想時のオッズで代用する
      returned += r.payout ?? Math.round((hit.odds ?? 0) * 100);
    }
  }

  return {
    settled,
    buyRaces,
    skipRaces,
    totalPicks,
    hitRaces,
    invested,
    returned,
    roi: invested > 0 ? returned / invested : null,
    hitRate: buyRaces > 0 ? hitRaces / buyRaces : null,
    topPickHitRate: topPickTried > 0 ? topPickHit / topPickTried : null,
    skipWouldHaveHit: skipTried > 0 ? skipHit / skipTried : null,
  };
}

/** LINE に返すテキストへ整形する。 */
export function formatSummary(s: Summary, unsettled: number): string {
  if (s.settled === 0) {
    return [
      "📊 実運用成績",
      "",
      "まだ結果が確定した予想がありません。",
      unsettled > 0 ? `（結果待ち ${unsettled} 件）` : "",
      "",
      "レースの予想を出すと自動で記録され、",
      "翌日までに答え合わせされます。",
    ]
      .filter(Boolean)
      .join("\n");
  }

  const pct = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`);
  const lines = [
    "📊 実運用成績",
    "",
    `記録したレース: ${s.settled}件` + (unsettled ? `（結果待ち ${unsettled}件）` : ""),
    `　買い ${s.buyRaces}件 / 見送り ${s.skipRaces}件`,
    "",
    "【推奨買い目を買っていた場合】",
    `　投資 ${s.invested.toLocaleString()}円 / 払戻 ${s.returned.toLocaleString()}円`,
    `　回収率 ${pct(s.roi)}`,
    `　的中率 ${pct(s.hitRate)}（${s.hitRaces}/${s.buyRaces}レース・計${s.totalPicks}点）`,
    "",
    "【比較】",
    `　的中率トップ1点だけ買った場合の的中率 ${pct(s.topPickHitRate)}`,
  ];

  if (s.skipWouldHaveHit !== null) {
    lines.push(`　見送ったレースで本命が当たっていた割合 ${pct(s.skipWouldHaveHit)}`);
  }

  lines.push("");
  if (s.settled < 100) {
    lines.push(
      `⚠️ まだ${s.settled}件しかないので、この数字はほぼ運です。`,
      "判断できるのは数百件たまってからです。",
    );
  } else if (s.roi !== null && s.roi < 1.0) {
    lines.push("※ 回収率が100%未満です。長期的には負けます。");
  }

  return lines.join("\n");
}
