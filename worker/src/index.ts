/**
 * ボートレース3連単予測ボット — Cloudflare Workers エントリポイント
 *
 * ルート:
 *   POST /webhook          LINE Messaging API からのWebhook
 *   GET  /predict?...      ブラウザから動作確認するためのデバッグ用
 *   GET  /health           疎通確認
 */

import { predict } from "./predict";
import { parseCommand, STADIUMS, stadiumName } from "./stadiums";
import { predictionFlex, predictionText, reply, textMessage, verifySignature } from "./line";
import { settlePredictions } from "./settle";
import { formatSummary, summarize, type PredictionRow } from "./stats";

export interface Env {
  LINE_CHANNEL_SECRET: string;
  LINE_CHANNEL_ACCESS_TOKEN: string;
  ALLOWED_USER_ID?: string; // 自分だけが使えるようにする（任意）
  DB?: D1Database;
}

const HELP = [
  "レース場とレース番号を送ってください。",
  "",
  "例：",
  "  大村 12",
  "  住之江5",
  "  24 12   （場コードでも可）",
  "",
  "「今日」と送ると本日の開催場一覧を返します。",
  "「成績」と送るとこれまでの予想の実績を表示します。",
  "「whoami」と送ると自分のuserIdを表示します。",
].join("\n");

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response("ok");
    }

    if (url.pathname === "/webhook" && request.method === "POST") {
      return handleWebhook(request, env, ctx);
    }

    if (url.pathname === "/predict") {
      return handleDebugPredict(url);
    }

    if (url.pathname === "/stats") {
      return new Response(await buildStats(env), {
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }

    // 手動で答え合わせを走らせる（Cronを待たずに確認したいとき）
    if (url.pathname === "/settle" && request.method === "POST") {
      if (!env.DB) return new Response("D1が設定されていません", { status: 500 });
      return Response.json(await settlePredictions(env.DB));
    }

    return new Response("kyoutei-bot", { status: 200 });
  },

  /**
   * Cron Trigger。予想の答え合わせを行う。
   * これがないと predictions に未確定の行が溜まり続け、
   * 「実際に当たっていたのか」が永久に分からなくなる。
   */
  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext) {
    if (!env.DB) {
      console.log("D1が設定されていないため答え合わせをスキップ");
      return;
    }
    ctx.waitUntil(
      settlePredictions(env.DB)
        .then((r) =>
          console.log(
            `答え合わせ: ${r.settled}/${r.checked} 件を確定（残り ${r.pending} 件）`,
          ),
        )
        .catch((e) => console.error("答え合わせに失敗", e)),
    );
  },
};

// ---------------------------------------------------------------- 成績

async function buildStats(env: Env): Promise<string> {
  if (!env.DB) return "D1が設定されていないため成績を記録していません。";

  try {
    const { results } = await env.DB.prepare(
      `SELECT race_id, hd, jcd, rno, verdict, picks_json, win_probs_json, actual, payout
         FROM predictions ORDER BY hd DESC LIMIT 1000`,
    ).all<PredictionRow>();

    const rows = results ?? [];
    const unsettled = rows.filter((r) => !r.actual).length;
    return formatSummary(summarize(rows), unsettled);
  } catch (e) {
    console.error(e);
    return "成績の集計に失敗しました。";
  }
}

// ---------------------------------------------------------------- Webhook

async function handleWebhook(request: Request, env: Env, ctx: ExecutionContext) {
  const body = await request.text();
  const ok = await verifySignature(
    body,
    request.headers.get("x-line-signature"),
    env.LINE_CHANNEL_SECRET,
  );
  if (!ok) return new Response("invalid signature", { status: 401 });

  const payload = JSON.parse(body) as { events?: LineEvent[] };
  const events = payload.events ?? [];

  // LINE は10秒以内の200応答を期待するので、処理は waitUntil に逃がす
  ctx.waitUntil(
    Promise.all(events.map((e) => handleEvent(e, env).catch((err) => console.error(err)))),
  );

  return new Response("ok");
}

interface LineEvent {
  type: string;
  replyToken?: string;
  source?: { userId?: string };
  message?: { type: string; text?: string };
}

async function handleEvent(event: LineEvent, env: Env) {
  if (event.type !== "message" || event.message?.type !== "text") return;
  const replyToken = event.replyToken;
  if (!replyToken) return;

  // 個人用ボットなので、設定されていれば自分以外は弾く
  if (env.ALLOWED_USER_ID && event.source?.userId !== env.ALLOWED_USER_ID) {
    await reply(replyToken, [textMessage("このボットは個人用です。")], env.LINE_CHANNEL_ACCESS_TOKEN);
    return;
  }

  const text = (event.message.text ?? "").trim();
  const userId = event.source?.userId;

  // userId は「メッセージを受け取って初めて分かる」値なので、
  // wrangler tail で拾えるようにログに出しておく。
  console.log(`message from ${userId ?? "unknown"}: ${text}`);

  // ALLOWED_USER_ID に設定する値を確認するためのコマンド
  if (/^(whoami|id|ユーザーid|ユーザーID)$/i.test(text)) {
    await reply(
      replyToken,
      [
        textMessage(
          userId
            ? `あなたのuserIdは\n\n${userId}\n\nこれを worker/.dev.vars の ALLOWED_USER_ID に設定して npm run secrets を実行すると、自分以外からのメッセージを無視するようになります。`
            : "userIdを取得できませんでした（グループやルームからの送信の可能性があります）",
        ),
      ],
      env.LINE_CHANNEL_ACCESS_TOKEN,
    );
    return;
  }

  if (!text || /^(help|ヘルプ|使い方|？|\?)$/i.test(text)) {
    await reply(replyToken, [textMessage(HELP)], env.LINE_CHANNEL_ACCESS_TOKEN);
    return;
  }

  if (/^(成績|せいせき|実績|stats)$/i.test(text)) {
    await reply(
      replyToken,
      [textMessage(await buildStats(env))],
      env.LINE_CHANNEL_ACCESS_TOKEN,
    );
    return;
  }

  if (/^(今日|本日|きょう)$/.test(text)) {
    await reply(
      replyToken,
      [textMessage(await todayStadiums())],
      env.LINE_CHANNEL_ACCESS_TOKEN,
    );
    return;
  }

  const parsed = parseCommand(text);
  if ("error" in parsed) {
    await reply(
      replyToken,
      [textMessage(`${parsed.error}\n\n${HELP}`)],
      env.LINE_CHANNEL_ACCESS_TOKEN,
    );
    return;
  }

  const hd = todayJst();
  try {
    const p = await predict(parsed.jcd, parsed.rno, hd);
    await savePrediction(env, p);
    await reply(replyToken, [predictionFlex(p) as unknown], env.LINE_CHANNEL_ACCESS_TOKEN);
  } catch (err) {
    console.error(err);
    const msg =
      err instanceof Error && err.message.includes("出走表")
        ? `${stadiumName(parsed.jcd)} ${parsed.rno}R の出走表が見つかりませんでした。本日開催がないか、まだ公開されていない可能性があります。`
        : "予想の取得に失敗しました。少し待ってからもう一度送ってください。";
    await reply(replyToken, [textMessage(msg)], env.LINE_CHANNEL_ACCESS_TOKEN);
  }
}

// ---------------------------------------------------------------- 補助

/** 日本時間の YYYYMMDD */
function todayJst(): string {
  const now = new Date(Date.now() + 9 * 3600 * 1000);
  return now.toISOString().slice(0, 10).replace(/-/g, "");
}

async function todayStadiums(): Promise<string> {
  const hd = todayJst();
  try {
    const res = await fetch(`https://www.boatrace.jp/owpc/pc/race/index?hd=${hd}`, {
      headers: { "User-Agent": "kyoutei-bot/1.0" },
    });
    const html = await res.text();
    const found = new Set<number>();
    for (const m of html.matchAll(/jcd=(\d{2})/g)) {
      const n = parseInt(m[1], 10);
      if (STADIUMS[n]) found.add(n);
    }
    if (!found.size) return "本日の開催情報を取得できませんでした。";
    const names = [...found].sort((a, b) => a - b).map((j) => `${j} ${STADIUMS[j]}`);
    return `本日（${hd}）の開催場\n\n${names.join("\n")}\n\n例：「${STADIUMS[[...found][0]]} 12」`;
  } catch {
    return "本日の開催情報を取得できませんでした。";
  }
}

/** 後から的中率・回収率を検証できるよう、予想をD1に残す。 */
async function savePrediction(env: Env, p: Awaited<ReturnType<typeof predict>>) {
  if (!env.DB) return;
  try {
    await env.DB.prepare(
      `INSERT OR REPLACE INTO predictions
         (race_id, hd, jcd, rno, predicted_at, verdict, picks_json, win_probs_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        `${p.hd}-${String(p.jcd).padStart(2, "0")}-${String(p.rno).padStart(2, "0")}`,
        p.hd,
        p.jcd,
        p.rno,
        new Date().toISOString(),
        p.verdict,
        JSON.stringify({ byEv: p.byEv, byProb: p.byProb }),
        JSON.stringify(p.winProbs),
      )
      .run();
  } catch (e) {
    console.error("savePrediction failed", e);
  }
}

// ---------------------------------------------------------------- デバッグ用

async function handleDebugPredict(url: URL): Promise<Response> {
  const jcd = parseInt(url.searchParams.get("jcd") ?? "", 10);
  const rno = parseInt(url.searchParams.get("rno") ?? "", 10);
  const hd = url.searchParams.get("hd") ?? todayJst();

  if (!STADIUMS[jcd] || !(rno >= 1 && rno <= 12)) {
    return new Response("usage: /predict?jcd=24&rno=12&hd=20260725", { status: 400 });
  }

  try {
    const p = await predict(jcd, rno, hd);
    const format = url.searchParams.get("format");
    if (format === "json") {
      return Response.json(p);
    }
    return new Response(predictionText(p), {
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  } catch (err) {
    return new Response(String(err), { status: 500 });
  }
}
