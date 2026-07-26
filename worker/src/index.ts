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
import { todayJst } from "./time";
import { formatSummary, summarize, type PredictionRow } from "./stats";

export interface Env {
  LINE_CHANNEL_SECRET: string;
  LINE_CHANNEL_ACCESS_TOKEN: string;
  ALLOWED_USER_ID?: string; // 自分だけが使えるようにする（任意）
  /** 設定すると /stats /settle に ?key= または X-Admin-Key ヘッダが必要になる。
   *  成績（個人データ）の閲覧と、答え合わせの起動（D1書き込み+外部fetch）を
   *  第三者に開放しないための鍵。未設定なら従来通り誰でも叩ける。 */
  ADMIN_KEY?: string;
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
      if (!authorized(url, request, env)) return new Response("forbidden", { status: 403 });
      return new Response(await buildStats(env), {
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    }

    // 手動で答え合わせを走らせる（Cronを待たずに確認したいとき）
    if (url.pathname === "/settle" && request.method === "POST") {
      if (!authorized(url, request, env)) return new Response("forbidden", { status: 403 });
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
    // 未確定件数はウィンドウ外も含めて正確に数える
    const unsettledRow = await env.DB.prepare(
      "SELECT COUNT(*) AS n FROM predictions WHERE settled_at IS NULL",
    ).first<{ n: number }>();

    const { results } = await env.DB.prepare(
      `SELECT race_id, hd, jcd, rno, verdict, picks_json, win_probs_json, actual, payout
         FROM predictions
        WHERE actual IS NOT NULL
        ORDER BY hd DESC, predicted_at DESC LIMIT 1000`,
    ).all<PredictionRow>();

    return formatSummary(summarize(results ?? []), unsettledRow?.n ?? 0);
  } catch (e) {
    console.error(e);
    return "成績の集計に失敗しました。";
  }
}

/**
 * ADMIN_KEY が設定されていれば照合する。
 * /stats は個人の成績、/settle はD1書き込みを伴うため、公開URLのままにしない。
 */
function authorized(url: URL, request: Request, env: Env): boolean {
  if (!env.ADMIN_KEY) return true; // 未設定なら開放（個人用の既定）
  const key = url.searchParams.get("key") ?? request.headers.get("x-admin-key");
  return key === env.ADMIN_KEY;
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

  let events: LineEvent[] = [];
  try {
    const payload = JSON.parse(body) as { events?: LineEvent[] };
    events = payload.events ?? [];
  } catch {
    // 署名は正しいのにJSONが壊れているのは想定外だが、
    // 500を返すとLINEが同じボディを再送し続けるため200で受け流す
    console.error("webhook body is not JSON");
    return new Response("ok");
  }

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
    // 返信が最優先（replyTokenには期限がある）。記録はその後。
    await reply(replyToken, [predictionFlex(p) as unknown], env.LINE_CHANNEL_ACCESS_TOKEN);
    await savePrediction(env, p);
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
    const sorted = [...found].sort((a, b) => a - b);
    const names = sorted.map((j) => `${j} ${STADIUMS[j]}`);
    // 例に使う場は一覧の先頭と揃える（Setの挿入順だと一覧と食い違う）
    return `本日（${hd}）の開催場\n\n${names.join("\n")}\n\n例：「${STADIUMS[sorted[0]]} 12」`;
  } catch {
    return "本日の開催情報を取得できませんでした。";
  }
}

/** 後から的中率・回収率を検証できるよう、予想をD1に残す。 */
async function savePrediction(env: Env, p: Awaited<ReturnType<typeof predict>>) {
  if (!env.DB) return;
  try {
    // INSERT OR REPLACE は行をDELETE→INSERTするため、答え合わせ済みの
    // actual / payout / settled_at がNULLに戻ってしまう（レース後に同じ
    // レースを聞き直すだけで成績データが壊れる）。
    // 未確定の行だけ予想を上書きし、確定済みの行には触らない。
    await env.DB.prepare(
      `INSERT INTO predictions
         (race_id, hd, jcd, rno, predicted_at, verdict, picks_json, win_probs_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(race_id) DO UPDATE SET
         predicted_at   = excluded.predicted_at,
         verdict        = excluded.verdict,
         picks_json     = excluded.picks_json,
         win_probs_json = excluded.win_probs_json
       WHERE predictions.settled_at IS NULL`,
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

  if (!STADIUMS[jcd] || !(rno >= 1 && rno <= 12) || !/^\d{8}$/.test(hd)) {
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
