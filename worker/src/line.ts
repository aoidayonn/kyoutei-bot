/** LINE Messaging API のユーティリティ（署名検証・返信・Flex Message生成）。 */

import type { Prediction, Pick } from "./predict";

// ---------------------------------------------------------------- 署名検証

/** X-Line-Signature を検証する。改ざん・なりすまし防止のため必須。 */
export async function verifySignature(
  body: string,
  signature: string | null,
  channelSecret: string,
): Promise<boolean> {
  if (!signature) return false;
  // シークレット未設定だと TextEncoder が空文字を鍵にしたHMACとして
  // 「検証が通ってしまう」形で表面化しない。明示的に弾く。
  if (!channelSecret) return false;

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(channelSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body));
  const expected = btoa(String.fromCharCode(...new Uint8Array(mac)));

  // タイミング攻撃を避けるため定数時間比較
  if (expected.length !== signature.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= expected.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return diff === 0;
}

// ---------------------------------------------------------------- 返信

export async function reply(
  replyToken: string,
  messages: unknown[],
  accessToken: string,
): Promise<void> {
  const res = await fetch("https://api.line.me/v2/bot/message/reply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ replyToken, messages: messages.slice(0, 5) }),
  });
  if (!res.ok) {
    console.error("LINE reply failed", res.status, await res.text());
  }
}

export function textMessage(text: string) {
  return { type: "text", text: text.slice(0, 4900) };
}

// ---------------------------------------------------------------- Flex Message

const COLORS = {
  text: "#333333",
  sub: "#888888",
  good: "#2A7D2E",
  bad: "#C0392B",
  line: "#E5E5E5",
};

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function pickRow(p: Pick, highlight: boolean) {
  const evText = p.ev === null ? "—" : p.ev.toFixed(2);
  const evColor = p.ev === null ? COLORS.sub : p.ev >= 1.0 ? COLORS.good : COLORS.bad;
  return {
    type: "box",
    layout: "horizontal",
    margin: "sm",
    contents: [
      {
        type: "text",
        text: p.combo,
        size: "sm",
        weight: highlight ? "bold" : "regular",
        color: COLORS.text,
        flex: 3,
      },
      { type: "text", text: pct(p.prob), size: "sm", color: COLORS.sub, flex: 2, align: "end" },
      {
        type: "text",
        text: p.odds === null ? "—" : `${p.odds.toFixed(1)}倍`,
        size: "sm",
        color: COLORS.sub,
        flex: 3,
        align: "end",
      },
      { type: "text", text: evText, size: "sm", color: evColor, weight: "bold", flex: 2, align: "end" },
    ],
  };
}

function header(label: string) {
  return {
    type: "box",
    layout: "horizontal",
    margin: "md",
    contents: [
      { type: "text", text: "買い目", size: "xxs", color: COLORS.sub, flex: 3 },
      { type: "text", text: "確率", size: "xxs", color: COLORS.sub, flex: 2, align: "end" },
      { type: "text", text: "オッズ", size: "xxs", color: COLORS.sub, flex: 3, align: "end" },
      { type: "text", text: "期待値", size: "xxs", color: COLORS.sub, flex: 2, align: "end" },
    ],
  };
}

function sectionTitle(text: string, color: string) {
  return {
    type: "text",
    text,
    size: "sm",
    weight: "bold",
    color,
    margin: "lg",
  };
}

export function predictionFlex(p: Prediction) {
  const contents: unknown[] = [];

  // 期待値トップ
  if (p.byEv.length > 0) {
    contents.push(sectionTitle("推奨買い目（期待値1.05超・確率順）", COLORS.good));
    contents.push(header("ev"));
    p.byEv.forEach((x) => contents.push(pickRow(x, true)));
  } else {
    contents.push(sectionTitle("推奨買い目（期待値1.05超・確率順）", COLORS.good));
    contents.push({
      type: "text",
      text: p.hasOdds
        ? "期待値が基準を超える買い目なし → 見送り推奨"
        : "オッズ未取得のため判定できません",
      size: "sm",
      color: COLORS.bad,
      wrap: true,
      margin: "sm",
    });
  }

  // 的中率トップ
  contents.push(sectionTitle("的中率トップ", COLORS.text));
  contents.push(header("prob"));
  p.byProb.forEach((x) => contents.push(pickRow(x, false)));

  // 根拠
  contents.push(sectionTitle("根拠", COLORS.text));
  p.reasons.forEach((r) =>
    contents.push({
      type: "text",
      text: `・${r}`,
      size: "xs",
      color: COLORS.sub,
      wrap: true,
      margin: "sm",
    }),
  );

  const subtitleParts = [
    p.deadline ? `締切 ${p.deadline}` : null,
    p.hasBeforeInfo ? null : "展示前",
  ].filter(Boolean);

  return {
    type: "flex",
    altText: `${p.stadium} ${p.rno}R の予想`,
    contents: {
      type: "bubble",
      size: "mega",
      header: {
        type: "box",
        layout: "vertical",
        backgroundColor: "#1B4F72",
        paddingAll: "md",
        contents: [
          {
            type: "text",
            text: `${p.stadium} ${p.rno}R`,
            color: "#FFFFFF",
            size: "lg",
            weight: "bold",
          },
          {
            type: "text",
            text: subtitleParts.join(" / ") || " ",
            color: "#CFE2F3",
            size: "xs",
          },
        ],
      },
      body: { type: "box", layout: "vertical", paddingAll: "md", contents },
      footer: {
        type: "box",
        layout: "vertical",
        contents: [
          {
            type: "text",
            text: "予測であり的中を保証するものではありません。舟券の払戻率は約75%です。",
            size: "xxs",
            color: COLORS.sub,
            wrap: true,
          },
        ],
      },
    },
  };
}

/** Flex が使えない場面のためのテキスト版。 */
export function predictionText(p: Prediction): string {
  const lines: string[] = [];
  lines.push(`🚤 ${p.stadium} ${p.rno}R${p.deadline ? `  締切 ${p.deadline}` : ""}`);
  if (!p.hasBeforeInfo) lines.push("（直前情報が未公開のため暫定予想）");
  lines.push("");

  lines.push("【推奨買い目（期待値1.05超・確率順）】");
  if (p.byEv.length) {
    p.byEv.forEach((x, i) =>
      lines.push(
        `${i + 1}. ${x.combo}  ${pct(x.prob)} / ${x.odds?.toFixed(1)}倍 / EV ${x.ev?.toFixed(2)}`,
      ),
    );
  } else {
    lines.push(p.hasOdds ? "該当なし → 見送り推奨" : "オッズ未取得");
  }

  lines.push("");
  lines.push("【的中率トップ】");
  p.byProb.forEach((x, i) =>
    lines.push(
      `${i + 1}. ${x.combo}  ${pct(x.prob)} / ${x.odds ? `${x.odds.toFixed(1)}倍 / EV ${x.ev?.toFixed(2)}` : "オッズ未取得"}`,
    ),
  );

  lines.push("");
  lines.push("【根拠】");
  p.reasons.forEach((r) => lines.push(`・${r}`));
  lines.push("");
  lines.push("※予測であり的中を保証するものではありません");

  return lines.join("\n");
}
