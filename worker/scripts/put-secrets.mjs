/**
 * .dev.vars の値を Cloudflare Workers のシークレットとして登録する。
 *
 *   npm run secrets
 *
 * なぜスクリプトにするか
 * ----------------------
 * LINEのアクセストークンは170文字前後あり、`+` `/` `=` を含みます。
 * 手で打つと必ずどこかで間違えますが、間違えても wrangler は成功と表示するため、
 * 「デプロイは通ったのにWebhookが401」という分かりにくい失敗になります。
 * ファイルから直接流し込めばこの事故が起きません。
 */

import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const devVars = join(here, "..", ".dev.vars");

if (!existsSync(devVars)) {
  console.error("✖ .dev.vars がありません。");
  console.error("  cp .dev.vars.example .dev.vars   としてから実際の値を書いてください。");
  process.exit(1);
}

// KEY=VALUE を読む。値に `=` が含まれるので最初の `=` だけで分割する。
const vars = new Map();
for (const line of readFileSync(devVars, "utf-8").split(/\r?\n/)) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) continue;
  const eq = trimmed.indexOf("=");
  if (eq < 1) continue;
  vars.set(trimmed.slice(0, eq).trim(), trimmed.slice(eq + 1).trim());
}

// 値の形が明らかにおかしい場合はここで止める
const checks = {
  LINE_CHANNEL_SECRET: {
    required: true,
    test: (v) => /^[0-9a-f]{32}$/i.test(v),
    hint: "32桁の16進数です。LINE Developers > チャネル基本設定 > チャネルシークレット",
  },
  LINE_CHANNEL_ACCESS_TOKEN: {
    required: true,
    test: (v) => v.length >= 100,
    hint: "170文字前後の長い文字列です。LINE Developers > Messaging API設定 > チャネルアクセストークン（長期）",
  },
  ALLOWED_USER_ID: {
    required: false,
    test: (v) => v === "" || /^U[0-9a-f]{32}$/i.test(v),
    hint: "U で始まる33文字です。空でも構いません（その場合は登録をスキップします）",
  },
  ADMIN_KEY: {
    required: false,
    test: (v) => v === "" || v.length >= 8,
    hint: "8文字以上のランダムな文字列。/stats /settle の保護用（空ならスキップ）",
  },
};

let hasError = false;
for (const [key, check] of Object.entries(checks)) {
  const value = vars.get(key);
  if (value === undefined || value === "") {
    if (check.required) {
      console.error(`✖ ${key} が .dev.vars に設定されていません`);
      console.error(`  ${check.hint}`);
      hasError = true;
    }
    continue;
  }
  if (!check.test(value)) {
    console.error(`✖ ${key} の形式が正しくありません（長さ ${value.length}）`);
    console.error(`  ${check.hint}`);
    hasError = true;
  }
}
if (hasError) {
  console.error("\n値を直してからもう一度実行してください。");
  process.exit(1);
}

// Windows では Node 18.20 以降、.cmd ファイルを shell なしで spawn すると
// EINVAL になる（セキュリティ修正の副作用）。npx.cmd を直接叩かず、
// wrangler の JS 本体を node で実行することで OS 差を吸収する。
const wranglerJs = join(here, "..", "node_modules", "wrangler", "bin", "wrangler.js");

function putSecret(name, value) {
  return new Promise((resolve, reject) => {
    const useDirect = existsSync(wranglerJs);
    const child = useDirect
      ? spawn(process.execPath, [wranglerJs, "secret", "put", name], {
          stdio: ["pipe", "inherit", "inherit"],
        })
      : spawn("npx", ["wrangler", "secret", "put", name], {
          stdio: ["pipe", "inherit", "inherit"],
          shell: true, // npx.cmd を解決するために必要
        });

    child.stdin.write(value);
    child.stdin.end();
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${name} の登録に失敗しました (exit ${code})`)),
    );
    child.on("error", reject);
  });
}

const targets = ["LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"];
if (vars.get("ALLOWED_USER_ID")) targets.push("ALLOWED_USER_ID");
if (vars.get("ADMIN_KEY")) targets.push("ADMIN_KEY");

for (const name of targets) {
  const value = vars.get(name);
  console.log(`\n→ ${name} を登録します（${value.length} 文字）`);
  await putSecret(name, value);
}

console.log("\n✔ すべて登録しました。 npm run deploy を実行してください。");
