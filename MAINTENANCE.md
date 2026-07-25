# 定期メンテナンス

## 結論

**普段やることはありません。** LINEで予想を聞くだけです。

人間の判断が必要なのは「スクレイパーが壊れたとき」だけで、
それも異常を検知したらLINEに通知が来ます。

---

## 自動で動くもの

| いつ | 何が | 失敗したら |
|---|---|---|
| 毎朝 5:00 (JST) | 前日のレース結果を取り込み | 翌日の実行で自動的に追いつく |
| 30分おき | 全開催レースのオッズを保存 | その回のぶんが欠けるだけ |
| 毎日 12:30 | 本番のボットが正常か検証 | LINEに通知が飛ぶ |
| 毎日 23:30 / 翌 5:00 | 予想の答え合わせ | 次回の実行で再試行 |
| 毎週月曜 6:00 | 再学習 → 改善していれば自動デプロイ | LINEに通知が飛ぶ |
| 毎月1日 | スケジュール維持のためのコミット | 警告メールが届く |

### 再学習の採用基準

人間がPRを眺める代わりに、検証データ（学習に使っていない直近3か月）の
指標だけで機械的に判定します。

- LogLoss が 0.001 以上改善している
- 1着的中率が 0.5ポイント以上落ちていない
- 1号艇を買い続けるより良い
- Python と TypeScript の特徴量が完全に一致している（テストが通る）

**すべて満たしたときだけデプロイします。** 満たさなければ何もせず、
現行モデルのまま動き続けます。どちらの場合もLINEに結果が届きます。

LogLoss を主指標にしているのは、3連単の期待値計算が確率の正確さで決まるからです。
「1着を当てた回数」より「確率がどれだけ正確か」の方が重要です。

---

## 初回だけ必要な設定

自動化を全部効かせるには、GitHub に以下を登録してください。
**未設定でも壊れません**（該当する機能だけスキップされます）。

### Settings → Secrets and variables → Actions → Secrets

| 名前 | 用途 | 取得方法 |
|---|---|---|
| `CLOUDFLARE_API_TOKEN` | 自動デプロイ | Cloudflare ダッシュボード → My Profile → API Tokens → "Edit Cloudflare Workers" テンプレート |
| `CLOUDFLARE_ACCOUNT_ID` | 同上 | Workers の画面右側に表示されている |
| `LINE_CHANNEL_ACCESS_TOKEN` | 異常・結果の通知 | `.dev.vars` と同じ値 |
| `LINE_USER_ID` | 同上 | LINEでボットに「whoami」と送ると分かる |

任意（推奨）: Cloudflare 側にも `ADMIN_KEY` を登録すると、
成績ページ（/stats）と答え合わせの起動（/settle）が鍵なしで叩けなくなります。
`.dev.vars` に8文字以上のランダム文字列を書いて `npm run secrets` を実行し、
以後は `/stats?key=<その値>` の形で使います。

### Settings → Secrets and variables → Actions → Variables

| 名前 | 値 |
|---|---|
| `WORKER_URL` | `https://kyoutei-bot.<subdomain>.workers.dev` |

### Settings → Actions → General → Workflow permissions

- **Read and write permissions** を選択

> プッシュ通知は無料枠 月200通 を消費します（応答メッセージは無制限）。
> 通知が飛ぶのは異常時と週1回の学習結果だけなので、まず超えません。

---

## 自動化できないもの

### スクレイパーの修復

boatrace.jp のページ構造が変わると壊れます。**これだけは人間が直すしかありません。**

保存済みHTMLを使う `npm test` では検知できない（テストは通り続ける）ため、
毎日本番のボットに実際にリクエストを投げて確認しています。
異常があればLINEに通知が飛びます。

```bash
cd worker
node scripts/fetch-fixtures.mjs 22 1 20260720   # 最新HTMLを取り直す
npm test                                        # どこが壊れたか確認
# 差分を見ながら src/scrape.ts を直す
npm run deploy
```

### 60日ルール（自動化しているが保証はない）

GitHub は60日間リポジトリに動きがないとスケジュール実行を無効化します。
毎月1日に自動コミットして回避していますが、**これは「コミットがあれば活動中と
みなされる」という挙動に依存しており、確実ではありません。**

無効化される前に GitHub から警告メールが届きます。
届いたら Actions タブから再有効化してください。

---

## 壊れたときの対処

### データベースのキャッシュが消えた

日次データ更新が「レース数が少なすぎます」で止まったら、キャッシュ失効です。
Actions からバックフィルし直してください。

```
start=2023-07-01  end=2024-06-30
start=2024-07-01  end=2025-06-30
start=2025-07-01  end=(今日)
```

直近90日以内なら、Actions の成果物 `kyotei-db-backup` からも復旧できます。

### LINE のトークンを再発行したとき

```bash
cd worker
# .dev.vars を新しい値に書き換えてから
npm run secrets
```

GitHub Secrets の `LINE_CHANNEL_ACCESS_TOKEN` も忘れずに更新してください。
再デプロイは不要です。

### モデルを1つ前に戻したい

```bash
git log --oneline -- worker/src/model.json
git checkout <コミット> -- worker/src/model.json
cd worker && npm run deploy
```

---

## 次にやると良いこと

1. **λ（市場ブレンド率）の実測** — オッズが3週間分（約1万レース）たまったら、
   `0.25` という暫定値が妥当か検証できます
2. **LightGBM への差し替え** — 線形モデルの表現力が今の頭打ちの原因です
3. **平均ST・F数の追加** — 出走表ページにはあるが未使用の特徴量
