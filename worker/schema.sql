-- D1 スキーマ
-- 予想を保存しておくことで、後から「実際に的中率・回収率がどうだったか」を検証できる。
-- これがないと改善のしようがないので、運用開始時から必ず残す。

CREATE TABLE IF NOT EXISTS predictions (
    race_id        TEXT PRIMARY KEY,   -- "20260725-24-12"
    hd             TEXT NOT NULL,      -- "20260725"
    jcd            INTEGER NOT NULL,
    rno            INTEGER NOT NULL,
    predicted_at   TEXT NOT NULL,
    verdict        TEXT NOT NULL,      -- "buy" | "skip"
    picks_json     TEXT NOT NULL,      -- {byEv:[...], byProb:[...]}
    win_probs_json TEXT NOT NULL,      -- [p1..p6]
    -- 後から結果を突き合わせて埋める
    actual         TEXT,               -- "1-4-2"
    payout         INTEGER,
    settled_at     TEXT,
    -- 結果取得に失敗した回数。上限に達したら諦めて後回しにする
    -- （中止レース等で永久に結果が出ない行が答え合わせを塞ぐのを防ぐ）。
    -- 既存DBには settle.ts が起動時に ALTER TABLE で自動追加する。
    attempts       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_predictions_hd ON predictions(hd);
CREATE INDEX IF NOT EXISTS idx_predictions_settled ON predictions(settled_at);
