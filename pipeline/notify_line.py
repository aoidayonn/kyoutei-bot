"""LINEへプッシュ通知を送る（ワークフローの失敗を知らせるため）。

    python notify_line.py "本文"

環境変数:
    LINE_CHANNEL_ACCESS_TOKEN  必須
    LINE_USER_ID               必須（自分のuserId。LINEで「whoami」と送ると分かる）

未設定なら何もせず正常終了する。通知を使わない構成でも壊れないようにするため。

注意: プッシュメッセージは無料枠 月200通 を消費する（応答メッセージは無制限）。
失敗通知は滅多に飛ばないので問題にならないが、乱発しないこと。
"""
from __future__ import annotations

import os
import sys

import requests

API = "https://api.line.me/v2/bot/message/push"


def notify(text: str) -> bool:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")

    if not token or not user_id:
        print("LINE通知は未設定のためスキップします")
        return False

    try:
        res = requests.post(
            API,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]},
            timeout=20,
        )
    except requests.RequestException as e:
        print(f"LINE通知の送信に失敗: {e}")
        return False

    if res.status_code != 200:
        print(f"LINE通知の送信に失敗: {res.status_code} {res.text[:200]}")
        return False

    print("LINEへ通知しました")
    return True


if __name__ == "__main__":
    notify(sys.argv[1] if len(sys.argv) > 1 else "通知")
