"""新しいモデルが古いモデルより良いかを機械的に判定する。

週次の自動学習で「人間がPRを見て判断する」を置き換えるための基準。
主観を挟まないよう、検証データ（学習に使っていない期間）の指標だけで決める。

    python compare_models.py old.json new.json
    → 良くなっていれば終了コード0、そうでなければ1

判定基準
--------
主指標は LogLoss。3連単の期待値計算は確率の正確さで決まるので、
「1着を当てた回数」よりも「確率がどれだけ正確か」の方が重要。

1着的中率は補助指標。LogLossが改善していても的中率が大きく落ちるなら、
何かがおかしいので採用しない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# LogLoss はこれ以上改善していないと「誤差の範囲」とみなす
MIN_LOGLOSS_GAIN = 0.001
# 1着的中率がこれ以上落ちたら、LogLossが良くても採用しない
MAX_ACCURACY_DROP = 0.005


def load_metrics(path: Path):
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return (model.get("metrics") or {}).get("test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    a = p.parse_args()

    old = load_metrics(Path(a.old))
    new = load_metrics(Path(a.new))

    if new is None:
        print("新しいモデルに検証指標がありません。採用しません。")
        sys.exit(1)

    if old is None:
        # 以前は「比較対象がない＝無条件採用」だったが、これは fail-open。
        # 現行指標が読めない原因はファイル欠損・パース失敗・検証データ不足の
        # いずれかで、どれも「壊れた状態で自動デプロイしてよい」理由にならない。
        print("現行モデルの指標を読めませんでした。安全側に倒して採用を見送ります。")
        print("（初回セットアップ等で意図的に採用したい場合は force を指定）")
        sys.exit(1)

    d_logloss = old["logloss"] - new["logloss"]      # 正なら改善
    d_acc = new["win_accuracy"] - old["win_accuracy"]  # 正なら改善

    print("検証データでの比較（学習に使っていない期間）")
    print(f"  LogLoss    {old['logloss']:.4f} → {new['logloss']:.4f}  ({d_logloss:+.4f})")
    print(f"  1着的中率  {old['win_accuracy']:.2%} → {new['win_accuracy']:.2%}  ({d_acc:+.2%})")
    print(f"  基準線     1号艇固定 {new['baseline_lane1']:.2%}")

    if new["win_accuracy"] <= new["baseline_lane1"]:
        print("\n✖ 1号艇を買い続けるより悪いモデルです。採用しません。")
        sys.exit(1)

    if d_acc < -MAX_ACCURACY_DROP:
        print(f"\n✖ 1着的中率が {abs(d_acc):.2%} 落ちています。採用しません。")
        sys.exit(1)

    if d_logloss < MIN_LOGLOSS_GAIN:
        print("\n― LogLoss の改善が誤差の範囲です。現行モデルのままにします。")
        sys.exit(1)

    print("\n✔ 改善しているので新しいモデルを採用します。")
    sys.exit(0)


if __name__ == "__main__":
    main()
