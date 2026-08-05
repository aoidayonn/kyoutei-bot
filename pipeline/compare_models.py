"""新しいモデルが古いモデルより良いかを機械的に判定する。

週次の自動学習で「人間がPRを見て判断する」を置き換えるための基準。
主観を挟まないよう、検証データ（学習に使っていない期間）の指標だけで決める。

    python compare_models.py old.json new.json
    終了コード:
      0 = 改善（採用）
      1 = 同等（誤差の範囲。レシピは同じなのでデータ更新だけ配備してよい）
      2 = 劣化（配備してはいけない）

判定基準
--------
主指標は **3連単NLL**（商品そのものの対数尤度）。かつては1着LogLossを
主指標にしていたが、「1着のみ学習で1着LLは改善、3連単は悪化」という
モデルを通してしまったため、商品指標に変えた。

1着的中率は補助指標。NLLが改善していても的中率が大きく落ちるなら、
何かがおかしいので採用しない。
（旧形式のファイルに3連単NLLが無い場合だけLogLossで判定する）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 3連単NLL（商品指標）がこれ以上改善していないと「誤差の範囲」とみなす。
# 週ごとの温度再推定のゆらぎより大きい値にしている
MIN_NLL3_GAIN = 0.002
# 後方互換（3連単NLLが無い旧形式ファイル用）
MIN_LOGLOSS_GAIN = 0.001
# 1着的中率がこれ以上落ちたら、NLLが良くても採用しない
MAX_ACCURACY_DROP = 0.005
# 3連単NLLがこれ以上悪化していたら「劣化」とみなす（データ更新配備も止める）
MAX_NLL3_REGRESSION = 0.005


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

    d_acc = new["win_accuracy"] - old["win_accuracy"]  # 正なら改善

    use_nll3 = "trifecta_nll" in old and "trifecta_nll" in new
    print("検証データでの比較（新旧とも出荷状態のまま・同一期間で採点）")
    if use_nll3:
        d_main = old["trifecta_nll"] - new["trifecta_nll"]  # 正なら改善
        print(f"  3連単NLL   {old['trifecta_nll']:.4f} → {new['trifecta_nll']:.4f}  ({d_main:+.4f})")
        min_gain, label = MIN_NLL3_GAIN, "3連単NLL"
    else:
        d_main = old["logloss"] - new["logloss"]
        min_gain, label = MIN_LOGLOSS_GAIN, "LogLoss"
    print(f"  1着LogLoss {old['logloss']:.4f} → {new['logloss']:.4f}")
    print(f"  1着的中率  {old['win_accuracy']:.2%} → {new['win_accuracy']:.2%}  ({d_acc:+.2%})")
    print(f"  基準線     1号艇固定 {new['baseline_lane1']:.2%}")

    if new["win_accuracy"] <= new["baseline_lane1"]:
        print("\n✖ 1号艇を買い続けるより悪いモデルです。配備しません。")
        sys.exit(2)

    if d_acc < -MAX_ACCURACY_DROP:
        print(f"\n✖ 1着的中率が {abs(d_acc):.2%} 落ちています。配備しません。")
        sys.exit(2)

    if d_main < -MAX_NLL3_REGRESSION:
        print(f"\n✖ {label} が {abs(d_main):.4f} 悪化しています。配備しません。")
        sys.exit(2)

    if d_main < min_gain:
        print(f"\n― {label} は同等（誤差の範囲）です。レシピは変わらないので、"
              "データを最新化した配備だけ行います。")
        sys.exit(1)

    print("\n✔ 改善しているので新しいモデルを採用します。")
    sys.exit(0)


if __name__ == "__main__":
    main()
