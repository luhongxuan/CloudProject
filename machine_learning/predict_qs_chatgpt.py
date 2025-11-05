# -*- coding: utf-8 -*-
"""
用「同一場」賽前特徵預測該場的優質先發(QS)機率與結果，並計算成功率（若有標籤）
輸入: C:\\CloudProject\\machine_learning\\pitcher_record\\all_pitchers_2024\\All_Pitchers_2024_Consolidated.xlsx
模型(建議): .\\artifacts_qs_xgb\\qs_xgb_classifier_calibrated.joblib
輸出: 同資料夾，檔名加 _qs_pred.xlsx
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, average_precision_score

# === 路徑 ===
MODEL_PATH = Path(r'.\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib')  # 校準後模型
INPUT_XLSX = Path(r'C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx')

# 與訓練時一致的特徵（同場賽前可得）
FEATURES = [
    "rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
    "season_era","season_whip","hand","opp_team","Team","pitcher"
]

def parse_ip(ip):
    """把 '6.1'/'6.2' 轉成 6+1/3、6+2/3；其餘轉 float。"""
    try:
        if isinstance(ip, str) and "." in ip:
            a, b = ip.split(".")
            if b in {"1","2"}:
                return float(a) + int(b)/3.0
        return float(ip)
    except Exception:
        return np.nan

def main():
    # 1) 載入模型（Pipeline/CalibratedClassifierCV 皆可直接 predict_proba）
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)  # 前處理 + 分類器一體化。:contentReference[oaicite:2]{index=2}

    # 2) 讀取資料
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到輸入檔: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)

    # 3) 建立同場實際 QS（若要評估）
    if {"IP","ER"}.issubset(df.columns):
        df["IP_float"] = df["IP"].apply(parse_ip)
        df["QS_actual"] = ((df["IP_float"] >= 6.0) & (df["ER"].astype(float) <= 3)).astype(int)
    else:
        df["QS_actual"] = np.nan  # 沒有標籤也可純預測

    # 4) 確認必要欄位存在
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"輸入檔缺少必要欄位: {missing}")

    # 5) 直接用「同一場」特徵做推論（不做 shift）
    X = df[FEATURES].copy()
    qs_prob = pipe.predict_proba(X)[:, 1]  # Pipeline 會先做 transform 再呼叫最終 estimator 的 predict_proba。:contentReference[oaicite:3]{index=3}
    qs_pred = (qs_prob >= 0.5).astype(int) # 若你有最佳閾值，可改成讀 json 使用

    # 6) 輸出與（可選）評估
    out = df.copy()
    out["qs_prob_pred"] = qs_prob
    out["qs_pred"] = qs_pred

    has_label = out["QS_actual"].notna()
    if has_label.any():
        y_true = out.loc[has_label, "QS_actual"].astype(int).values
        y_score = out.loc[has_label, "qs_prob_pred"].astype(float).values
        y_pred  = out.loc[has_label, "qs_pred"].astype(int).values

        acc   = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_score)  # Brier 越小越好。:contentReference[oaicite:4]{index=4}
        if len(np.unique(y_true)) > 1:
            auc   = roc_auc_score(y_true, y_score)
            prauc = average_precision_score(y_true, y_score)
        else:
            auc = prauc = np.nan

        print("=== 同場特徵預測同場（有實際標籤的列） ===")
        print(f"Accuracy : {acc:.3f}")
        print(f"Brier    : {brier:.3f}")
        print(f"ROC AUC  : {auc:.3f}" if not np.isnan(auc) else "ROC AUC: N/A（單一類別）")
        print(f"PR AUC   : {prauc:.3f}" if not np.isnan(prauc) else "PR AUC : N/A（單一類別）")
    else:
        print("無真實 QS 標籤，僅輸出機率與預測結果。")

    # 7) 輸出 Excel（同資料夾，_qs_pred 後綴）
    out_path = INPUT_XLSX.with_name(INPUT_XLSX.stem + "_qs_pred.xlsx")
    out.to_excel(out_path, index=False)
    print(f"已輸出：{out_path}")

if __name__ == "__main__":
    main()
