# -*- coding: utf-8 -*-
from pathlib import Path
import json              # === 新增：讀取 json 用 ===
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, average_precision_score

# === 路徑 ===
MODEL_PATH = Path(r'C:\CloudProject\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib')  # 校準後模型
THRESH_PATH = Path(r'C:\CloudProject\artifacts_qs_xgb\qs_best_threshold.json')              # === 新增：最佳閥值 json ===
INPUT_XLSX = Path(r'C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2025\All_Pitchers_2025_Consolidated.xlsx')

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

def load_best_threshold(default: float = 0.5) -> float:
    """從 qs_best_threshold.json 讀取 best_threshold；失敗就用預設值。"""
    t = default
    if THRESH_PATH.exists():
        try:
            with open(THRESH_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "best_threshold" in obj:
                t = float(obj["best_threshold"])
                print(f"使用 json 閥值 best_threshold = {t:.3f}")
            else:
                print(f"[警告] {THRESH_PATH} 中沒有 best_threshold 欄位，改用預設 {default:.2f}")
        except Exception as e:
            print(f"[警告] 讀取 {THRESH_PATH} 失敗：{e}，改用預設 {default:.2f}")
    else:
        print(f"[提示] 找不到 {THRESH_PATH}，改用預設閥值 {default:.2f}")
    return t

def main():
    # 1) 載入模型
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)

    # === 新增：載入最佳閥值 ===
    # best_t = load_best_threshold(default=0.5)
    best_t = 0.5
    
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

    # 5) 用「同一場」特徵做推論
    X = df[FEATURES].copy()
    qs_prob = pipe.predict_proba(X)[:, 1]
    qs_pred = (qs_prob >= best_t).astype(int)   # <<< 改成用 json 閥值

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
        brier = brier_score_loss(y_true, y_score)
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

    # 7) 輸出 Excel
    out_path = INPUT_XLSX.with_name(INPUT_XLSX.stem + "_qs_pred.xlsx")
    out.to_excel(out_path, index=False)
    print(f"已輸出：{out_path}")

if __name__ == "__main__":
    main()
