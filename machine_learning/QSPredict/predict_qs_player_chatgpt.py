# -*- coding: utf-8 -*-
"""
從 Excel 抽取「某一列」投手資料，預測該場 QS 機率並輸出儀表圖 (PNG)
- 支援以 --row (0-based) 指定；或以 --pitcher + [--date YYYY-MM-DD] 指定
- 讀取校準後模型 (建議)；若無可改路徑
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# ========== 路徑 ==========
MODEL_PATH = Path(r'.\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib')  # 建議：校準後模型
THRESH_PATH = Path(r'.\artifacts_qs_xgb\qs_best_threshold.json')  # 可選：若你也想輸出 0/1 判定
INPUT_XLSX = Path(r'C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx')

# 訓練時的一致特徵（同場賽前可得）
FEATURES = [
    "rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
    "season_era","season_whip","hand","opp_team","Team","pitcher"
]

def load_model_and_threshold():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)  # Pipeline / CalibratedClassifierCV 皆可直接 predict_proba
    # 嘗試讀最佳閾值（可選）
    best_t = 0.5
    if THRESH_PATH.exists():
        try:
            with open(THRESH_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "best_threshold" in obj:
                best_t = float(obj["best_threshold"])
        except Exception:
            pass
    return pipe, best_t

def pick_row(df: pd.DataFrame, row: int|None, pitcher: str|None, date: str|None) -> pd.Series:
    if row is not None:
        if row < 0 or row >= len(df):
            raise IndexError(f"--row 超出範圍 (0 ~ {len(df)-1})")
        return df.iloc[row]

    # 以 pitcher(+date) 篩選
    mask = df["pitcher"].astype(str).str.lower() == str(pitcher).strip().lower()
    sub = df[mask]
    if date is not None and "game_date" in df.columns:
        dt = pd.to_datetime(date)
        sub = sub[pd.to_datetime(sub["game_date"]) == dt]
    if sub.empty:
        raise ValueError("找不到符合的列，請檢查 pitcher 名稱或日期。")
    if len(sub) > 1:
        # 若多列，取最接近該日期或最後一列
        sub = sub.sort_values("game_date") if "game_date" in sub.columns else sub
        print(f"[提示] 找到 {len(sub)} 列，預設取其中最後一列。")
        return sub.iloc[-1]
    return sub.iloc[0]

def draw_gauge(prob: float, title: str, out_png: Path):
    """
    以半圓儀表顯示機率（0~1）。用 Wedge 畫半圓、線段畫指針。
    """
    prob = float(np.clip(prob, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'aspect': 'equal'})
    ax.axis('off')

    # 半圓底盤 (外圈)
    outer = Wedge(center=(0, 0), r=1.0, theta1=180, theta2=0, width=0.28)
    ax.add_patch(outer)

    # 刻度區段 (0-0.33 / 0.33-0.66 / 0.66-1.0) 可自行調整
    ax.add_patch(Wedge((0, 0), 1.0, 180, 120, width=0.28, alpha=0.25))
    ax.add_patch(Wedge((0, 0), 1.0, 120, 60,  width=0.28, alpha=0.25))
    ax.add_patch(Wedge((0, 0), 1.0, 60,  0,   width=0.28, alpha=0.25))

    # 指針角度：0% 在 180 度（最左），100% 在 0 度（最右）
    theta = math.radians(180 * (1.0 - prob))
    r = 0.85
    x, y = r * math.cos(theta), r * math.sin(theta)
    ax.plot([0, x], [0, y], linewidth=3)
    ax.add_patch(Circle((0, 0), 0.035))

    # 百分比文字
    ax.text(0, -0.20, f"{prob*100:.1f}%", ha="center", va="center", fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.3, 1.1)
    ax.set_title(title, fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=None, help="以 0-based 列索引選取資料")
    parser.add_argument("--pitcher", type=str, default=None, help="以投手名選取資料（不分大小寫）")
    parser.add_argument("--date", type=str, default=None, help="限定日期 YYYY-MM-DD（搭配 --pitcher）")
    parser.add_argument("--output", type=Path, default=None, help="儀表圖輸出路徑 (PNG)")
    args = parser.parse_args()

    # 0) 載模型 + 閾值（閾值可用於是否 QS 的 0/1 判定；儀表圖只用機率）
    pipe, best_t = load_model_and_threshold()

    # 1) 讀 Excel
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到輸入檔: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    # 2) 檢查特徵欄
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"輸入檔缺少必要欄位: {missing}")

    # 3) 挑出要預測的「一列」
    row_s = pick_row(df, row=args.row, pitcher=args.pitcher, date=args.date)

    X_one = pd.DataFrame([row_s[FEATURES].to_dict()])  # 保留欄位名稱，丟進 Pipeline
    prob = float(pipe.predict_proba(X_one)[0, 1])      # Pipeline/CalibratedClassifierCV 皆支援 predict_proba

    pred = int(prob >= best_t)  # 若需 0/1 判定
    print(f"Pitcher={row_s.get('pitcher','?')}, Date={row_s.get('game_date','?')}, "
          f"QS_Prob={prob:.4f}, Pred(at t={best_t:.2f})={pred}")

    # 4) 輸出儀表圖
    title = f"{row_s.get('pitcher','?')} QS Probability"
    out_png = (args.output if args.output is not None
               else INPUT_XLSX.with_name(f"QS_gauge_{str(row_s.get('pitcher','NA')).replace(' ','_')}"
                                         f"_{str(row_s.get('game_date','')).split(' ')[0]}.png"))
    draw_gauge(prob, title, out_png)
    print(f"儀表圖已輸出：{out_png}")

if __name__ == "__main__":
    main()
