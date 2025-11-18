# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import shap

# === 新增：解釋單一預測 ===
def explain_prediction(raw_pipe,
                       df_all: pd.DataFrame,
                       row_s: pd.Series,
                       prob: float,
                       thresh: float,
                       topk: int = 8,
                       out_dir: Path | None = None):
    """
    利用 SHAP 解釋：為什麼這一場的 QS 機率是 prob？
    並把前 topk 個重要特徵畫成圖（若 out_dir 不為 None）。
    """

    if raw_pipe is None:
        print("\n[警告] 找不到 qs_xgb_pipeline_raw.joblib，無法做 SHAP 解釋。")
        print("       請確認訓練腳本有 joblib.dump(pipe_base, 'qs_xgb_pipeline_raw.joblib')。")
        return

    prep = raw_pipe.named_steps["prep"]
    xgb  = raw_pipe.named_steps["model"]

    # 1) 背景資料：從全部資料中抽樣，避免太慢
    bg = df_all[FEATURES].copy()
    if len(bg) > 500:
        bg = bg.sample(500, random_state=0)

    X_bg_enc = prep.transform(bg)

    # 2) 這一列 → DataFrame → 同樣做前處理
    X_one_df  = pd.DataFrame([row_s[FEATURES].to_dict()])
    X_one_enc = prep.transform(X_one_df)

    # 3) 建立 SHAP explainer 對 XGB 做樹模型解釋
    explainer = shap.TreeExplainer(xgb)

    # 這裡要注意：不同版本的 shap 回傳格式不一樣：
    # - 有些會回傳 ndarray: (n_samples, n_features)
    # - 有些會回傳 list: [class0_shap, class1_shap, ...]
    raw_shap = explainer.shap_values(X_one_enc)

    if isinstance(raw_shap, list):
        # 二元分類：取「正類別」的 SHAP 值；raw_shap[1].shape = (n_samples, n_features)
        vals = raw_shap[1][0]
    else:
        # 單一矩陣：直接取第 1 筆樣本
        vals = raw_shap[0]

    # 4) 編碼後特徵名稱，例如 num__rest_days / cat__opp_team_Tigers
    encoded_names = prep.get_feature_names_out()

    # 5) 合併回原本欄位（數值欄直接對應；類別欄把 one-hot 加總）
    contrib = {f: 0.0 for f in FEATURES}

    for name, v in zip(encoded_names, vals):
        if "__" in name:
            _, rest = name.split("__", 1)
        else:
            rest = name

        if rest in NUMERIC:
            contrib[rest] += float(v)
            continue

        for cat in CATEG:
            prefix = cat + "_"
            if rest.startswith(prefix):
                contrib[cat] += float(v)
                break

    # 6) 排序、組成 DataFrame
    rows = []
    for f in FEATURES:
        rows.append({
            "feature": f,
            "value": row_s[f],
            "contrib": contrib[f],
            "abs_contrib": abs(contrib[f]),
        })
    imp = pd.DataFrame(rows).sort_values("abs_contrib", ascending=False)

    # ========= 終端機文字解釋 =========
    print("\n=== 單場預測解釋 (SHAP, 對 log-odds 的貢獻) ===")
    print(f"QS 機率 = {prob*100:.1f}%   (門檻 = {thresh*100:.1f}%)")

    pos = imp[imp["contrib"] > 0].sort_values("contrib", ascending=False).head(topk)
    neg = imp[imp["contrib"] < 0].sort_values("contrib", ascending=True).head(topk)

    if prob >= thresh:
        print("\n模型判定：『這場比較有機會 QS』，主要拉高機率的特徵：")
        for _, r in pos.iterrows():
            print(f"  + {r['feature']} = {r['value']}  → 貢獻 +{r['contrib']:.4f}")
        if not neg.empty:
            print("\n同時，有一些特徵在拉低機率：")
            for _, r in neg.iterrows():
                print(f"  - {r['feature']} = {r['value']}  → 貢獻 {r['contrib']:.4f}")
    else:
        print("\n模型判定：『這場不太容易 QS』，主要拉低機率的特徵：")
        for _, r in neg.iterrows():
            print(f"  - {r['feature']} = {r['value']}  → 貢獻 {r['contrib']:.4f}")
        if not pos.empty:
            print("\n但也有一些特徵在拉高機率：")
            for _, r in pos.iterrows():
                print(f"  + {r['feature']} = {r['value']}  → 貢獻 +{r['contrib']:.4f}")

    print("\n(註：貢獻是對『log-odds』的影響，但正負號可以當作「拉高 / 拉低機率」來解讀)")

    # ========= 畫圖輸出（條狀圖） =========
    if out_dir is not None:
        out_dir = Path(out_dir)

        # 取前 topk 個影響最大的特徵，倒序讓貢獻最大的在上面
        top_plot = imp.head(topk).iloc[::-1]  # 反轉順序，配合 barh 從上到下

        fig, ax = plt.subplots(figsize=(8, 5))

        y_labels = [
            f"{f} = {v}"
            for f, v in zip(top_plot["feature"].astype(str),
                            top_plot["value"].astype(str))
        ]

        ax.barh(y_labels, top_plot["contrib"])
        ax.axvline(0, linewidth=1)  # 0 的參考線

        ax.set_xlabel("SHAP contribution (>0 increases QS probability, <0 decreases)")
        title_main = f"{row_s.get('pitcher','?')} – single-game feature contributions"
        title_sub  = f"QS prob = {prob*100:.1f}% | threshold = {thresh*100:.1f}%"
        ax.set_title(title_main + "\n" + title_sub, fontsize=11)

        plt.tight_layout()

        # 檔名：QS_contrib_投手_日期.png
        pitcher_name = str(row_s.get('pitcher', 'NA')).replace(" ", "_")
        game_date    = str(row_s.get('game_date', '')).split(" ")[0]
        fname = f"QS_contrib_{pitcher_name}_{game_date}.png"
        out_path = out_dir / fname

        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        print(f"\n特徵貢獻圖已輸出：{out_path}")


# ========== 路徑 ==========
MODEL_PATH = Path(r'C:\CloudProject\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib')  # 校準後模型
THRESH_PATH = Path(r'C:\CloudProject\artifacts_qs_xgb\qs_best_threshold.json')              # 閾值
RAW_PIPE_PATH = Path(r'C:\CloudProject\artifacts_qs_xgb\qs_xgb_pipeline_raw.joblib')       # <<< 新：原始 XGB pipeline
INPUT_XLSX = Path(r'C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx')

# === 新：跟訓練程式保持一致 ===
NUMERIC = ["rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3","season_era","season_whip"]
CATEG   = ["hand","opp_team","Team","pitcher"]
FEATURES = NUMERIC + CATEG

# 訓練時的一致特徵（同場賽前可得）
FEATURES = [
    "rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
    "season_era","season_whip","hand","opp_team","Team","pitcher"
]

def load_model_and_threshold():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
    pipe_calibrated = joblib.load(MODEL_PATH)  # CalibratedClassifierCV

    # 嘗試載入原始 XGB pipeline（可能不存在，就給 None）
    raw_pipe = joblib.load(RAW_PIPE_PATH) if RAW_PIPE_PATH.exists() else None

    # 閾值（預設 0.5，但如果有 JSON 就用 JSON）
    best_t = 0.5
    # if THRESH_PATH.exists():
    #     try:
    #         with open(THRESH_PATH, "r", encoding="utf-8") as f:
    #             obj = json.load(f)
    #         if "best_threshold" in obj:
    #             best_t = float(obj["best_threshold"])
    #     except Exception:
    #         pass
    return pipe_calibrated, raw_pipe, best_t

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

    # 0) 載模型 + 閾值
    pipe_calibrated, raw_pipe, best_t = load_model_and_threshold()

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
    prob = float(pipe_calibrated.predict_proba(X_one)[0, 1])  # 使用校準後模型算機率
    print(prob)

    pred = int(prob >= best_t)  # 若需 0/1 判定
    print(f"Pitcher={row_s.get('pitcher','?')}, Date={row_s.get('game_date','?')}, "
          f"QS_Prob={prob:.4f}, Pred(at t={best_t:.2f})={pred}")
    
    # === 新增：解釋為什麼是這個機率（>50% / <50%）===
    out_dir = INPUT_XLSX.parent

    explain_prediction(
        raw_pipe=raw_pipe,
        df_all=df,
        row_s=row_s,
        prob=prob,
        thresh=best_t,
        topk=8,
        out_dir=out_dir,
    )

    # 4) 輸出儀表圖
    title = f"{row_s.get('pitcher','?')} QS Probability"
    out_png = (args.output if args.output is not None
               else INPUT_XLSX.with_name(f"QS_gauge_{str(row_s.get('pitcher','NA')).replace(' ','_')}"
                                         f"_{str(row_s.get('game_date','')).split(' ')[0]}.png"))
    draw_gauge(prob, title, out_png)
    print(f"儀表圖已輸出：{out_png}")

if __name__ == "__main__":
    main()
