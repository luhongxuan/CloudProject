# -*- coding: utf-8 -*-
"""
用「同一場」賽前特徵預測該場的優質先發(QS)機率與結果，並輸出圖表報告
輸入:  C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx
模型:  .\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib
輸出:  同資料夾，輸出 _qs_pred.xlsx 以及 *_qs_pred_reports/ 圖表
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score, brier_score_loss, roc_auc_score,
    average_precision_score, confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

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

def gauge_figure(prob: float, title: str) -> go.Figure:
    """回傳儀表圖（0-100%）"""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100.0,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': '#ff4d4d'},
                {'range': [40, 70], 'color': '#ffcc00'},
                {'range': [70, 100], 'color': '#00cc66'},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': prob * 100.0}
        }
    ))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    # 1) 載入模型
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)

    # 2) 讀取資料
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到輸入檔: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["pitcher","game_date"]).reset_index(drop=True)

    # 3) 建立同場實際 QS（若要評估）
    if {"IP","ER"}.issubset(df.columns):
        df["IP_float"] = df["IP"].apply(parse_ip)
        df["QS_actual"] = ((df["IP_float"] >= 6.0) & (df["ER"].astype(float) <= 3)).astype(int)
    else:
        df["QS_actual"] = np.nan

    # 4) 確認必要欄位存在
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"輸入檔缺少必要欄位: {missing}")

    # 5) 推論
    X = df[FEATURES].copy()
    qs_prob = pipe.predict_proba(X)[:, 1]
    qs_pred = (qs_prob >= 0.5).astype(int)

    out = df.copy()
    out["qs_prob_pred"] = qs_prob
    out["qs_pred"] = qs_pred

    # 6) 評估（若有標籤）
    metrics = {}
    has_label = out["QS_actual"].notna()
    if has_label.any():
        y_true = out.loc[has_label, "QS_actual"].astype(int).values
        y_score = out.loc[has_label, "qs_prob_pred"].astype(float).values
        y_pred  = out.loc[has_label, "qs_pred"].astype(int).values

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["brier"]    = brier_score_loss(y_true, y_score)
        metrics["auc"]      = roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
        metrics["prauc"]    = average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan

        print("=== 同場特徵預測同場（有實際標籤的列） ===")
        print(f"Accuracy : {metrics['accuracy']:.3f}")
        print(f"Brier    : {metrics['brier']:.3f}")
        print(f"ROC AUC  : {metrics['auc']:.3f}" if not np.isnan(metrics["auc"]) else "ROC AUC: N/A")
        print(f"PR AUC   : {metrics['prauc']:.3f}" if not np.isnan(metrics["prauc"]) else "PR AUC : N/A")
    else:
        print("無真實 QS 標籤，僅輸出機率與預測結果。")

    # 7) 輸出 Excel
    out_path = INPUT_XLSX.with_name(INPUT_XLSX.stem + "_qs_pred.xlsx")
    out.to_excel(out_path, index=False)
    print(f"已輸出：{out_path}")

    # 8) 視覺化輸出資料夾
    report_dir = INPUT_XLSX.with_name(INPUT_XLSX.stem + "_qs_pred_reports")
    ensure_dir(report_dir)

    # ---------- (A) 全體：機率分佈 & 隊/投手分佈 ----------
    fig_hist = px.histogram(out, x="qs_prob_pred", nbins=30, title="QS Probability Distribution (All)")
    fig_hist.update_xaxes(title="QS Probability")
    fig_hist.update_yaxes(title="Count")
    fig_hist.write_html(str(report_dir / "hist_all_qs_prob.html"))

    # 若有標籤：分布（label 分色）
    if has_label.any():
        fig_hist_lbl = px.histogram(out.loc[has_label], x="qs_prob_pred", color="QS_actual",
                                    nbins=30, barmode="overlay",
                                    title="QS Probability by Actual Label (0/1)")
        fig_hist_lbl.update_xaxes(title="QS Probability")
        fig_hist_lbl.update_yaxes(title="Count")
        fig_hist_lbl.write_html(str(report_dir / "hist_qs_prob_by_label.html"))

    # ---------- (B) 每位投手：機率趨勢線 ----------
    for name, g in out.groupby("pitcher"):
        g = g.sort_values("game_date")
        title = f"{name} — QS Probability Over Time"
        fig_line = px.line(g, x="game_date", y="qs_prob_pred", title=title, markers=True)
        fig_line.update_yaxes(range=[0,1], title="QS Probability")
        fig_line.update_xaxes(title="Game Date")
        fig_line.write_html(str(report_dir / f"{name.replace(' ','_')}_prob_trend.html"))

        # 最新一場儀表圖
        latest = g.iloc[-1]
        gauge = gauge_figure(latest["qs_prob_pred"], f"{name} Latest Game QS Probability")
        gauge.write_html(str(report_dir / f"{name.replace(' ','_')}_latest_gauge.html"))

    # ---------- (C) ROC / PR / 校準 / 混淆矩陣 ----------
    if has_label.any() and len(np.unique(out.loc[has_label,"QS_actual"]))>1:
        y_true = out.loc[has_label, "QS_actual"].astype(int).values
        y_score = out.loc[has_label, "qs_prob_pred"].astype(float).values
        y_pred  = out.loc[has_label, "qs_pred"].astype(int).values

        # ROC
        plt.figure(figsize=(5,4))
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.title(f"ROC Curve (AUC={metrics['auc']:.3f})")
        plt.tight_layout()
        plt.savefig(report_dir / "roc_curve.png", dpi=160)
        plt.close()

        # PR
        plt.figure(figsize=(5,4))
        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.title(f"Precision-Recall (AP={metrics['prauc']:.3f})")
        plt.tight_layout()
        plt.savefig(report_dir / "pr_curve.png", dpi=160)
        plt.close()

        # Calibration (Reliability)
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
        plt.figure(figsize=(5,4))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0,1],[0,1], "--", color="gray", label="Perfectly Calibrated")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration (Reliability) Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / "calibration_curve.png", dpi=160)
        plt.close()

        # Confusion Matrix (閾值=0.5)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=["0","1"], y=["0","1"], title="Confusion Matrix @ 0.5")
        fig_cm.write_html(str(report_dir / "confusion_matrix.html"))

    # ---------- (D) 簡單索引頁（HTML） ----------
    index_lines = [
        "<h2>QS Prediction Report</h2>",
        f"<p><b>Input:</b> {INPUT_XLSX.name}<br><b>Model:</b> {MODEL_PATH.name}</p>",
        "<h3>Global Charts</h3>",
        '<ul>',
        '<li><a href="hist_all_qs_prob.html">Probability Histogram (All)</a></li>',
    ]
    if has_label.any():
        index_lines.append('<li><a href="hist_qs_prob_by_label.html">Probability Histogram by Label</a></li>')
        index_lines.append('<li><a href="roc_curve.png">ROC Curve (PNG)</a></li>')
        index_lines.append('<li><a href="pr_curve.png">PR Curve (PNG)</a></li>')
        index_lines.append('<li><a href="calibration_curve.png">Calibration Curve (PNG)</a></li>')
        index_lines.append('<li><a href="confusion_matrix.html">Confusion Matrix</a></li>')
    index_lines.append('</ul>')
    index_lines.append("<h3>Per-Pitcher</h3><ul>")

    for name in sorted(out["pitcher"].dropna().unique()):
        base = name.replace(" ","_")
        line = f'<li>{name}: ' \
               f'<a href="{base}_prob_trend.html">Trend</a> | ' \
               f'<a href="{base}_latest_gauge.html">Gauge(latest)</a></li>'
        index_lines.append(line)
    index_lines.append("</ul>")

    (report_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"📊 圖表已輸出到：{report_dir}\n   打開 index.html 可瀏覽全部圖表。")

if __name__ == "__main__":
    main()
