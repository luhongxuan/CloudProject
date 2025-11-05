# -*- coding: utf-8 -*-
"""
QS (Quality Start) 強化版訓練腳本
改進點：
1) 時間序交叉驗證 TimeSeriesSplit + 報告 AUC/PR-AUC/Brier/Acc
2) 機率校準 CalibratedClassifierCV（isotonic 或 sigmoid）
3) 自動處理不平衡：scale_pos_weight = neg / pos
4) 在訓練集末段切一個 validation fold 搜尋最佳決策閾值（非固定 0.5）
5) 產出：校準後模型 + 最佳閾值 + 評估報表 + ROC 圖

參考：
- TimeSeriesSplit（避免用未來資料評估過去）: scikit-learn docs
- 機率校準/ Brier score：scikit-learn docs
- XGBoost scale_pos_weight：常用為 neg/pos
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, classification_report, RocCurveDisplay,
    precision_recall_curve, mean_squared_error, mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor

# ====== 路徑 ======
DATA_PATH = Path(r'C:\CloudProject\machine_learning\pitcher_record\All_Pitchers.xlsx')
ART_DIR = Path("./artifacts_qs_xgb")
ART_DIR.mkdir(parents=True, exist_ok=True)

# ====== 小工具 ======
def parse_ip(ip) -> float:
    try:
        if isinstance(ip, str) and "." in ip:
            a, b = ip.split(".")
            if b in {"1", "2"}:
                return float(a) + int(b)/3.0
        return float(ip)
    except Exception:
        return np.nan

def rmse_of(y_true, y_pred):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    if n < 2:
        raise ValueError(f"資料量過小({n})")
    cut = max(1, min(int(n * (1 - test_ratio)), n - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# ====== 讀檔與標籤 ======
if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到檔案：{DATA_PATH}")
df_raw = pd.read_excel(DATA_PATH)
df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
df_raw["IP_float"] = df_raw["IP"].apply(parse_ip)
df_raw["QS"] = ((df_raw["IP_float"] >= 6.0) & (df_raw["ER"].astype(float) <= 3)).astype(int)

# 只用賽前可得特徵（避免洩漏）
NUMERIC = ["rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3","season_era","season_whip"]
CATEG   = ["hand","opp_team","Team","pitcher"]
FEATURES = NUMERIC + CATEG

need_cols = list(set(FEATURES + ["game_date","IP_float","ER","QS"]))
miss = [c for c in need_cols if c not in df_raw.columns]
if miss:
    raise KeyError(f"缺少必要欄位: {miss}")

df = df_raw[need_cols].sort_values("game_date").reset_index(drop=True)

# ====== 時間序切分（最後 20% 當最終測試）======
train_df, test_df = time_split(df, test_ratio=0.2)
X_tr, y_tr = train_df[FEATURES], train_df["QS"].astype(int)
X_te, y_te = test_df[FEATURES],  test_df["QS"].astype(int)

print(f"Train size={len(X_tr)}, Test size={len(X_te)}")

# ====== 前處理 Pipeline ======
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="UNK", keep_empty_features=True)),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
prep = ColumnTransformer([("num", num_pipe, NUMERIC),
                          ("cat", cat_pipe, CATEG)])

# ====== 不平衡處理：自動計算 scale_pos_weight ======
pos = int(y_tr.sum())
neg = int(len(y_tr) - pos)
scale_pos_weight = float(neg / max(1, pos))  # 典型做法：neg/pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

base_xgb = XGBClassifier(
    n_estimators=1000, max_depth=10, learning_rate=0.01,
    subsample=1, colsample_bytree=1,
    reg_lambda=1.0, min_child_weight=2.0,
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42, tree_method="hist",
)

pipe_base = Pipeline([("prep", prep), ("model", base_xgb)])

# ====== 時間序交叉驗證（僅用訓練資料做 CV）======
tscv = TimeSeriesSplit(n_splits=5)
cv = cross_validate(
    pipe_base, X_tr, y_tr, cv=tscv, n_jobs=-1,
    scoring=["roc_auc","average_precision","neg_brier_score","accuracy"],
    error_score="raise"
)
print("CV scores (mean ± std):")
for k in ["test_roc_auc","test_average_precision","test_neg_brier_score","test_accuracy"]:
    print(f" - {k}: {np.mean(cv[k]):.3f} ± {np.std(cv[k]):.3f}")

# ====== 機率校準（isotonic；資料少可改 sigmoid）======
try:
    calibrated = CalibratedClassifierCV(estimator=pipe_base, method="isotonic", cv=tscv)
except TypeError:  # 舊版 fallback
    calibrated = CalibratedClassifierCV(base_estimator=pipe_base, method="isotonic", cv=tscv)
calibrated.fit(X_tr, y_tr)

# ====== 用訓練集末段做「閾值搜尋」而非固定 0.5 ======
# 取最後一折當 validation（TimeSeriesSplit 的最後一折即最接近未來）
val_split = list(tscv.split(X_tr, y_tr))[-1]
tr_idx, val_idx = val_split
X_val, y_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
proba_val = calibrated.predict_proba(X_val)[:,1]
prec, rec, th = precision_recall_curve(y_val, proba_val)
f1 = 2*prec*rec/(prec+rec+1e-12)
best_threshold = float(th[np.argmax(f1)]) if len(th)>0 else 0.5
print(f"Best threshold on validation (F1-opt): {best_threshold:.3f}")

# ====== 最終在測試集上評估 ======
proba_te = calibrated.predict_proba(X_te)[:,1]
pred_te  = (proba_te >= best_threshold).astype(int)

def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan
def safe_ap(y_true, y_score):
    return average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else np.nan

roc  = safe_auc(y_te, proba_te)
pra  = safe_ap(y_te, proba_te)
brier= brier_score_loss(y_te, proba_te)
acc  = accuracy_score(y_te, pred_te)

print("\n=== QS Classification (Test) ===")
print(f"ROC AUC={roc:.3f}  PR AUC={pra:.3f}  Brier={brier:.3f}  Acc@{best_threshold:.2f}={acc:.3f}")
print(classification_report(y_te, pred_te, digits=3))

if len(np.unique(y_te))>1:
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_te, proba_te)
    plt.title("QS ROC Curve (Calibrated, Test)")
    plt.tight_layout()
    plt.savefig(ART_DIR / "qs_roc_calibrated.png", dpi=140)

# ====== 回歸（ER/IP）維持原設計 ======
def fit_regression(target_col: str, name: str) -> dict:
    reg = XGBRegressor(
        n_estimators=1000, max_depth=10, learning_rate=0.01,
        subsample=1, colsample_bytree=0.1,
        reg_lambda=1.0, min_child_weight=2.0,
        objective="reg:squarederror", random_state=42, tree_method="hist"
    )
    pipe_reg = Pipeline([("prep", prep), ("model", reg)])
    pipe_reg.fit(X_tr, train_df[target_col].astype(float))
    pred = pipe_reg.predict(X_te)
    rmse = rmse_of(test_df[target_col].astype(float), pred)
    mae  = mean_absolute_error(test_df[target_col].astype(float), pred)
    joblib.dump(pipe_reg, ART_DIR / f"{name}_xgb_regressor.joblib")
    print(f"=== {target_col} Regression ===  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return {"rmse": rmse, "mae": mae}

er_res = fit_regression("ER", "er")
ip_res = fit_regression("IP_float", "ip")

# ====== 輸出：校準模型 + 閾值 + 報表 ======
model_path = ART_DIR / "qs_xgb_classifier_calibrated.joblib"
joblib.dump(calibrated, model_path)

with open(ART_DIR / "qs_best_threshold.json", "w", encoding="utf-8") as f:
    json.dump({"best_threshold": best_threshold}, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "metric": ["roc_auc","pr_auc","brier","accuracy_at_best_t","best_threshold",
               "er_rmse","er_mae","ip_rmse","ip_mae"],
    "value": [float(roc) if not np.isnan(roc) else np.nan,
              float(pra) if not np.isnan(pra) else np.nan,
              float(brier), float(acc), float(best_threshold),
              float(er_res["rmse"]), float(er_res["mae"]),
              float(ip_res["rmse"]), float(ip_res["mae"])],
}).to_csv(ART_DIR / "eval_report_calibrated.csv", index=False)

print("\nArtifacts saved to:", ART_DIR.resolve())
print(" - qs_xgb_classifier_calibrated.joblib")
print(" - qs_best_threshold.json")
print(" - er_xgb_regressor.joblib")
print(" - ip_xgb_regressor.joblib")
print(" - eval_report_calibrated.csv")
print(" - qs_roc_calibrated.png")