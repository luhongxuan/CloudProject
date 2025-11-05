# -*- coding: utf-8 -*-
"""
Robust XGBoost for QS (classification) + ER/IP (regression)
- Excel: r'C:\\CloudProject\\machine_learning\\pitcher_record\\All_Pitchers.xlsx'
- Fix 1: RMSE 相容性：優先使用 root_mean_squared_error；否則以 sqrt(MSE)
- Fix 2: 類別欄缺值以常數 'UNK' 填補，避免整欄 NaN 造成 Imputer 警告
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, classification_report, RocCurveDisplay,
    mean_squared_error, mean_absolute_error
)
# 盡量相容的 RMSE 取得函數（新舊版都可用）
def rmse_of(y_true, y_pred):
    try:
        from sklearn.metrics import root_mean_squared_error  # >=1.4
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        # 舊版：沒有 root_mean_squared_error；某些版本的 MSE 也不接受 squared 參數
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from xgboost import XGBClassifier, XGBRegressor

# ===== 路徑 =====
DATA_PATH = Path(r'C:\CloudProject\machine_learning\pitcher_record\All_Pitchers.xlsx')
ART_DIR = Path("./artifacts_qs_xgb")
ART_DIR.mkdir(parents=True, exist_ok=True)

# ===== 小工具 =====
def parse_ip(ip) -> float:
    """把 '6.1' / '6.2' 轉 6 + 1/3 或 6 + 2/3；其他數值直接轉 float。"""
    try:
        if isinstance(ip, str) and "." in ip:
            a, b = ip.split(".")
            if b in {"1", "2"}:
                return float(a) + int(b)/3.0
        return float(ip)
    except Exception:
        return np.nan

def safe_time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """時間序切分，且保證 train/test 都 >=1 筆。"""
    n = len(df)
    if n < 2:
        raise ValueError(f"資料量過小（{n}）。至少需要 2 筆。")
    cut = int(n * (1 - test_ratio))
    cut = max(1, min(cut, n - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# ===== 讀資料 =====
if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到檔案：{DATA_PATH}")
df_raw = pd.read_excel(DATA_PATH)
print(f"Loaded Excel: {DATA_PATH}")

# ===== 基本處理與標籤 =====
df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
df_raw["IP_float"] = df_raw["IP"].apply(parse_ip)
# QS：>=6 局 且 ER <= 3
df_raw["QS"] = ((df_raw["IP_float"] >= 6.0) & (df_raw["ER"].astype(float) <= 3)).astype(int)

# 只用賽前可得特徵（避免洩漏）
NUMERIC = ["rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3","season_era","season_whip"]
CATEG   = ["hand","opp_team","Team","pitcher"]  # 類別欄
FEATURES = NUMERIC + CATEG

need_cols = list(set(FEATURES + ["game_date","IP_float","ER","QS"]))
missing = [c for c in need_cols if c not in df_raw.columns]
if missing:
    raise KeyError(f"缺少必要欄位: {missing}")

df = df_raw[need_cols].sort_values("game_date").reset_index(drop=True)

# ===== 時間序切分 =====
train, test = safe_time_split(df, test_ratio=0.2)
X_tr, y_tr = train[FEATURES], train["QS"].astype(int)
X_te, y_te = test[FEATURES],  test["QS"].astype(int)

print(f"Train size = {len(X_tr)}, Test size = {len(X_te)}")

# ===== 前處理：數值中位數、類別常數 'UNK' + OneHot(忽略未知類別) =====
num_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median"))
])
cat_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # 測試集中未見新類別時不報錯
])

prep = ColumnTransformer([
    ("num", num_pipe, NUMERIC),
    ("cat", cat_pipe, CATEG),
])

# ===== 分類：預測 QS =====
clf = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.99, colsample_bytree=0.99, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, tree_method="hist",
)
pipe_cls = Pipeline([("prep", prep), ("model", clf)])
pipe_cls.fit(X_tr, y_tr)

proba_te = pipe_cls.predict_proba(X_te)[:, 1]
pred_te  = (proba_te >= 0.5).astype(int)

def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
def safe_ap(y_true, y_score):
    return average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

roc  = safe_auc(y_te, proba_te)
ap   = safe_ap(y_te, proba_te)
brier = brier_score_loss(y_te, proba_te)
acc   = accuracy_score(y_te, pred_te)

print("\n=== QS Classification ===")
print(f"ROC AUC={roc:.3f}  PR AUC={ap:.3f}  Brier={brier:.3f}  Acc={acc:.3f}")
print(classification_report(y_te, pred_te, digits=3))

if len(np.unique(y_te)) > 1:
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_te, proba_te)
    plt.title("QS ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig(ART_DIR / "qs_roc.png", dpi=140)

# ===== 回歸：ER / IP =====
from sklearn.metrics import mean_absolute_error

def fit_regression(target_col: str, name: str) -> dict:
    reg = XGBRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.99, colsample_bytree=0.99, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, tree_method="hist",
    )
    pipe_reg = Pipeline([("prep", prep), ("model", reg)])
    pipe_reg.fit(X_tr, train[target_col].astype(float))
    pred = pipe_reg.predict(X_te)
    # 相容計算 RMSE/MAE
    rmse = rmse_of(test[target_col].astype(float), pred)
    mae  = mean_absolute_error(test[target_col].astype(float), pred)
    joblib.dump(pipe_reg, ART_DIR / f"{name}_xgb_regressor.joblib")
    print(f"=== {target_col} Regression ===  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return {"rmse": rmse, "mae": mae}

er_res = fit_regression("ER", "er")
ip_res = fit_regression("IP_float", "ip")

# ===== 輸出模型與報表 =====
joblib.dump(pipe_cls, ART_DIR / "qs_xgb_classifier.joblib")
pd.DataFrame({
    "metric": ["roc_auc","pr_auc","brier","accuracy","er_rmse","er_mae","ip_rmse","ip_mae"],
    "value": [float(roc) if not np.isnan(roc) else np.nan,
              float(ap)  if not np.isnan(ap)  else np.nan,
              float(brier), float(acc),
              float(er_res["rmse"]), float(er_res["mae"]),
              float(ip_res["rmse"]), float(ip_res["mae"])],
}).to_csv(ART_DIR / "eval_report.csv", index=False)

print("\nArtifacts saved to:", ART_DIR.resolve())
print(" - qs_xgb_classifier.joblib")
print(" - er_xgb_regressor.joblib")
print(" - ip_xgb_regressor.joblib")
print(" - eval_report.csv")
