import matplotlib
matplotlib.use('Agg') # 必須在 import matplotlib.pyplot 之前設定
import pandas as pd
import pandas as pd
import numpy as np
import joblib
import os
import json
import math
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# =======================================================================
# I. 全域設定與模型載入 (服務啟動時執行一次)
# =======================================================================

app = FastAPI(
    title="MLB 投手預測 API",
    description="包含即時推論 (/predict) 與歷史儀表板 (/get_historical_gauge) 的 API"
)

# --- 1. 載入「即時推論」模型 (用於 /predict) ---
try:
    qs_model_realtime = joblib.load('qs_model.joblib')
    er_model_realtime = joblib.load('er_model.joblib')
    training_columns_realtime = joblib.load('training_columns.pkl')
    print("✅ [Use Case 1] 即時推論模型 (qs, er) 載入成功！")
except FileNotFoundError:
    print("⚠️ [Use Case 1] 找不到即時推論模型 (qs_model.joblib / er_model.joblib)。/predict 端點將無法運作。")
    qs_model_realtime, er_model_realtime, training_columns_realtime = None, None, None

# --- 2. 載入「歷史儀表板」模型 (用於 /get_historical_gauge) ---
# 這些是您新腳本中定義的路徑
MODEL_PATH_GAUGE = Path(r'.\artifacts_qs_xgb\qs_xgb_classifier_calibrated.joblib')
THRESH_PATH_GAUGE = Path(r'.\artifacts_qs_xgb\qs_best_threshold.json')

def load_gauge_model_and_threshold():
    if not MODEL_PATH_GAUGE.exists():
        print(f"⚠️ [Use Case 2] 找不到儀表板模型: {MODEL_PATH_GAUGE}。/get_historical_gauge 將無法運作。")
        return None, 0.5
    pipe = joblib.load(MODEL_PATH_GAUGE)
    best_t = 0.5
    if THRESH_PATH_GAUGE.exists():
        try:
            with open(THRESH_PATH_GAUGE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "best_threshold" in obj:
                best_t = float(obj["best_threshold"])
        except Exception:
            pass
    print("✅ [Use Case 2] 儀表板模型 (calibrated_qs) 載入成功！")
    return pipe, best_t

qs_model_gauge, best_threshold_gauge = load_gauge_model_and_threshold()

# --- 3. 載入「歷史資料」 (用於 /get_historical_gauge) ---
INPUT_XLSX = Path(r'C:\CloudProject\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx')
FEATURES_GAUGE = [
    "rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
    "season_era","season_whip","hand","opp_team","Team","pitcher"
]
df_historical = None

try:
    df_historical = pd.read_excel(INPUT_XLSX)
    if "game_date" in df_historical.columns:
        df_historical["game_date"] = pd.to_datetime(df_historical["game_date"])
    print(f"✅ [Use Case 2] 成功載入 {len(df_historical)} 筆歷史資料！")
except FileNotFoundError:
    print(f"⚠️ [Use Case 2] 找不到歷史資料 Excel: {INPUT_XLSX}。/get_historical_gauge 將無法運作。")
except Exception as e:
    print(f"⚠️ [Use Case 2] 載入歷史資料 Excel 時發生錯誤: {e}")


# =======================================================================
# II. 即時推論 API (Use Case 1: 給 Node.js 呼叫)
# =======================================================================

class PitcherFeatures(BaseModel):
    pitcher: str
    opp_team: str
    Team: str
    hand: str
    rest_days: int
    opp_ops: float
    is_home: int
    avg_ip_last3: float
    avg_er_last3: float
    season_era: float
    season_whip: float

def process_realtime_features(features: PitcherFeatures, expected_columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([features.dict()])
    df_encoded = pd.get_dummies(df, columns=['pitcher', 'opp_team', 'Team', 'hand'], drop_first=True)
    df_aligned = df_encoded.reindex(columns=expected_columns, fill_value=0)
    return df_aligned

@app.post("/predict")
def predict_qs_and_er(features: PitcherFeatures):
    if not all([qs_model_realtime, er_model_realtime, training_columns_realtime]):
        return JSONResponse(
            status_code=500,
            content={"error": "即時推論模型尚未成功載u, 請檢查伺服器日誌。"}
        )

    X_predict = process_realtime_features(features, training_columns_realtime)
    
    qs_probability = qs_model_realtime.predict_proba(X_predict)[:, 1][0]
    er_prediction = er_model_realtime.predict(X_predict)[0]
    
    return {
        "pitcher": features.pitcher,
        "opponent": features.opp_team,
        "predicted_qs_probability": np.round(qs_probability, 4),
        "predicted_qs_binary": 1 if qs_probability > 0.5 else 0,
        "predicted_er": np.round(er_prediction, 2)
    }

# =======================================================================
# III. 歷史儀表板 API (Use Case 2: 您的新腳本)
# =======================================================================

# --- 複製您腳本中的輔助函式 ---

def pick_row(df: pd.DataFrame, pitcher: str, date: str | None) -> pd.Series:
    """從已載入的 df_historical 中挑選一列"""
    mask = df["pitcher"].astype(str).str.lower() == str(pitcher).strip().lower()
    sub = df[mask]
    if date is not None and "game_date" in df.columns:
        try:
            dt = pd.to_datetime(date)
            sub = sub[pd.to_datetime(sub["game_date"]).dt.date == dt.date()]
        except Exception:
             raise ValueError("日期格式錯誤，請使用 YYYY-MM-DD")
    
    if sub.empty:
        raise ValueError("找不到符合的列，請檢查 pitcher 名稱或日期。")
    
    if len(sub) > 1:
        sub = sub.sort_values("game_date") if "game_date" in sub.columns else sub
        return sub.iloc[-1] # 取多筆中的最後一筆
    
    return sub.iloc[0]

def draw_gauge(prob: float, title: str, out_png: Path):
    """將儀表板繪製到指定的 out_png 路徑"""
    prob = float(np.clip(prob, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'aspect': 'equal'})
    ax.axis('off')
    outer = Wedge(center=(0, 0), r=1.0, theta1=180, theta2=0, width=0.28)
    ax.add_patch(outer)
    ax.add_patch(Wedge((0, 0), 1.0, 180, 120, width=0.28, alpha=0.25))
    ax.add_patch(Wedge((0, 0), 1.0, 120, 60,  width=0.28, alpha=0.25))
    ax.add_patch(Wedge((0, 0), 1.0, 60,  0,   width=0.28, alpha=0.25))
    theta = math.radians(180 * (1.0 - prob))
    r = 0.85
    x, y = r * math.cos(theta), r * math.sin(theta)
    ax.plot([0, x], [0, y], linewidth=3)
    ax.add_patch(Circle((0, 0), 0.035))
    ax.text(0, -0.20, f"{prob*100:.1f}%", ha="center", va="center", fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.3, 1.1)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)

@app.get("/get_historical_gauge", response_class=FileResponse)
def get_gauge_image(
    pitcher: str = Query(..., description="投手姓名 (例如: Aaron Civale)"), 
    date: Optional[str] = Query(None, description="比賽日期 (YYYY-MM-DD)。若省略，則回傳該投手最近一筆資料")
):
    """
    查詢一筆歷史資料，產生 QS 機率預測，並回傳儀表板 PNG 圖片。
    """
    if df_historical is None or qs_model_gauge is None:
        return JSONResponse(
            status_code=500,
            content={"error": "歷史資料或儀表板模型尚未成功載入，請檢查伺服器日誌。"}
        )
    
    try:
        # 1. 挑出要預測的「一列」
        row_s = pick_row(df_historical, pitcher=pitcher, date=date)
        
        # 2. 準備特徵
        missing = [c for c in FEATURES_GAUGE if c not in row_s]
        if missing:
             return JSONResponse(
                status_code=400,
                content={"error": f"選中的資料列缺少必要欄位: {missing}"}
            )
            
        X_one = pd.DataFrame([row_s[FEATURES_GAUGE].to_dict()])
        
        # 3. 預測機率
        prob = float(qs_model_gauge.predict_proba(X_one)[0, 1])
        
        # 4. 產生圖片
        game_date_str = str(row_s.get('game_date','')).split(' ')[0]
        title = f"{row_s.get('pitcher','?')} vs {row_s.get('opp_team','?')} ({game_date_str})\nQS Probability"
        
        # 建立一個暫存檔案來儲存圖片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            out_png_path = Path(tmp_file.name)
            
        draw_gauge(prob, title, out_png_path)
        
        # 5. 回傳圖片檔案
        # FastAPI 會在回傳後自動刪除此暫存檔案
        return FileResponse(out_png_path, media_type="image/png", delete_file_on_close=True)

    except (ValueError, IndexError) as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"處理時發生未預期錯誤: {e}"})


@app.get("/")
def read_root():
    return {"message": "歡迎使用 MLB 投手預測 API！請訪問 /docs 查看 API 文件。"}