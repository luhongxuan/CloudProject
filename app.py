# app.py
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os, joblib, psycopg
import pandas as pd

FEATURES = ["rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
            "season_era","season_whip","hand","opp_team","team","pitcher"]

MODEL_PATH = "./artifacts_qs_xgb/qs_xgb_classifier_calibrated.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup：載入模型與連線資料庫 ---
    pipe = joblib.load(MODEL_PATH)                     # sklearn Pipeline/CalibratedClassifierCV
    db = psycopg.connect(os.environ["DATABASE_URL"])   # Render Postgres（建議用 Internal URL）
    app.state.pipe = pipe
    app.state.db = db
    try:
        yield
    finally:
        # --- shutdown：關閉資料庫連線 ---
        try:
            db.close()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)  # ← 取代 @app.on_event(...) 寫法

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(request: Request, pitcher: str = Query(...)):
    # 取出共享資源
    db = request.app.state.db
    pipe = request.app.state.pipe

    q = """
      SELECT rest_days, opp_ops, is_home, avg_ip_last3, avg_er_last3,
             season_era, season_whip, hand, opp_team, team, pitcher
      FROM pitcher_features
      WHERE pitcher = %s
      ORDER BY game_date DESC
      LIMIT 1
    """
    with db.cursor() as cur:
        cur.execute(q, (pitcher,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="pitcher not found")

    df = pd.DataFrame([row], columns=FEATURES)
    prob = float(pipe.predict_proba(df)[0, 1])  # Pipeline 會先做前處理再輸出機率
    return {"pitcher": pitcher, "qs_prob": prob}
