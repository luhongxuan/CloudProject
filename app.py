# app.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from datetime import date
import os, joblib, psycopg
import pandas as pd

FEATURES = ["rest_days","opp_ops","is_home","avg_ip_last3","avg_er_last3",
            "season_era","season_whip","hand","opp_team","Team","pitcher"]

MODEL_PATH = "./artifacts_qs_xgb/qs_xgb_classifier_calibrated.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動：載模型、開 DB 連線
    app.state.pipe = joblib.load(MODEL_PATH)
    app.state.db = psycopg.connect(os.environ["DATABASE_URL"])
    try:
        yield
    finally:
        try:
            app.state.db.close()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    # 讓首頁自動導到 Swagger
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(
    request: Request,
    pitcher: str = Query(..., description="投手姓名（完全比對）"),
    game_date: date | None = Query(None, description="YYYY-MM-DD；不給就取最近一場")
):
    """
    依投手＋（可選）日期，查 DB 抓同場特徵，丟進校準過的管線模型產生 QS 機率。
    """
    db = request.app.state.db
    pipe = request.app.state.pipe

    if game_date is None:
        # 只比對投手，取最近一場
        sql = """
        SELECT rest_days, opp_ops, is_home, avg_ip_last3, avg_er_last3,
               season_era, season_whip, hand, opp_team, team AS "Team", pitcher
        FROM pitcher_features
        WHERE pitcher = %s
        ORDER BY game_date DESC
        LIMIT 1
        """
        params = (pitcher,)
    else:
        # 同時比對投手＋日期（資料表 game_date 是 DATE 型別就可直接 =）
        sql = """
        SELECT rest_days, opp_ops, is_home, avg_ip_last3, avg_er_last3,
               season_era, season_whip, hand, opp_team, team AS "Team", pitcher
        FROM pitcher_features
        WHERE pitcher = %s AND game_date = %s
        LIMIT 1
        """
        params = (pitcher, game_date)

    with db.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No matching row (check pitcher name / date).")

    df = pd.DataFrame([row], columns=FEATURES)
    prob = float(pipe.predict_proba(df)[0, 1])
    return {
        "pitcher": pitcher,
        "game_date": (game_date.isoformat() if game_date else "latest"),
        "qs_prob": prob
    }
