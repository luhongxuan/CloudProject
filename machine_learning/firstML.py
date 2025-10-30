# ============================================
# 整合版：逐場投手特徵表 (for XGBoost: QS)
# - 來源1：Baseball-Reference via pitching_stats_range (IP/ER/R/H/BB/SO/Pit)
# - 來源2：Statcast (確認登板日期、主客場/對手、回退球數)
# - 來源3：FanGraphs team_batting (對手 OPS/wOBA)
# - 加入：pybaseball cache、自動縮寫映射
# ============================================
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from pybaseball import (
    playerid_lookup,
    statcast_pitcher,
    pitching_stats_range,
    pitching_stats,
    team_batting,
    cache
)

# ===== 快取啟用 =====
cache.enable()  # 啟用 pybaseball 官方快取
# print(f"✅ pybaseball 快取已啟用：{cache.cache_location()}")

# ===== 隊伍縮寫映射表 =====
TEAM_ABBREVIATION_MAP = {
    # AL East
    'BAL': 'BAL', 'BOS': 'BOS', 'NY': 'NYY', 'TB': 'TBR', 'TOR': 'TOR',
    # AL Central
    'CH': 'CHW', 'CLE': 'CLE', 'DET': 'DET', 'KC': 'KCR', 'MIN': 'MIN',
    # AL West
    'HO': 'HOU', 'LA': 'LAA', 'OA': 'OAK', 'SEA': 'SEA', 'TX': 'TEX',
    # NL East
    'ATL': 'ATL', 'MI': 'MIA', 'NYM': 'NYM', 'PHI': 'PHI', 'WAS': 'WSN',
    # NL Central
    'CHC': 'CHC', 'CIN': 'CIN', 'MIL': 'MIL', 'PIT': 'PIT', 'STL': 'STL',
    # NL West
    'ARI': 'ARI', 'AZ': 'ARI', 'COL': 'COL', 'LAD': 'LAD', 'SD': 'SDP', 'SF': 'SFG',
    # 歷史或常見替代簡稱
    'CHW': 'CHW', 'LAA': 'LAA', 'NYY': 'NYY', 'NYM': 'NYM', 'TBR': 'TBR',
    'WSN': 'WSN', 'WSH': 'WSN', 'MIA': 'MIA', 'KCR': 'KCR', 'CWS': 'CHW', 'TBD': 'TBR'
}

# ===== 參數 =====
FIRST, LAST, YEAR = "Yoshinobu", "Yamamoto", 2024
START_DT, END_DT = f"{YEAR}-03-20", f"{YEAR}-11-02"
OUTFILE = f"machine_learning/pitcher_record/{FIRST}_{LAST}_{YEAR}_game_features.xlsx".replace(" ", "_")

# ===== 取得 MLBAM ID =====
pid = playerid_lookup(LAST, FIRST)
if pid.empty:
    raise RuntimeError("找不到球員 ID，請檢查姓名。")
MLBAM = int(pid["key_mlbam"].iloc[0])
DISPLAY_NAME = f"{FIRST} {LAST}"
print("MLBAM:", MLBAM)

# ===== Statcast：抓逐球資料 =====
pitches = statcast_pitcher(START_DT, END_DT, MLBAM)
if pitches.empty:
    raise RuntimeError("此年度沒有 Statcast 投球資料。")
pitches["game_date"] = pd.to_datetime(pitches["game_date"])
pitches = pitches.sort_values(["game_pk","inning","inning_topbot","at_bat_number","pitch_number"])

# ===== 主客場旗標 =====
if "inning_topbot" in pitches.columns:
    pitches["inning_topbot"] = pitches["inning_topbot"].astype(str).str.strip().str.lower()

def is_home_for_game(df):
    top_ratio = (df["inning_topbot"].str.lower() == "top").mean()
    return 1 if top_ratio >= 0.5 else 0

home_flag = pitches.groupby("game_pk").apply(is_home_for_game).rename("is_home")
teams = pitches.groupby("game_pk")[["home_team","away_team","game_date"]].agg(lambda s: s.iloc[0])
teams["opp_team"] = np.where(home_flag.values==1, teams["away_team"], teams["home_team"])
teams = teams.join(home_flag)
teams["game_date"] = pd.to_datetime(teams["game_date"])

# 登板日期清單
dates = teams["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()

# ===== 逐場投手成績 =====
rows = []
for d in dates:
    print("抓取日期:", d)
    day = pitching_stats_range(d, d)
    if day is None or day.empty:
        continue
    p = day[day["Name"] == DISPLAY_NAME]
    if p.empty:
        continue

    def num(s, default=np.nan):
        return pd.to_numeric(s, errors="coerce").fillna(default)

    rec = {
        "pitcher": DISPLAY_NAME,
        "game_date": d,
        "IP":   float(num(p["IP"]).iloc[0]) if "IP" in p.columns else np.nan,
        "ER":   int(num(p["ER"], 0).iloc[0]) if "ER" in p.columns else np.nan,
        "R":    int(num(p["R"], 0).iloc[0])  if "R"  in p.columns else np.nan,
        "H":    int(num(p["H"], 0).iloc[0])  if "H"  in p.columns else np.nan,
        "BB":   int(num(p["BB"], 0).iloc[0]) if "BB" in p.columns else np.nan,
        "SO":   int(num(p["SO"], 0).iloc[0]) if "SO" in p.columns else np.nan,
        "Pit":  int(num(p["Pit"], np.nan).iloc[0]) if "Pit" in p.columns else np.nan,
        "Team": p["Team"].iloc[0] if "Team" in p.columns else None,
        "Opp":  p["Opp"].iloc[0]  if "Opp"  in p.columns else None,
    }
    rows.append(rec)

games = pd.DataFrame(rows)
if games.empty:
    raise RuntimeError("pitching_stats_range 沒抓到任何逐場資料（Name 對不上？請檢查姓名或 print(day.columns)）。")

# ===== 日期 & 排序 =====
games["game_date"] = pd.to_datetime(games["game_date"])
games = games.sort_values("game_date").reset_index(drop=True)

# 若某天沒 Pit → 回退 Statcast 總筆數
if games["Pit"].isna().any():
    pit_statcast = (
        pitches.groupby("game_pk").size().rename("Pit_sc").reset_index()
        .merge(teams[["game_pk","game_date"]].reset_index(), on="game_pk", how="left")
        .groupby("game_date")["Pit_sc"].sum().reset_index()
    )
    games = games.merge(pit_statcast, on="game_date", how="left")
    games["Pit"] = games["Pit"].fillna(games["Pit_sc"]).astype("Int64")
    games.drop(columns=["Pit_sc"], inplace=True, errors="ignore")

# ===== 合併主客場/對手 =====
games = games.merge(
    teams.reset_index()[["game_pk","game_date","opp_team","is_home"]],
    on="game_date", how="left"
)
games = games.drop_duplicates(subset=["game_pk","game_date"]).reset_index(drop=True)

# ===== 休息天數 =====
games["rest_days"] = games["game_date"].diff().dt.days.fillna(5).astype(int)

# ===== 近期平均 =====
games["avg_ip_last3"] = games["IP"].rolling(3, min_periods=1).mean().round(2)
games["avg_er_last3"] = games["ER"].rolling(3, min_periods=1).mean().round(2)

# ===== 對手 OPS / wOBA (含映射表) =====
fg = team_batting(YEAR).copy()
fg["Team_std"] = fg["Team"].map(TEAM_ABBREVIATION_MAP).fillna(fg["Team"])
games["opp_team_std"] = games["opp_team"].map(TEAM_ABBREVIATION_MAP).fillna(games["opp_team"])

games = games.merge(
    fg.rename(columns={"Team_std":"opp_team_std"})[["opp_team_std","OPS","wOBA"]],
    on="opp_team_std", how="left"
)
games.rename(columns={"OPS":"opp_ops"}, inplace=True)

# ===== 投手賽季屬性 =====
ps = pitching_stats(YEAR, YEAR, qual=0)
ps_sel = ps[ps["Name"].str.fullmatch(DISPLAY_NAME, case=False, na=False)]
if ps_sel.empty:
    ps_sel = ps[ps["Name"].str.contains(LAST, case=False, na=False)]
season_era  = float(ps_sel["ERA"].iloc[0]) if not ps_sel.empty else np.nan
season_whip = float(ps_sel["WHIP"].iloc[0]) if not ps_sel.empty else np.nan
hand = pitches["p_throws"].dropna().mode()
hand = str(hand.iloc[0]) if not hand.empty else None

games["season_era"] = season_era
games["season_whip"] = season_whip
games["hand"] = hand

# ===== 輸出 =====
out_cols = [
    "pitcher", "game_date", "IP", "ER", "R", "H", "BB", "SO",
    "Pit", "rest_days", "opp_ops", "is_home",
    "avg_ip_last3", "avg_er_last3",
    "opp_team", "Team", "season_era", "season_whip", "hand"
]
games_out = games[out_cols].sort_values("game_date")

if os.path.exists(OUTFILE):
    os.remove(OUTFILE)
games_out.to_excel(OUTFILE, index=False)
print(f"✅ 輸出完成：{OUTFILE}")
