# ============================================
# MLB 投手特徵資料自動生成（pybaseball 2.x 版本）
# 依賽季逐球資料 → 匯總為逐場 → 計算特徵 → 輸出 Excel
# ============================================
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from pybaseball import playerid_lookup, statcast_pitcher, pitching_stats, team_batting

# ========= 使用者參數 =========
PITCHER_FIRST = "Yoshinobu"
PITCHER_LAST  = "Yamamoto"
YEAR = 2025
START_DT = f"{YEAR}-01-01"
END_DT   = f"{YEAR}-12-31"
OUTFILE  = f"machine_learning/pitcher_record/{PITCHER_FIRST}_{PITCHER_LAST}_features_{YEAR}.xlsx".replace(" ", "_")

# ========= 取得 MLBAM 投手 ID =========
pid_df = playerid_lookup(PITCHER_LAST, PITCHER_FIRST)
if pid_df.empty:
    raise RuntimeError("找不到球員 ID，請確認姓名拼字。")
PITCHER_ID = int(pid_df["key_mlbam"].iloc[0])
print(f"MLBAM ID of {PITCHER_FIRST} {PITCHER_LAST}: {PITCHER_ID}")

# ========= 抓逐球資料（Statcast） =========
pitches = statcast_pitcher(START_DT, END_DT, PITCHER_ID)
if pitches.empty:
    raise RuntimeError("此年度沒有 Statcast 投球資料。")
pitches["game_date"] = pd.to_datetime(pitches["game_date"])

# ------- 由逐球 → 逐場匯總 -------
# 1) 每場投球數（pitches）= 該場列數
game_pcount = pitches.groupby("game_date").size().rename("pitches")

# 2) 出局數 → IP：利用 outs_when_up 的遞增量（僅統計投手在場時造成的出局）
def outs_from_half_inning(df):
    # 先用「同一場+同半局」的正向 diff 做基礎的 outs_delta
    df["outs_when_up"] = pd.to_numeric(df["outs_when_up"], errors="coerce").fillna(0)
    group_keys = [k for k in ["game_pk","inning_topbot"] if k in df.columns]
    df["outs_delta"] = (
        df.groupby(group_keys)["outs_when_up"]
        .diff().clip(lower=0).fillna(0).astype(int)
    )

    extra = pd.DataFrame([{
        "game_pk": df["game_pk"].iloc[-1],
        "game_date": df["game_date"].iloc[-1],
        "inning": df["inning"].iloc[-1] + 1 if "inning" in df.columns else np.nan,
        "outs_when_up": 0,
        "outs_delta": 0
    }])

    df = pd.concat([df, extra], ignore_index=True)

    # print(day[["game_date","inning","outs_when_up","outs_delta"]].to_string(index=True))

    # ---- 簡單修正：補上半局最後一球的「第3個出局」 ----
    # 若出現「上一列=2、下一列=0」且同一場，且半局或局數改變，就把前一列的 outs_delta +1
    idx = df.index.to_list()
    for i in range(1, len(df)):
        prev, cur = idx[i-1], idx[i]

        same_game = ("game_pk" in df.columns) and (df.at[prev, "game_pk"] == df.at[cur, "game_pk"])
        if not same_game:
            continue

        prev_out = df.at[prev, "outs_when_up"]
        cur_out  = df.at[cur, "outs_when_up"]

        # 半局/局數是否換了（任何一種變化都算半局結束）
        half_changed = False
        if "inning_topbot" in df.columns and df.at[prev,"inning_topbot"] != df.at[cur,"inning_topbot"]:
            half_changed = True
        if "inning" in df.columns and df.at[prev,"inning"] != df.at[cur,"inning"]:
            half_changed = True

        if prev_out == 2 and cur_out == 0 and half_changed:
            df.at[prev, "outs_delta"] += 1  # 給上一球補上第3個出局
    return df["outs_delta"].sum()

if "inning_topbot" in pitches.columns:
    pitches["inning_topbot"] = pitches["inning_topbot"].astype(str).str.strip().str.lower()

sort_cols = [c for c in ["game_pk","inning","inning_topbot","at_bat_number","pitch_number"] if c in pitches.columns]
pitches = pitches.sort_values(sort_cols)

outs_by_game = pitches.groupby("game_date").apply(outs_from_half_inning).rename("outs")
ip_by_game = (outs_by_game / 3.0).round(2).rename("IP_num")

# 3) 主客場與對手：若投手在「上半局」投球，則為主隊；多數票決定
def is_home_for_game(df):
    # inning_topbot: 'Top' or 'Bot'；主隊在 Top 投球
    # 以該場多數值決定（避免少數異常）
    top_ratio = (df["inning_topbot"].str.lower() == "top").mean()
    return 1 if top_ratio >= 0.5 else 0

home_flag = pitches.groupby("game_date").apply(is_home_for_game).rename("is_home")

# 取該場主客隊縮寫
teams = pitches.groupby("game_date")[["home_team", "away_team"]].agg(lambda x: x.iloc[0])
teams = teams.rename(columns={"home_team":"home_abbr","away_team":"away_abbr"})

# 對手隊伍縮寫
opp_team = pd.Series(
    np.where(home_flag.values==1, teams["away_abbr"].values, teams["home_abbr"].values),
    index=teams.index, name="opp_team"
)

# 4) 試算每場「失分（近似 ER）」：以對手分數的遞增量加總（近似，非正式 ER）
# Statcast 常有 home_score/away_score 欄位（得分變動），我們以分數增加量估算投手在場時的失分
def runs_allowed_estimate(df, is_home):
    # 選擇對手分數欄位
    score_col = "away_score" if is_home==1 else "home_score"
    if score_col not in df.columns:
        return np.nan
    sc = df[score_col].fillna(method="ffill").fillna(0)
    inc = sc.diff().clip(lower=0).fillna(0)
    return float(inc.sum())

runs_allowed = pitches.groupby("game_date").apply(
    lambda g: runs_allowed_estimate(g, is_home_for_game(g))
).rename("R_est")

# ------- 合併逐場匯總 -------
games = pd.concat([game_pcount, ip_by_game, home_flag, teams, opp_team, runs_allowed], axis=1).reset_index()

# 5) 近期 3 場平均
games = games.sort_values("game_date")
games["avg_ip_last3"] = games["IP_num"].rolling(3, min_periods=1).mean().round(2)
games["avg_er_last3"] = games["R_est"].rolling(3, min_periods=1).mean().round(2)

# 6) 休息天數
games["rest_days"] = games["game_date"].diff().dt.days
# 對第一場給個合理預設（例如 5 天）
games["rest_days"] = games["rest_days"].fillna(5).astype(int)

# 7) 最近 7 天累積投球數
games["cum_pitch_count_7d"] = [
    games.loc[(games["game_date"] >= d - timedelta(days=7)) & (games["game_date"] < d), "pitches"].sum()
    for d in games["game_date"]
]

# ========= 對手火力（OPS / wOBA） =========
tb = team_batting(YEAR).copy()  # FanGraphs 球隊打擊
# 嘗試用縮寫對齊：部分版本 Team 就是縮寫，否則請自行建立 mapping
tb.rename(columns={"Team": "opp_team"}, inplace=True)
# 若 Team 不是縮寫，可另外準備一個字典把 'New York Yankees'→'NYY' 之類做轉換

# 合併對手火力
games = games.merge(tb[["opp_team","OPS","wOBA"]], on="opp_team", how="left")
games.rename(columns={"OPS":"opp_ops","wOBA":"opp_woba"}, inplace=True)

# ========= 投手賽季屬性（ERA/WHIP/手別） =========
ps = pitching_stats(YEAR, YEAR, qual=0)
# 名稱在 FanGraphs 可能為 "First Last"；保險起見以姓氏包含過濾再挑選
ps_sel = ps[ps["Name"].str.contains(PITCHER_LAST, case=False, na=False)]
if ps_sel.empty:
    # 找不到就用整表再次以全名嘗試
    ps_sel = ps[ps["Name"] == f"{PITCHER_FIRST} {PITCHER_LAST}"]
if ps_sel.empty:
    raise RuntimeError("找不到投手的季級資料，請手動檢查名字在 FanGraphs 欄位的寫法。")

season_era = float(ps_sel["ERA"].iloc[0])
season_whip = float(ps_sel["WHIP"].iloc[0])

# 投手手別：直接取 Statcast 的 p_throws 眾數（R/L）
hand = pitches["p_throws"].dropna().mode()
hand = str(hand.iloc[0]) if not hand.empty else np.nan

games["season_era"] = season_era
games["season_whip"] = season_whip
games["hand"] = hand

# ========= 環境（park_factor / 天氣 / 主客場） =========
# 主客場已在 is_home。park_factor/天氣需外部來源；這裡先給預設或留空欄位，避免中斷。
# 你可以之後用 FanGraphs 的 Park Factors 表合併，或用 Open-Meteo 依日期+城市查天氣。
games["park_factor"] = np.nan    # 建議用 ballpark 對照表後續合併
games["temp_c"] = np.nan         # 建議外部氣象 API
games["humidity"] = np.nan       # 建議外部氣象 API

# ========= 產生 QS 標籤（用 R_est 近似 ER；若你有真實 ER，替換這一欄） =========
games["QS"] = ((games["IP_num"] >= 6) & (games["R_est"] <= 3)).astype(int)

# ========= 輸出欄位 =========
out_cols = [
    "game_date", "IP_num", "avg_ip_last3", "avg_er_last3",
    "rest_days", "cum_pitch_count_7d",
    "opp_team", "opp_ops", "opp_woba",
    "is_home", "park_factor", "temp_c", "humidity",
    "season_era", "season_whip", "hand", "pitches", "R_est", "home_abbr","away_abbr"
]

if os.path.exists(OUTFILE):
    os.remove(OUTFILE)

games[out_cols].to_excel(OUTFILE, index=False)
print(f"✅ Done. 輸出：{OUTFILE}")
