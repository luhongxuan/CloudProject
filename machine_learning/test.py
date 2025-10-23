# ============================================
# MLB 投手特徵資料自動生成（pybaseball 2.x 版本）
# 依賽季逐球資料 → 匯總為逐場 → 計算特徵 → 輸出 Excel
# ============================================
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
OUTFILE  = f"{PITCHER_FIRST}_{PITCHER_LAST}_features_{YEAR}.xlsx".replace(" ", "_")

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
import pandas as pd

DATE = "2025-03-18"

# 取一天並排序（確保 diff 正確）
day = pitches.copy()
day["game_date"] = pd.to_datetime(day["game_date"])
day = day[day["game_date"].dt.strftime("%Y-%m-%d") == DATE].copy()

sort_cols = [c for c in ["game_pk","inning","inning_topbot","at_bat_number","pitch_number"] if c in day.columns]
day = day.sort_values(sort_cols)

# 確保 outs_when_up 是數值
day["outs_when_up"] = pd.to_numeric(day["outs_when_up"], errors="coerce")

# ✅ 关键：用 groupby().diff()（會保留原索引，不會出現不相容索引）
day["outs_delta"] = (
    day.groupby(["game_pk","inning_topbot"])["outs_when_up"]
       .diff()                          # 同半局內的變化
       .clip(lower=0)                   # 只保留正向增加
       .fillna(0)
       .astype(int)
)

# 只印你要的兩欄
print(day[["outs_when_up", "outs_delta"]])

# （可選）當天每場總出局數與投球局數
outs_by_game = day.groupby("game_pk")["outs_delta"].sum().rename("outs")
ip_by_game = (outs_by_game / 3.0).round(2).rename("IP_num")
print(pd.concat([outs_by_game, ip_by_game], axis=1))
