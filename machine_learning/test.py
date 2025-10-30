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
OUTFILE  = f"pitcher_record/{PITCHER_FIRST}_{PITCHER_LAST}_features_{YEAR}.xlsx".replace(" ", "_")

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
DATE = "2025-10-25"

# 取一天並排序（確保時序正確）
day = pitches.copy()
day["game_date"] = pd.to_datetime(day["game_date"])
day = day[day["game_date"].dt.strftime("%Y-%m-%d") == DATE].copy()

# 標準化半局標記
if "inning_topbot" in day.columns:
    day["inning_topbot"] = day["inning_topbot"].astype(str).str.strip().str.lower()

sort_cols = [c for c in ["game_pk","inning","inning_topbot","at_bat_number","pitch_number"] if c in day.columns]
day = day.sort_values(sort_cols)

# 先用「同一場+同半局」的正向 diff 做基礎的 outs_delta
day["outs_when_up"] = pd.to_numeric(day["outs_when_up"], errors="coerce").fillna(0)
group_keys = [k for k in ["game_pk","inning_topbot"] if k in day.columns]
day["outs_delta"] = (
    day.groupby(group_keys)["outs_when_up"]
       .diff().clip(lower=0).fillna(0).astype(int)
)

extra = pd.DataFrame([{
    "game_date": day["game_date"].iloc[-1],
    "inning": day["inning"].iloc[-1] + 1 if "inning" in day.columns else np.nan,
    "outs_when_up": 0,
    "outs_delta": 0
}])

day = pd.concat([day, extra], ignore_index=True)

# print(day[["game_date","inning","outs_when_up","outs_delta"]].to_string(index=True))

# ---- 簡單修正：補上半局最後一球的「第3個出局」 ----
# 若出現「上一列=2、下一列=0」且同一場，且半局或局數改變，就把前一列的 outs_delta +1
idx = day.index.to_list()
for i in range(1, len(day)):
    prev, cur = idx[i-1], idx[i]
    print(i)

    same_game = ("game_pk" in day.columns) and (day.at[prev, "game_pk"] == day.at[cur, "game_pk"])
    if not same_game:
        continue

    prev_out = day.at[prev, "outs_when_up"]
    cur_out  = day.at[cur, "outs_when_up"]

    # 半局/局數是否換了（任何一種變化都算半局結束）
    half_changed = False
    if "inning_topbot" in day.columns and day.at[prev,"inning_topbot"] != day.at[cur,"inning_topbot"]:
        half_changed = True
    if "inning" in day.columns and day.at[prev,"inning"] != day.at[cur,"inning"]:
        half_changed = True

    if prev_out == 2 and cur_out == 0 and half_changed:
        day.at[prev, "outs_delta"] += 1  # 給上一球補上第3個出局

# 只印你要看的兩欄（可加 inning 方便檢查）
print(day[["game_date","inning","outs_when_up","outs_delta"]].to_string(index=True))

# 當天每場的總出局與 IP
outs_by_game = day.groupby("game_pk")["outs_delta"].sum().rename("outs")
ip_by_game = (outs_by_game / 3.0).round(2).rename("IP_num")
print("\n=== 當天逐場合計 ===")
print(pd.concat([outs_by_game, ip_by_game], axis=1).to_string())
