# ============================================
# MLB 全先發投手資料生成器 (for XGBoost: QS)
# 自動更新 Proxy + 每次啟動 driver 使用新 IP
# ============================================
import os
import random
import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from time import sleep
from tqdm import tqdm
from pybaseball import (
    playerid_lookup,
    statcast_pitcher,
    pitching_stats,
    team_batting,
    cache
)
from pythonbaseball import pitching_stats_range
from selenium import webdriver
from selenium.webdriver.edge.options import Options

# ===== 快取設定 =====
cache.purge()
cache.enable()

# ===== 自動抓取 Proxy Pool =====
def fetch_fresh_proxies():
    """
    從 proxyscrape API 自動抓取免費 Proxy List
    傳回格式為 ["ip:port", ...]
    """
    urls = [
        "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&proxy_format=ipport&timeout=2000&country=all",
        "https://www.proxy-list.download/api/v1/get?type=http",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/main/http.txt",
    ]
    proxies = []
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                new_proxies = [p.strip() for p in res.text.split("\n") if p.strip()]
                proxies.extend(new_proxies)
        except Exception:
            continue
    proxies = list(set(proxies))  # 去重
    if proxies:
        print(f"✅ 取得 {len(proxies)} 個代理 IP")
        return proxies
    else:
        print("⚠️ 無法從網路更新 Proxy，使用本地備用列表。")
        return [
            "103.152.112.120:80", "104.223.135.178:10000", "51.91.100.252:443",
            "190.61.84.166:9812", "8.219.97.248:80"
        ]

# ===== 初始化靜態代理池 (fallback) =====
proxy_list = fetch_fresh_proxies()

# ===== 年份與輸出資料夾 =====
YEAR = 2024
START_DT, END_DT = f"{YEAR}-03-28", f"{YEAR}-11-02"
OUTDIR = f"machine_learning/pitcher_record/all_pitchers_{YEAR}"
os.makedirs(OUTDIR, exist_ok=True)

# ===== 縮寫對照表 =====
TEAM_ABBREVIATION_MAP = {
    'BAL':'BAL','BOS':'BOS','NY':'NYY','TB':'TBR','TOR':'TOR',
    'CH':'CHW','CLE':'CLE','DET':'DET','KC':'KCR','MIN':'MIN',
    'HO':'HOU','LA':'LAA','OA':'OAK','SEA':'SEA','TX':'TEX',
    'ATL':'ATL','MI':'MIA','NYM':'NYM','PHI':'PHI','WAS':'WSN',
    'CHC':'CHC','CIN':'CIN','MIL':'MIL','PIT':'PIT','STL':'STL',
    'ARI':'ARI','AZ':'ARI','COL':'COL','LAD':'LAD','SD':'SDP','SF':'SFG',
    'CHW':'CHW','LAA':'LAA','NYY':'NYY','NYM':'NYM','TBR':'TBR',
    'WSN':'WSN','WSH':'WSN','MIA':'MIA','KCR':'KCR','CWS':'CHW','TBD':'TBR'
}

# ===== 抓取先發投手名單 =====
ps = pitching_stats(YEAR, YEAR, qual=0)
starters = ps[ps['GS'] > 0][['Name','Team','ERA','WHIP']].drop_duplicates()
print(f"✅ 抓到 {len(starters)} 位先發投手")

# ===== FanGraphs 球隊打擊表 =====
fg = team_batting(YEAR).copy()
fg["Team_std"] = fg["Team"].map(TEAM_ABBREVIATION_MAP).fillna(fg["Team"])

# ===== 建立 driver 的函式 =====
    # 每次重新抓取新 proxy pool
    # proxy_list = fetch_fresh_proxies()
    # proxy = random.choice(proxy_list)
    # print(f"🌐 使用代理 IP: {proxy}")

    # options = ChromeOptions()
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--window-size=1920,1080")
    # options.add_argument("--start-maximized")
    # options.add_argument(f"--proxy-server={proxy}")
    # driver = Chrome(options=options)

user_agents = [
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
"Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
]

edge_options = Options()
edge_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
#edge_options.add_argument("--disable-blink-features=AutomationControlled")
edge_options.add_argument("--no-sandbox")
edge_options.add_argument("--disable-dev-shm-usage")
#edge_options.add_argument("--headless=chrome")
edge_options.add_argument("--window-size=1920,1080")  # 強制設定解析度
edge_options.add_argument("--start-maximized")  # 避免被網站識破
edge_options.add_argument(f"user-agent={random.choice(user_agents)}")
driver = webdriver.Chrome(options=edge_options)
driver.implicitly_wait(10)

# ===== 主流程 =====
for idx, row in tqdm(starters.iterrows(), total=len(starters), desc="Processing pitchers"):
    try:
        name = row["Name"]
        if not isinstance(name, str) or len(name.strip()) == 0:
            continue
        print(f"\n🎯 處理投手：{name}")

        # === 取得投手 MLBAM ID ===
        parts = name.split(" ", 1)
        first, last = parts[0], (parts[1] if len(parts) > 1 else "")
        pid = playerid_lookup(last, first)
        if pid.empty:
            print(f"⚠️ 找不到 {name} 的 MLBAM ID，略過")
            continue
        MLBAM = int(pid["key_mlbam"].iloc[0])
        OUTFILE = f"{OUTDIR}/{name.replace(' ','_')}_{YEAR}_features.xlsx"

        # === 抓 Statcast ===
        pitches = statcast_pitcher(START_DT, END_DT, MLBAM)
        if pitches.empty:
            print(f"⚠️ {name} 沒有 Statcast 資料")
            continue

        pitches["game_date"] = pd.to_datetime(pitches["game_date"])
        pitches = pitches.sort_values(["game_pk","inning","inning_topbot","at_bat_number","pitch_number"])
        if "inning_topbot" in pitches.columns:
            pitches["inning_topbot"] = pitches["inning_topbot"].astype(str).str.strip().str.lower()

        def is_home_for_game(df):
            return 1 if (df["inning_topbot"].str.lower()=="top").mean() >= 0.5 else 0

        home_flag = pitches.groupby("game_pk").apply(is_home_for_game).rename("is_home")
        teams = pitches.groupby("game_pk")[["home_team","away_team","game_date"]].agg(lambda s: s.iloc[0])
        teams["opp_team"] = np.where(home_flag.values==1, teams["away_team"], teams["home_team"])
        teams = teams.join(home_flag)
        teams["game_date"] = pd.to_datetime(teams["game_date"])
        dates = teams["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()

        # === 抓逐場 ===
        rows = []
        for d in dates:
            print(f"{name} - {d}")
            day = pitching_stats_range(driver, d, d)
            if day is None or day.empty:
                continue
            p = day[day["Name"] == name]
            if p.empty:
                continue

            def num(s, default=np.nan): return pd.to_numeric(s, errors="coerce").fillna(default)
            rec = {
                "pitcher": name, "game_date": d,
                "IP": float(num(p["IP"]).iloc[0]) if "IP" in p.columns else np.nan,
                "ER": int(num(p["ER"],0).iloc[0]) if "ER" in p.columns else np.nan,
                "R":  int(num(p["R"],0).iloc[0]) if "R" in p.columns else np.nan,
                "H":  int(num(p["H"],0).iloc[0]) if "H" in p.columns else np.nan,
                "BB": int(num(p["BB"],0).iloc[0]) if "BB" in p.columns else np.nan,
                "SO": int(num(p["SO"],0).iloc[0]) if "SO" in p.columns else np.nan,
                "Pit":int(num(p["Pit"],np.nan).iloc[0]) if "Pit" in p.columns else np.nan,
                "Team":p["Team"].iloc[0] if "Team" in p.columns else None,
                "Opp": p["Opp"].iloc[0]  if "Opp"  in p.columns else None,
            }
            rows.append(rec)

        games = pd.DataFrame(rows)
        if games.empty:
            print(f"⚠️ {name} 無逐場資料")
            continue

        # === 後續特徵工程 ===
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.sort_values("game_date").reset_index(drop=True)

        # 若沒 Pit → 用 Statcast 補
        if games["Pit"].isna().any():
            pit_statcast = (
                pitches.groupby("game_pk").size().rename("Pit_sc").reset_index()
                .merge(teams[["game_pk","game_date"]].reset_index(), on="game_pk", how="left")
                .groupby("game_date")["Pit_sc"].sum().reset_index()
            )
            games = games.merge(pit_statcast, on="game_date", how="left")
            games["Pit"] = games["Pit"].fillna(games["Pit_sc"]).astype("Int64")
            games.drop(columns=["Pit_sc"], inplace=True, errors="ignore")

        games = games.merge(
            teams.reset_index()[["game_pk","game_date","opp_team","is_home"]],
            on="game_date", how="left"
        ).drop_duplicates(subset=["game_pk","game_date"]).reset_index(drop=True)

        games["rest_days"] = games["game_date"].diff().dt.days.fillna(5).astype(int)
        games["avg_ip_last3"] = games["IP"].rolling(3, min_periods=1).mean().round(2)
        games["avg_er_last3"] = games["ER"].rolling(3, min_periods=1).mean().round(2)

        games["opp_team_std"] = games["opp_team"].map(TEAM_ABBREVIATION_MAP).fillna(games["opp_team"])
        games = games.merge(
            fg.rename(columns={"Team_std":"opp_team_std"})[["opp_team_std","OPS","wOBA"]],
            on="opp_team_std", how="left"
        )
        games.rename(columns={"OPS":"opp_ops"}, inplace=True)

        season_era, season_whip = row["ERA"], row["WHIP"]
        hand = pitches["p_throws"].dropna().mode()
        hand = str(hand.iloc[0]) if not hand.empty else None
        games["season_era"] = season_era
        games["season_whip"] = season_whip
        games["hand"] = hand

        out_cols = [
            "pitcher","game_date","IP","ER","R","H","BB","SO",
            "Pit","rest_days","opp_ops","is_home",
            "avg_ip_last3","avg_er_last3",
            "opp_team","Team","season_era","season_whip","hand"
        ]
        games_out = games[out_cols].sort_values("game_date")
        games_out.to_excel(OUTFILE, index=False)
        print(f"✅ 完成 {name} -> {OUTFILE}")

        sleep(random.uniform(2,5))  # 模擬人類行為，避免封鎖

    except Exception as e:
        print(f"❌ {row.get('Name', idx)} 錯誤: {e}")
        continue
driver.close()  # ✅ 每位投手結束後關閉當前 driver
print("🎯 全部投手處理完畢！")
