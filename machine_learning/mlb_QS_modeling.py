import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- 1. 設定：請更新您的檔案名稱 ---
# 這是您統整好的 Excel 檔案名稱
CONSOLIDATED_FILE = r'C:\CloudProject\machine_learning\pitcher_record\All_Pitchers.xlsx'
def load_and_prepare_data(filepath: str) -> tuple:
    """
    從指定的單一 Excel 檔案載入數據，並進行清理與特徵工程。
    """
    print(f"--- 1. 正在載入數據: {filepath} ---")
    
    if not os.path.exists(filepath):
         print(f"❌ 錯誤：找不到檔案 {filepath}。請檢查檔案名稱和路徑。")
         return None, None, None
         
    try:
        df = pd.read_excel(filepath)
        print(f"✅ 檔案讀取成功！總共 {len(df)} 筆記錄。")
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        return None, None, None

    # --- 2. 數據清理與轉換 ---
    print("--- 2. 正在清理數據與建立目標變數 ---")
    
    # 確保所有用於計算的欄位都是數值類型
    numeric_cols = [
        'IP', 'ER', 'R', 'H', 'BB', 'SO', 'Pit', 'rest_days', 'opp_ops', 
        'is_home', 'avg_ip_last3', 'avg_er_last3', 'season_era', 'season_whip'
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 簡單處理缺失值：假設缺失值為 0 或不適用
    df = df.fillna(0) 

    # --- 3. 建立目標變數 (Y) ---
    
    # A. 分類目標：QS (優質先發)
    # 定義：投球局數 (IP) >= 6 且 自責分 (ER) <= 3
    df['QS_Target'] = ((df['IP'] >= 6) & (df['ER'] <= 3)).astype(int)
    
    # B. 迴歸目標：單場自責分 (ER)
    df['ER_Target'] = df['ER']
    
    print(f"QS (優質先發) 目標分佈:\n{df['QS_Target'].value_counts(normalize=True) * 100}")

    # --- 4. 準備特徵 (X) ---
    print("--- 4. 正在進行特徵工程 (One-Hot Encoding) ---")
    
    # 選擇我們認為有預測能力的類別特徵
    # 【修正 1】: 將 'opp' 改為 'opp_team' (根據您的錯誤訊息)
    categorical_cols = ['pitcher', 'opp_team', 'Team', 'hand'] 
    
    # 確保這些欄位都存在於 DataFrame 中
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    
    # 將類別特徵轉換為 One-Hot 編碼
    df_encoded = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
    
    # 準備 X (特徵) 和 Y (目標)
    
    # 移除會造成「數據洩漏」(Data Leakage) 的欄位
    features_to_drop = [
        'IP', 'ER', 'R', 'H', 'BB', 'SO', 'Pit', # 比賽結果
        'QS_Target', 'ER_Target', # 目標變數
        'game_date', # 非特徵
        'Pitcher_Source' # 【修正 2】: 刪除這個識別符欄位
    ]
    
    # 確保所有要刪除的欄位都存在
    existing_cols_to_drop = [col for col in features_to_drop if col in df_encoded.columns]
    
    X = df_encoded.drop(columns=existing_cols_to_drop)
    
    y_qs = df_encoded['QS_Target'] 
    y_er = df_encoded['ER_Target'] 
    
    print(f"✅ 數據準備完成！最終使用 {X.shape[1]} 個特徵。")
    
    return X, y_qs, y_er

def train_and_evaluate_models(X: pd.DataFrame, y_qs: pd.Series, y_er: pd.Series):
    """
    訓練並評估 QS 分類和 ER 迴歸模型。
    """
    print("\n--- 4. 正在劃分訓練集與測試集 ---")
    
    # (A) 劃分 QS 模型的數據 (使用 stratify 確保 QS 比例在訓練/測試集中一致)
    X_train_qs, X_test_qs, y_qs_train, y_qs_test = train_test_split(
        X, y_qs, test_size=0.1, random_state=42, stratify=y_qs 
    )
    
    # (B) 劃分 ER 模型的數據
    X_train_er, X_test_er, y_er_train, y_er_test = train_test_split(
        X, y_er, test_size=0.1, random_state=42
    )
    
    print(f"訓練集大小: {X_train_qs.shape[0]} | 測試集大小: {X_test_qs.shape[0]}")

    # --- 5. 訓練模型 A：QS 機率 (分類) ---
    print("\n--- 5. 正在訓練 XGBoost QS 分類模型 ---")
    
    # 初始化 XGBoost 分類器
    qs_model = XGBClassifier(
        objective='binary:logistic',  # 二元分類，輸出機率
        n_estimators=100,             # 樹的數量
        learning_rate=0.1,            # 學習率
        max_depth=5,                  # 樹的最大深度
        use_label_encoder=False,
        eval_metric='logloss',        # 評估指標
        random_state=42
    )
    
    qs_model.fit(X_train_qs, y_qs_train)
    
    # --- 評估 QS 模型 ---
    print("\n--- QS 模型 (分類) 評估結果 ---")
    y_qs_pred = qs_model.predict(X_test_qs)
    y_qs_proba = qs_model.predict_proba(X_test_qs)[:, 1] # 取得「是QS」的機率

    print(f"✅ 準確度 (Accuracy): {accuracy_score(y_qs_test, y_qs_pred):.4f}")
    print("分類報告 (Classification Report):")
    print(classification_report(y_qs_test, y_qs_pred))
    print(f"範例預測機率 (前 5 筆): {np.round(y_qs_proba[:5], 3)}")

    # --- 6. 訓練模型 B：失分 (迴歸) ---
    print("\n--- 6. 正在訓練 XGBoost 失分 (ER) 迴歸模型 ---")
    
    er_model = XGBRegressor(
        objective='reg:squarederror', # 迴歸任務，目標是最小化平方誤差
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    er_model.fit(X_train_er, y_er_train)
    
    # --- 評估 ER 模型 ---
    print("\n--- ER 模型 (迴歸) 評估結果 ---")
    y_er_pred = er_model.predict(X_test_er)
    
    mse = mean_squared_error(y_er_test, y_er_pred)
    r2 = r2_score(y_er_test, y_er_pred)
    
    print(f"✅ 均方根誤差 (RMSE): {np.sqrt(mse):.4f} (代表平均預測失分與實際失分的差距)")
    print(f"✅ R-squared (R2): {r2:.4f} (代表模型對數據變異的解釋程度)")
    print(f"範例預測失分 (前 5 筆): {np.round(y_er_pred[:10], 2)}")
    print(f"實際失分 (前 5 筆): {y_er_test.iloc[:10].values}")

    # --- 【新增步驟】：儲存模型與特徵列表 ---
    print("\n--- 9. 正在打包（儲存）模型 ---")
    
    # 1. 儲存 QS 分類模型
    joblib.dump(qs_model, 'qs_model.joblib')
    
    # 2. 儲存 ER 迴歸模型
    joblib.dump(er_model, 'er_model.joblib')
    
    # 3. 儲存訓練時使用的欄位列表 (X_train_qs 和 X_train_er 的欄位相同)
    # 這是最關鍵的一步，確保未來預測時欄位一致
    training_columns = X_train_qs.columns.tolist()
    joblib.dump(training_columns, 'training_columns.pkl') 
    
    print("✅ 模型和欄位列表已儲存！")
    
    return qs_model, er_model

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    
    # 1. 載入並準備數據
    X_features, y_qs, y_er = load_and_prepare_data(CONSOLIDATED_FILE)
    
    if X_features is not None:
        # 2. 訓練並評估模型
        qs_model, er_model = train_and_evaluate_models(X_features, y_qs, y_er)
        
        print("\n🎉🎉🎉 流程執行完畢！ 🎉🎉🎉")