import pandas as pd
import numpy as np
import os
# 【新導入】: 導入 RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- 1. 設定：請更新您的檔案名稱 ---
CONSOLIDATED_FILE = r'C:\loginonly\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx'

def load_and_prepare_data(filepath: str) -> tuple:
    """
    從指定的單一 Excel 檔案載入數據，並進行清理與特徵工程。
    (已移除時間序列排序)
    """
    print(f"--- 1. 正在載入數據: {filepath} ---")
    
    if not os.path.exists(filepath):
         print(f"❌ 錯誤：找不到檔案 {filepath}。請檢查檔案名稱和路徑。")
         return None, None, None, None
         
    try:
        df = pd.read_excel(filepath)
        print(f"✅ 檔案讀取成功！總共 {len(df)} 筆記錄。")
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        return None, None, None, None

    # --- 2. 數據清理與轉換 ---
    print("--- 2. 正在清理數據與建立目標變數 ---")
    
    numeric_cols = [
        'IP', 'ER', 'R', 'H', 'BB', 'SO', 'Pit', 'rest_days', 'opp_ops', 
        'is_home', 'avg_ip_last3', 'avg_er_last3', 'season_era', 'season_whip'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0) 

    # --- 3. (已移除時間序列排序) ---
    
    # --- 4. 建立目標變數 (Y) ---
    df['QS_Target'] = ((df['IP'] >= 6) & (df['ER'] <= 3)).astype(int)
    df['ER_Target'] = df['ER']
    print(f"QS (優質先發) 目標分佈:\n{df['QS_Target'].value_counts(normalize=True) * 100}")

    # --- 5. 準備特徵 (X) ---
    print("--- 5. 正在進行特徵工程 (One-Hot Encoding) ---")
    
    categorical_cols = ['pitcher', 'opp_team', 'Team', 'hand'] 
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
    
    # 移除會造成「數據洩漏」(Data Leakage) 的欄位
    features_to_drop = [
        'IP', 'ER', 'R', 'H', 'BB', 'SO', 'Pit', # 比賽結果
        'QS_Target', 'ER_Target', # 目標變數
        'Pitcher_Source', # 識別符欄位
        'game_date' # 【修改】: 重新加入 'game_date' 到移除列表
    ]
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df_encoded.columns]
    
    X = df_encoded.drop(columns=existing_cols_to_drop)
    y_qs = df_encoded['QS_Target'] 
    y_er = df_encoded['ER_Target'] 
    
    print(f"✅ 數據準備完成！最終使用 {X.shape[1]} 個特徵。")
    
    return X, y_qs, y_er

def train_and_evaluate_models(X: pd.DataFrame, y_qs: pd.Series, y_er: pd.Series):
    """
    使用【隨機切割】法訓練、調優並評估模型。
    """
    print("\n--- 6. 正在劃分訓練集與測試集 (隨機切割) ---")
    
    # 【修改 3】: 重新使用 train_test_split
    X_train_qs, X_test_qs, y_qs_train, y_qs_test = train_test_split(
        X, y_qs, test_size=0.2, random_state=42, stratify=y_qs 
    )
    X_train_er, X_test_er, y_er_train, y_er_test = train_test_split(
        X, y_er, test_size=0.2, random_state=42
    )
    
    print(f"訓練集大小: {X_train_qs.shape[0]} | 測試集大小: {X_test_qs.shape[0]}")
    
    # --- 【超參數調優】: 定義參數網格 ---
    
    # 【修改 1】: 拿掉 tscv，我們將在 RandomizedSearchCV 中使用 cv=5
    
    # 定義要搜尋的參數範圍
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # --- 7. 調優並訓練模型 A：QS 機率 (分類) ---
    print("\n--- 7. 正在進行 XGBoost QS 分類模型超參數調優 ---")
    
    base_qs_model = XGBClassifier(
        objective='binary:logistic', 
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    qs_random_search = RandomizedSearchCV(
        estimator=base_qs_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,            # 【修改 1】: 使用標準的 5 折隨機交叉驗證
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    qs_random_search.fit(X_train_qs, y_qs_train)
    
    print(f"✅ QS 模型最佳參數: {qs_random_search.best_params_}")
    qs_model = qs_random_search.best_estimator_

    # --- 評估 QS 模型 ---
    print("\n--- QS 模型 (分類) 評估結果 [隨機切割, 已調優] ---")
    y_qs_pred = qs_model.predict(X_test_qs)
    y_qs_proba = qs_model.predict_proba(X_test_qs)[:, 1]

    print(f"✅ 準確度 (Accuracy): {accuracy_score(y_qs_test, y_qs_pred):.4f}")
    print("分類報告 (Classification Report):")
    print(classification_report(y_qs_test, y_qs_pred))

    # --- 8. 調優並訓練模型 B：失分 (迴歸) ---
    print("\n--- 8. 正在進行 XGBoost 失分 (ER) 迴歸模型超參數調優 ---")
    
    base_er_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    er_random_search = RandomizedSearchCV(
        estimator=base_er_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,           # 【修改 1】: 使用標準的 5 折隨機交叉驗證
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    er_random_search.fit(X_train_er, y_er_train)
    
    print(f"✅ ER 模型最佳參數: {er_random_search.best_params_}")
    er_model = er_random_search.best_estimator_
    
    # --- 評估 ER 模型 ---
    print("\n--- ER 模型 (迴歸) 評估結果 [隨機切割, 已調優] ---")
    y_er_pred = er_model.predict(X_test_er)
    
    mse = mean_squared_error(y_er_test, y_er_pred)
    r2 = r2_score(y_er_test, y_er_pred)
    
    print(f"✅ 均方根誤差 (RMSE): {np.sqrt(mse):.4f}")
    print(f"✅ R-squared (R2): {r2:.4f}")
    print(f"範例預測失分 (前 5 筆): {np.round(y_er_pred[:10], 2)}")
    print(f"實際失分 (前 5 筆): {y_er_test.iloc[:10].values}")
    
    return qs_model, er_model

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    
    # 1. 載入並準備數據
    X_features, y_qs, y_er = load_and_prepare_data(CONSOLIDATED_FILE)
    
    if X_features is not None:
        # 2. 訓練並評估模型
        qs_model, er_model = train_and_evaluate_models(X_features, y_qs, y_er)
        
        print("\n🎉🎉🎉 流程執行完畢！ 🎉🎉🎉")