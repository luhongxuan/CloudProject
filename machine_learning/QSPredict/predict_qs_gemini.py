import pandas as pd
import numpy as np
import joblib
import os
# 【新增】: 導入 sklearn 的評估工具
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# --- 1. 載入儲存的模型和欄位 ---
print("--- 1. 正在載入模型和訓練欄位 ---")
try:
    qs_model = joblib.load('qs_model.joblib')
    er_model = joblib.load('er_model.joblib')
    training_columns = joblib.load('training_columns.pkl')
    print("✅ 模型載入成功！")
except FileNotFoundError:
    print("❌ 錯誤：找不到模型檔案 (qs_model.joblib, er_model.joblib, training_columns.pkl)")
    print("請先執行訓練腳本來產生模型檔案。")
    exit()

# --- 2. 準備 2025 年的新數據 ---

def get_2025_data():
    """
    【重要】：這個範例現在假設您的 2025 數據包含【特徵】和【實際結果】。
    """
    
    print("--- 2. (範例) 正在載入 2025 年的【完整】比賽數據 ---")
    data_2025 = {
        # --- 特徵 (Features) ---
        'pitcher': ['Gerrit Cole', 'Aaron Nola', 'New Pitcher'],
        'opp_team': ['BOS', 'NYM', 'HOU'],
        'Team': ['NYY', 'PHI', 'ATL'],
        'hand': ['R', 'R', 'L'],
        'rest_days': [5, 4, 6],
        'opp_ops': [0.750, 0.710, 0.801],
        'is_home': [1, 0, 1],
        'avg_ip_last3': [6.5, 5.8, 0.0],
        'avg_er_last3': [2.1, 3.3, 0.0],
        'season_era': [3.10, 3.45, 0.0],
        'season_whip': [1.05, 1.12, 0.0],
        
        # --- 【新增】: 實際比賽結果 (Ground Truth) ---
        # 這些欄位將被用來評估準確率
        'IP': [7.0, 5.0, 6.0],
        'ER': [2, 4, 1]
    }
    df_2025 = pd.DataFrame(data_2025)
    
    df_2025.to_excel('2025_data_example.xlsx', index=False)
    
    return '2025_data_example.xlsx' # 返回檔案路徑

def process_new_data(filepath: str, expected_columns: list) -> tuple:
    """
    載入新數據，分離特徵(X)和實際結果(Y)，並將特徵處理成和訓練時一模一樣的格式。
    """
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"❌ 讀取新數據時發生錯誤: {e}")
        return None, None, None

    print(f"--- 3. 正在處理 {len(df)} 筆新數據 ---")
    
    # --- 【新增】步驟 A: 提取實際結果 (Ground Truth) ---
    # 檢查 IP 和 ER 是否存在，如果不存在則無法評估
    if 'IP' not in df.columns or 'ER' not in df.columns:
        print("⚠️ 警告：新數據中缺少 'IP' 或 'ER' 欄位，將僅進行預測，無法評估準確率。")
        y_true_qs = None
        y_true_er = None
    else:
        # 根據 IP 和 ER 建立 QS 的實際結果
        y_true_qs = ((df['IP'] >= 6) & (df['ER'] <= 3)).astype(int)
        y_true_er = df['ER']
    
    # --- 步驟 B: 執行和訓練時相同的基本清理 ---
    numeric_cols = [
        'rest_days', 'opp_ops', 'is_home', 'avg_ip_last3', 
        'avg_er_last3', 'season_era', 'season_whip'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0) 

    # --- 步驟 C: 執行和訓練時相同的 One-Hot Encoding ---
    categorical_cols = ['pitcher', 'opp_team', 'Team', 'hand']
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_categorical, drop_first=True)

    # --- 步驟 D: 【最關鍵】: 將欄位與訓練時對齊 ---
    # 這樣會自動移除 IP, ER (因為它們不在 expected_columns 中)
    # 並新增/移除 One-Hot 編碼的欄位以匹配訓練時的格式
    df_aligned = df_encoded.reindex(columns=expected_columns, fill_value=0)
    
    print("✅ 新數據已對齊訓練格式！")
    
    # 返回對齊後的特徵(X)和實際結果(Y)
    return df_aligned, y_true_qs, y_true_er

# --- 3. 執行預測流程 ---
if __name__ == "__main__":
    
    # 1. 獲取 2025 年數據的檔案路徑
    new_data_filepath = get_2025_data()
    
    # 2. 處理 2025 年數據，使其符合模型輸入要求
    # 【修改】: 現在會同時返回 X 和 Y
    X_2025, y_true_qs, y_true_er = process_new_data(new_data_filepath, training_columns)
    
    if not X_2025.empty:
        # 3. 進行預測
        print("\n--- 4. 正在進行預測 ---")
        
        # 預測 QS
        qs_probabilities = qs_model.predict_proba(X_2025)[:, 1] # 機率
        qs_predictions_binary = (qs_probabilities > 0.5).astype(int) # 二元預測 (0或1)
        
        # 預測失分 (ER)
        er_predictions = er_model.predict(X_2025)
        
        # 4. 整理並顯示結果
        df_original = pd.read_excel(new_data_filepath)
        df_original['Predicted_QS_Probability'] = np.round(qs_probabilities, 4)
        df_original['Predicted_QS_Binary'] = qs_predictions_binary
        df_original['Predicted_ER'] = np.round(er_predictions, 2)
        
        print("\n--- 預測結果 ---")
        print(df_original[['pitcher', 'opp_team', 'Predicted_QS_Probability', 'Predicted_ER']])
        
        # --- 5. 【新增】: 評估 2025 年的預測準確率 ---
        if y_true_qs is not None and y_true_er is not None:
            print("\n--- 2025 數據預測準確率評估 ---")
            
            # 評估 QS (分類)
            qs_acc = accuracy_score(y_true_qs, qs_predictions_binary)
            print(f"✅ QS (分類) 準確度 (Accuracy): {qs_acc:.4f}")
            print("QS (分類) 報告:")
            print(classification_report(y_true_qs, qs_predictions_binary))

            # 評估 ER (迴歸)
            er_r2 = r2_score(y_true_er, er_predictions)
            er_rmse = np.sqrt(mean_squared_error(y_true_er, er_predictions))
            
            print(f"✅ ER (迴歸) R-squared (R2): {er_r2:.4f}")
            print(f"✅ ER (迴歸) 均方根誤差 (RMSE): {er_rmse:.4f}")

            # 顯示詳細比較
            print("\n--- ER 詳細比較 (實際 vs 預測) ---")
            comparison_df = df_original[['pitcher', 'ER', 'Predicted_ER']]
            comparison_df.rename(columns={'ER': 'Actual_ER'}, inplace=True)
            print(comparison_df)
        
        else:
            print("\n--- 預測完成（未評估準確率，因為缺少 IP/ER 實際數據） ---")