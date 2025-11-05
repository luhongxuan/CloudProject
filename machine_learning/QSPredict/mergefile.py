import pandas as pd
import os
import glob
from typing import List

# --- 設定 ---
# 1. 指定您的檔案路徑模式
# 假設所有檔案都在當前目錄，且名稱都符合 "PitcherName_Year_features.xlsx" 格式
# 如果檔案在子資料夾，請更改路徑，例如：'./data/*.xlsx'
file_pattern = r'C:\loginonly\machine_learning\pitcher_record\all_pitchers_2024\*_features.xlsx'

# 2. 指定最終輸出的檔案名稱
output_filename = r'C:\loginonly\machine_learning\pitcher_record\all_pitchers_2024\All_Pitchers_2024_Consolidated.xlsx'

# 3. 指定您的資料夾路徑 (如果不在當前目錄)
# directory_path = 'C:/Users/YourName/Documents/PitchingData/'
# file_pattern = os.path.join(directory_path, '*_features.xlsx')


def consolidate_excel_files(pattern: str, output_file: str) -> pd.DataFrame:
    """
    讀取符合模式的所有 Excel 檔案，將它們合併成一個 DataFrame，並存儲為新的 Excel 檔案。
    """
    print(f"--- 開始處理符合模式 '{pattern}' 的檔案 ---")
    
    # 獲取所有符合模式的檔案路徑
    all_files: List[str] = glob.glob(pattern)
    
    if not all_files:
        print("❌ 錯誤：找不到任何符合指定模式的檔案。請檢查檔案路徑和模式。")
        return pd.DataFrame()

    print(f"✅ 找到 {len(all_files)} 個檔案，開始讀取與合併...")

    # 用於儲存所有讀取到的 DataFrame
    list_of_dfs = []
    
    # 逐一讀取並處理每個檔案
    for filename in all_files:
        try:
            # 讀取 Excel 檔案。假設您的數據在第一個工作表 (sheet_name=0)
            df = pd.read_excel(filename)
            
            # 選擇性：新增一個欄位來記錄數據來源是哪個檔案/投手，方便日後追蹤
            # 我們可以從檔名提取投手的名字
            # 檔名範例: Aaron_Brooks_2024_features.xlsx
            pitcher_name = filename.split('_2024')[0].replace('_', ' ')
            df['Pitcher_Source'] = pitcher_name
            
            list_of_dfs.append(df)
            
        except Exception as e:
            print(f"⚠️ 讀取檔案 {filename} 時發生錯誤: {e}")
            continue

    if not list_of_dfs:
        print("❌ 錯誤：所有檔案讀取失敗，無法進行合併。")
        return pd.DataFrame()

    # 使用 pandas.concat() 將所有 DataFrame 垂直堆疊合併
    # ignore_index=True 確保合併後的索引是連續的新數字
    final_df = pd.concat(list_of_dfs, ignore_index=True)
    
    # --- 儲存結果 ---
    try:
        final_df.to_excel(output_file, index=False)
        print(f"\n🎉 合併完成！")
        print(f"總共 {len(final_df)} 筆記錄已儲存到 '{output_file}'")
    except Exception as e:
        print(f"❌ 儲存檔案 {output_file} 時發生錯誤: {e}")
        
    return final_df

# 執行函式
consolidated_data = consolidate_excel_files(file_pattern, output_filename)