# 需自行準備來源數據集資料夾、輸出的待分配資料夾

import os
import pandas as pd
import random
import shutil

source_data_folder = input("請輸入來源數據集資料夾路徑：")

# 檢查待分配資料夾是否存在
if not os.path.exists(source_data_folder):
    print(f"來源數據集資料夾路徑 '{source_data_folder}' 不存在。")
    os.system("pause")
    exit()

output_data_folder = input("請輸入待分配資料夾路徑：")

# 檢查待分配資料夾是否存在
if not os.path.exists(source_data_folder):
    print(f"待分配資料夾路徑 '{output_data_folder}' 不存在。")
    os.system("pause")
    exit()

data_count_input = input("請輸入資料筆數或百分比（例如，100 或 10%）：")

# 讀取來源數據集的 metadata.csv
source_metadata_file = os.path.join(source_data_folder, "metadata.csv")
source_metadata = pd.read_csv(source_metadata_file)

# 計算資料筆數
if data_count_input.endswith("%"):
    # 輸入是百分比，計算要抽取的資料筆數
    percentage = int(data_count_input.strip("%"))
    all_data_count = len(source_metadata)
    data_count = int(all_data_count * (percentage / 100))
else:
    # 輸入是具體的資料筆數
    data_count = int(data_count_input)
    all_data_count = data_count
print(f"資料總筆數 {all_data_count} 筆")

# 檢查並建立待分配資料夾下的 metadata.csv
output_metadata_file = os.path.join(output_data_folder, "metadata.csv")
if not os.path.exists(output_metadata_file):
    # 如果 metadata.csv 不存在，則建立一個空的 DataFrame 並儲存為 metadata.csv
    empty_metadata = pd.DataFrame(columns=source_metadata.columns)
    empty_metadata.to_csv(output_metadata_file, index=False)

# 隨機抽取資料
random_samples = source_metadata.sample(n=data_count, random_state=42)

# 將抽取到的資料移除
source_metadata.drop(random_samples.index, inplace=True)

# 建立一個空的 DataFrame 並儲存為 metadata.csv
empty_metadata = pd.DataFrame(columns=source_metadata.columns)
empty_metadata.to_csv(source_metadata_file, mode='w',
                      index=False)  # 寫入空資料覆蓋來源的資料檔

# 寫入抽取後剩餘的資料
source_metadata.to_csv(source_metadata_file, mode='a', header=False,
                       index=False)

# 將抽取到的資料寫入待分配資料夾的 metadata.csv
random_samples.to_csv(output_metadata_file, mode='a', header=False, index=False)

# 檢查資料描述的音訊檔案是否存在於同層資料夾下，並移動音訊檔案
for index, row in random_samples.iterrows():
    audio_filename = row["file_name"]
    source_audio_path = os.path.join(source_data_folder, audio_filename)
    output_audio_path = os.path.join(output_data_folder, audio_filename)

    if os.path.exists(source_audio_path):
        # 移動音訊檔案到待分配資料夾
        shutil.move(source_audio_path, output_audio_path)

# 檢查待分配資料夾和 metadata 包含不重複的音訊檔案
output_dir_files = set(os.listdir(output_data_folder))

output_metadata = pd.read_csv(output_metadata_file)
output_audio_files = set(output_metadata["file_name"])
output_symmetric_difference_set = output_dir_files ^ output_audio_files
if output_symmetric_difference_set:
    check_list = []
    for file in output_symmetric_difference_set:
        if file.endswith(".wav"):
            check_list.append(file)
    if check_list:
        print("待分配資料夾和 metadata 間，有'不重複'的音訊檔案:")
        for item in check_list:
            print(item)

# 檢查來源數據集資料夾和 metadata 包含不重複的音訊檔案
source_dir_files = set(os.listdir(source_data_folder))
source_audio_files = set(source_metadata["file_name"])
source_symmetric_difference_set = source_dir_files ^ source_audio_files
if source_symmetric_difference_set:
    check_list = []
    for file in source_symmetric_difference_set:
        if file.endswith(".wav"):
            check_list.append(file)
    if check_list:
        print("來源數據集資料夾和 metadata 間，有'不重複'的音訊檔案:")
        for item in check_list:
            print(item)

# 檢查待分配資料夾和來源數據集包含重複的音訊檔案
duplicate_dir_files = output_dir_files & source_dir_files
if duplicate_dir_files:
    check_list = [] 
    for file in duplicate_dir_files:
        if file.endswith(".wav"):
            check_list.append(file)
    if check_list:
        print("待分配資料夾和來源數據集資料夾包含重複的音訊檔案:")
        for item in check_list:
            print(item)
            
# 檢查兩 metadata 包含重複的音訊檔案
duplicate_metadata_files = output_audio_files & source_audio_files
if duplicate_metadata_files:
    print("待分配資料夾 metadata 和來源數據集 metadata 包含重複的音訊檔案:")
    for file in duplicate_metadata_files:
        print(file)

print("\n...檢查完成")       
print(f"完成 {data_count} 筆資料重分配任務。")
