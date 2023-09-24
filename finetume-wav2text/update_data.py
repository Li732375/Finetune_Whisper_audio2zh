# 需自行準備撰寫完成的資料檔 (metadata.csv) 與音訊檔儲存的資料夾，並且通常資料檔 (metadata.csv) 就直接在資料夾下與音訊檔共存。

import os
import shutil
from collections import Counter

print("註:資料檔路徑內容會有像是如下格式描述的 txt 檔")
print(f"'''\n影片id_影片起始秒數1_影片結束秒數1.wav,描述內容1\n影片id_影片起始秒數2_影片結束秒數2.wav,描述內容2\n'''")
data_file_path = input("請輸入資料檔路徑：")

# 確認資料檔路徑存在
if not os.path.exists(data_file_path):
    print(f"資料檔路徑 '{data_file_path}' 不存在。")
    os.system("pause")
    exit()
    
output_folder_path = input("請輸入音訊檔儲存的資料夾路徑：")

# 確認音訊檔儲存的資料夾路徑存在
if not os.path.exists(output_folder_path):
    print(f"音訊檔儲存的資料夾 '{output_folder_path} '不存在。")
    os.system("pause")
    exit()
    

# 定義函數，用於處理每一筆資料
def process_data(data, other):
    # 以逗號分隔資料
    parts = data.split(',')
    if len(parts) != 2:
        other.append(f"{data} :','標示問題，超過兩段")
        return ""

    # 取得左半段
    left_half = parts[0]

    if "file_name" == left_half:
        return ""

    # 以底線分隔左半段，至少會有三段
    segments = left_half.split('_')
    if len(segments) < 3:
        other.append(f"{data} :'_'標示問題，小於三段")
        return ""

    # 取得前三段的資訊並去除空格
    audio_name = '_'.join(segments[:3]).strip()

    # 如果有第四段，則尋找對應的音訊檔
    if len(segments) == 4:
        #print(audio_name) 
        audio_file = os.path.join(output_folder_path, f"{audio_name}.wav")

        if os.path.exists(audio_file):
            # 建立新的音訊檔名稱
            new_audio_name = f"{audio_name}_{segments[3]}"
            new_audio_path = os.path.join(output_folder_path, new_audio_name)

            # 若新音訊檔不存在則建立一份
            if not os.path.exists(new_audio_path):
                # 複製並重新命名音訊檔
                shutil.copy(audio_file, new_audio_path)
                print(f"已複製並重新命名：{new_audio_path}")

            #print(new_audio_name)        
            return new_audio_name

        else:
            other.append(f"{audio_file} :檔案未找到")
            return ""
        
    else:
        #print(audio_name)
        return audio_name


# 讀取包含多筆資料的txt檔案
with open(data_file_path, 'r', encoding='utf-8') as file:
    data_lines = file.readlines()
    print("資料檔讀取完畢，開始進行處理...")

duplicate = []  # 存放重複的檔名
previous = ""  # 檔名
other = []  # 紀錄有問題的資料

# 逐筆處理資料
for line in data_lines:
    previous = process_data(line.strip(), other)

    if len(previous) > 0:
        duplicate.append(previous)

# 確認是否有重複內容
if duplicate:
    print()
    # 統計數量
    element_count = Counter(duplicate)
    
    for element, count in element_count.items():
        if count > 1:
            print(f"名稱 {element} 重複出現 {count} 次")
else:
    print("\n未發現重複音訊檔檔名的資料")

# 確認是否有異常資料
if other:
    print(f"\n資料異常:")

    for num in range(len(other)):
        print(f"{num + 1} >>> {other[num]}")
    
print("\n完成音訊檔資料更新任務。")
