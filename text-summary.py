import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("進入摘要階段")


# 設定文字檔目錄路徑
text_folder = "transcriptions_text"

# 載入摘要模型
model_name = "yihsuan/best_model_0427_small_long"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 初始化 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 建立儲存摘要的目錄
if not os.path.exists("summaries"):
    os.makedirs("summaries")

# 處理每個文字檔
for text_file in os.listdir(text_folder):
    if text_file.endswith(".txt"):
        text_path = os.path.join(text_folder, text_file)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            
            # 在日誌檔留下資料
            with open("log.txt", "a", encoding="utf-8") as log_file:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                log_file.write(f"完成作業時間: {current_time}, 讀入文字檔: {text_file} 進行摘要作業\n")
                print(f"完成作業時間: {current_time}, 讀入文字檔: {text_file} 進行摘要作業\n")
        
        # 文字處理
        paragraphs = [text.replace("\n", "")]  # 將文字串成一串
        ideal_length = 1000  # 設定文字長度

        # 切分成文字段
        paragraphs = [paragraphs[0][i: i+ ideal_length] for i in range(0, len(paragraphs[0]), ideal_length)]
        print("\n讀檔內容:\n", paragraphs)

        
        max_input_length = tokenizer.model_max_length - 2  # 預留 [CLS] 和 [SEP]
        summaries = []
        max_output_length = 100 #  模型最大輸出長度
        
        for paragraph in paragraphs:
            
            print("\nlen(paragraph):", len(paragraph),
                  "\nlen(tokenizer.tokenize(paragraph)):", len(tokenizer.tokenize(paragraph)),
                  "\nmax_input_length:", max_input_length)
            
            input_ids = tokenizer.encode(paragraph, return_tensors="pt"
                                         ).to(device)
                
            # input_ids: 這是輸入給模型的token ID序列。它表示你想要生成
            #           摘要的文本。通常，這是透過一個分詞器（tokenizer）
            #           將原始文本轉換成模型能理解的token序列。
            # max_length: 這是生成的文本的最大長度（token數）。模型生成的
            #           文本將不會超過這個長度。你可以根據你的需求設定此
            #           參數。
            # num_beams: 這是束搜索（beam search）的參數，用於控制生成時
            #           的多樣性。束搜索會同時保留幾個可能性，並選擇其中
            #           一個作為下一步的生成，以增加多樣性。num_beams 指
            #           定了保留的可能性數量。較大的值會增加多樣性，但也
            #           可能導致生成的結果較差。
            # early_stopping: 這是一個布林值，決定是否啟用提前停止機制。
            #           當模型生成的文本已經符合給定的 max_length 時，如
            #           果將此參數設為 True，則生成過程將停止。
            summary_ids = model.generate(input_ids,
                                         max_length=max_output_length,
                                         num_beams=2,
                                         early_stopping=True)[0]
            summary = tokenizer.decode(summary_ids,
                                       skip_special_tokens=True)
            summaries.append(summary)

            print("\n片段摘要內容:\n", summary)
            

        summary_content = "" #摘要內容
        
        # 將摘要寫入檔案
        summary_filename = os.path.splitext(text_file)[0] + "_summary.txt"
        with open(os.path.join("summaries", summary_filename),
                  "a", encoding="utf-8") as summary_file:

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            summary_file.write(f"{current_time}, 輸出摘要檔: {summary_filename}\n摘要內容:")
            
            for s in summaries:
                summary_file.write(s + "\n")
                summary_content += s

            summary_file.write("\n")

        # 印出影片標題、時間、摘要模型輸出內容
        print("\n摘要內容:\n", summary_content)
        
        # 在日誌檔留下資料
        with open("log.txt", "a", encoding="utf-8") as log_file:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            log_file.write(f"完成作業時間: {current_time}, 輸出摘要檔: {summary_filename}\n")
            print(f"\n完成作業時間: {current_time}, 輸出摘要檔: {summary_filename}\n")
        
        ##os.remove(text_path) # 移除語音辨識內容文字檔(.txt)
        
    else:
        with open("log.txt", "a", encoding="utf-8") as log_file:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            log_file.write(f"完成作業時間: {current_time}, 未找到待摘要的文字檔(.txt)\n")
            print("未找到待摘要的文字檔(.txt)")



# 取得目前執行的腳本的檔案路徑
current_file_path = os.path.abspath(__file__)

# 從檔案路徑中提取出檔案名稱
file_name = os.path.basename(current_file_path)

# 將操作過程寫入紀錄檔
with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"完成作業時間: {current_time}, {file_name} 完成所有步驟\n")
    print(f"完成作業時間: {current_time}, {file_name} 完成所有步驟")
    
