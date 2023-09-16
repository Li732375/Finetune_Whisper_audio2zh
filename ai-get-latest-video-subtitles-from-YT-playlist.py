import os
import ffmpeg
from pytube import Playlist
from datetime import datetime
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pydub import AudioSegment


# 設定影片清單的連結
playlist_links = ["your_youtube_playlist_link"]

# 下載存放資料夾
downloaded_videos_path = "downloaded_videos"
                   

# 逐一處理每個清單的影片
for playlist_link in playlist_links:
    playlist = Playlist(playlist_link)

    try:
        orders = [0]  # 指定順位
          
        for order in orders:
            
            video = playlist.videos[order]  # 取得所有影片的" YouTube 物件"
            video_id = video.video_id  # 取得影片 id
            video_title = video.title  # 取得影片標題

            # 儲存影片音訊檔的目錄
            if not os.path.exists(downloaded_videos_path):  # 若不存在就建立
                os.makedirs(downloaded_videos_path)
                print(f"[log] 未找到存放目錄 {downloaded_videos_path}，將自動建立")
            
            # 列出資料夾下的所有 mp4
            submp4s = [os.path.basename(os.path.splitext(f.path)[0]) for f in os.scandir(downloaded_videos_path) if f.is_file()]
    
            if not video_id in submp4s:                
                print(f"[log] 未找到音訊檔 {video_id}.mp4 ...將從線上下載...")

                # 下載影片音訊檔
                audio_stream = video.streams.filter(only_audio=True).first()
                audio_stream.download(output_path=downloaded_videos_path,
                                      filename=video_id + ".mp4")

                # 獲取影片的更新時間和標題
                update_time = video.publish_date
                print(f"[log] 影片標題: {video_title}")

                # 在日誌檔留下資料
                with open("log.txt", "a", encoding="utf-8") as log_file:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                    log_file.write(f"[log] 下載影片時間: {current_time}, 影片更新時間: {update_time}, 影片id: {video_id}, 路徑: {os.path.join(downloaded_videos_path, video_id)}.mp4")
                    print(f"[log] 下載影片時間: {current_time}, 影片更新時間: {update_time}, 影片id: {video_id}, 路徑: {os.path.join(downloaded_videos_path, video_id)}.mp4")
            else:
                print("[log] 已發現過去下載的音訊檔 {video_id}.mp4...")

    except Exception as e:
        with open("log.txt", "a", encoding="utf-8") as log_file:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            log_file.write(f"[error] 下載影片時間: {current_time}, 影片id: {video_id}, 錯誤訊息: {str(e)}")
            print(f"[error] 下載影片時間: {current_time}, 影片id: {video_id}, 錯誤訊息: {str(e)}")
        continue


# 設定音訊檔目錄路徑
audio_folder = downloaded_videos_path

# 建立儲存文字檔的目錄
transcriptions_path = "transcriptions_text"
if not os.path.exists("transcriptions_path"):
    os.makedirs("transcriptions_path")
    print(f"[log] 未找到存放目錄 {transcriptions_path}，將自動建立")
            
# 從 huggingface 下載模型的 cmd 指令
# transformers-cli download model_name

# 載入模型和轉換器
model_name = "ADT109119/whisper-small-zh-TW"  # 繁體內容語音轉文字模型
        
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[log] 模型運算裝置：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 將模型和處理器移至GPU
model = model.to(device)

target_sampling_rate = 16000  # 設定新取樣率


for audio_file in os.listdir(audio_folder):
    if audio_file.endswith(".mp4"):

        # 列出資料夾下的所有子資料夾
        subfolders = [os.path.splitext(os.path.basename(f.path))[0] for f in os.scandir(downloaded_videos_path) if f.is_dir()]

        if not os.path.splitext(audio_file)[0] in subfolders:

            # 設定目標音訊檔案路徑
            audio_path = os.path.join(audio_folder, audio_file)
            print("\n[log] 找到 mp4 檔案：", os.path.basename(audio_path),
                  ", 路徑:", audio_path)

            # 設定轉換後的音訊檔案路徑
            video_id = os.path.splitext(audio_file)[0]
            converted_audio = video_id + ".wav"
            converted_audio_path = os.path.join(audio_folder, converted_audio)
            
            # 使用 ffmpeg-python 執行音訊轉換
            input_stream = ffmpeg.input(audio_path)
            output_stream = ffmpeg.output(input_stream, converted_audio_path,
                                          format='wav', y=True)
            ffmpeg.run(output_stream)

            # 在日誌檔留下資料
            with open("log.txt", "a", encoding="utf-8") as log_file:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                log_file.write(f"[log] 完成作業時間: {current_time}, 影片id: {video_id} 已轉檔, 路徑: {converted_audio_path}")
                print(f"[log] 完成作業時間: {current_time}, 影片id: {video_id} 已轉檔, 路徑: {converted_audio_path}")

            print("[log] 已轉檔:", os.path.basename(converted_audio_path),
                  ", 路徑:", converted_audio_path)
            
            os.remove("True")  # 移除不明檔案"True"
  

            print("[log] 進行取樣率調整階段")
            
            # 讀取原始音訊檔案
            waveform, original_sampling_rate = torchaudio.load(converted_audio_path)

            # 進行重新取樣
            resampler = torchaudio.transforms.Resample(
                original_sampling_rate, target_sampling_rate)
            resampled_waveform = resampler(waveform)

            # 儲存重新取樣後的音訊
            torchaudio.save(converted_audio_path, resampled_waveform,
                            target_sampling_rate)

            # 在日誌檔留下資料
            with open("log.txt", "a", encoding="utf-8") as log_file:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                log_file.write(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已重新取樣音訊, 存放路徑: {converted_audio_path}")
                print(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已重新取樣音訊, 存放路徑: {converted_audio_path}")

            print("[log] 已重新取樣音訊並儲存至:", converted_audio_path)

            
            print("[log] 依特定大小開始切分音訊檔")
            
            # 創建存放子音訊檔案的資料夾
            os.makedirs(os.path.join(audio_folder, os.path.splitext(audio_file)[0]),
                        exist_ok=True)

            # 設定參數(毫秒，1秒等於1000毫秒)
            chunk_long = 10000  # 片段時長
            overlap_size = 2000  # 相鄰片段的重疊時長

            # 進行音訊檔分割
            audio = AudioSegment.from_wav(converted_audio_path)
            total_duration = len(audio)
            current_position = 0

            count = 0  # 統計分割數
            
            while current_position < total_duration:
                start_time = max(current_position, 0)
                end_time = min(current_position + chunk_long, total_duration)
                
                chunk = audio[start_time:end_time]
                output_path = f"{audio_folder}/{os.path.splitext(audio_file)[0]}/{os.path.splitext(audio_file)[0]}_{start_time//1000}_{end_time//1000}.wav"
                chunk.export(output_path, format="wav")
                
                current_position += (chunk_long - overlap_size)
                count += 1

            # 在日誌檔留下資料
            with open("log.txt", "a", encoding="utf-8") as log_file:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
                log_file.write(f"[log] 子音訊時長(秒): {chunk_long / 1000}, 重疊時長(秒): {1000 / 1000}, 取樣頻率: {target_sampling_rate}, 影片id: {video_id}")
                log_file.write(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已切出 {count} 個子音訊檔, 路徑: {output_path}")
                print(f"[log] 子音訊時長: {chunk_long / 1000}, 重疊時長: {1000 / 1000}, 取樣頻率: {target_sampling_rate}, 影片id: {video_id}")
                print(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已切出 {count} 個子音訊檔, 路徑: {output_path}")

            print("[log] 完成音訊檔分段作業，共切出 ", count, " 個子音訊檔")
            
            os.remove(converted_audio_path)  # 移除轉換的音訊檔(.wav)

            
            print("\n[log] 進行語音辨識階段")

            transcriptions = [""] * count  #  存放子音訊辨識結果
            
            for sub_audio in os.listdir(os.path.join(audio_folder,
                                                     os.path.splitext(
                                                         audio_file)[0])):
                if sub_audio.endswith(".wav"):

                    # 設定目標音訊檔案路徑
                    sub_audio = os.path.join(os.path.join(audio_folder,
                                                          os.path.splitext(
                                                              audio_file)[0],
                                                          sub_audio))
                    ##print("\n找到音訊檔:", os.path.basename(sub_audio), ", 路徑:", sub_audio)
                    
                    # 讀取音訊檔案
                    waveform, sampling_rate = torchaudio.load(sub_audio)
                    
                    # 使用處理器將音訊轉換為模型的輸入特徵
                    input_features = processor(waveform.numpy(),
                                                sampling_rate=sampling_rate,
                                                return_tensors="pt"
                                                ).input_features.to(device)
                    
                    # 使用模型進行辨識
                    with torch.no_grad():
                        predicted_ids = model.generate(input_features)

                    # 使用處理器將模型輸出的 token ids 轉換為文字
                    text = processor.batch_decode(predicted_ids,
                                                  skip_special_tokens=True)

                    index = int(os.path.splitext(os.path.basename(
                        sub_audio))[0].split("_")[1]) // ((chunk_long - overlap_size) // 1000)


                    text_list_filename = os.path.splitext(
                        audio_file)[0] + "_list.txt"  # 設定 list 文字檔案名稱
                    
                    # list 文字檔案路徑
                    save_list_path = os.path.join("transcriptions_text",
                                            text_list_filename)
                            
                    # 儲存 list 成文字檔案
                    with open(save_list_path, "a", encoding="utf-8") as text_file:           
                        """
                        #os.path.join 的首個參數可以設定欲存放的資料夾名稱
                        text_file.write(os.path.join("", os.path.basename(sub_audio))
                                        + "_" + text[0] + '\n')  # 加上換行符號
                        """
                        text_file.write(os.path.basename(sub_audio)
                                        + "_" + text[0] + '\n')  # 加上換行符號
                    
                    # 將歷次辨識結果加入結果列表
                    transcriptions[index] = text[0]
                    
                    # 輸出辨識結果
                    ##print("[log] 第 ", index, " 段", ", 辨識結果:\n", transcriptions[index]) # 因為雖然兩次的向量有細微差異，但不影響辨識結果


    """
    # 移除存放子音訊的目錄
    for item in os.listdir(os.path.join(audio_folder, video_id)):
        item_path = os.path.join(os.path.join(audio_folder, video_id), item)
                        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            remove_folder(item_path)
                            
    os.rmdir(os.path.join(audio_folder, video_id))
                    

    # 在日誌檔留下資料
    with open("log.txt", "a", encoding="utf-8") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        log_file.write(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已移除子音訊目錄")
        print(f"\n[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已移除子音訊目錄")
    """ 
                                    
    # 在日誌檔留下資料
    with open("log.txt", "a", encoding="utf-8") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        log_file.write(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已存進文字檔路徑: {save_list_path}\n")
        print(f"[log] 完成作業時間: {current_time}, 影片id: {video_id}, 已存進文字檔路徑: {save_list_path}\n")
                        
# 將操作過程寫入紀錄檔
with open("log.txt", "a", encoding="utf-8") as log_file:

    # 取得目前執行的腳本的檔案路徑
    current_file_path = os.path.abspath(__file__)

    # 從檔案路徑中提取出檔案名稱
    file_name = os.path.basename(current_file_path)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    
    log_file.write(f"[log] 完成作業時間: {current_time}, {file_name} 完成所有步驟, 檔案路徑: {current_file_path}\n")
    print(f"[log] 完成作業時間: {current_time}, {file_name} 完成所有步驟, 檔案路徑: {current_file_path}\n")
