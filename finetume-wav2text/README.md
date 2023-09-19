# 如何產生自己的音訊資料集
- [事前準備](https://github.com/Li732375/Finetune_Whisper_audio2zh/edit/master/finetume-wav2text/README.md#事前準備)
- [資料集(數據集)](https://github.com/Li732375/Finetune_Whisper_audio2zh/edit/master/finetume-wav2text/README.md#資料集(數據集))
- [資料檔撰寫指南](https://github.com/Li732375/Finetune_Whisper_audio2zh/edit/master/finetume-wav2text/README.md#資料檔撰寫指南)
- [輔助工具 (選讀)](https://github.com/Li732375/Finetune_Whisper_audio2zh/edit/master/finetume-wav2text/README.md#輔助工具 (選讀))
- [如何載入資料集](https://github.com/Li732375/Finetune_Whisper_audio2zh/edit/master/finetume-wav2text/README.md#如何載入資料集)

***

## 事前準備
+ #### 音訊檔
	+ 要作為訓練或測試資料的音訊檔。有幾點要注意到：
		+ ##### 副檔名
			+ 必須為模型支援的副檔名，**留意是否需要先行轉檔**。
			> + mp3
			> + mp4
			> + mpeg
			> + mpga
			> + m4a
			> + wav => 本專案採用
			> + webm
		+ ##### 取樣頻率
			+ 輸入模型的音訊檔普遍是** 16000 Hz**，若是不同就需要先行轉換。
			> 音訊檔滑鼠右鍵 > 內容，上排標籤選 "詳細內容"，在 "屬性"那欄，往下找到 "音訊" > "音訊取樣率"，單位是 "kHz"(1 kHz = 1000 Hz)。
 
+ #### 資料檔
	+ 紀錄該音訊檔路徑和對應的文字輸出內容。
* * *

## 資料集(數據集)
#### 完整的結構樹狀圖
```
dataset
    ├─ test
    │      ├─ metadata.csv
    │      ├─ 測試集音訊檔_4.wav
    │      ├─ 測試集音訊檔_5.wav
    │      └─ 測試集音訊檔_6.wav
    └─ train
           ├─ metadata.csv
           ├─ 訓練集音訊檔_1.wav
           ├─ 訓練集音訊檔_2.wav
           └─ 訓練集音訊檔_3.wav
```
> 註: 且同檔名的 metadata.csv 裡，盡管格式相同，但描述的資料不同

但...畢竟是從 0 開始，所以先做成這樣的結構，應該會比較好起手...
```
dataset
    ├─ test
    └─ train
           ├─ metadata.csv
           ├─ 訓練集音訊檔_1.wav
           ├─ 訓練集音訊檔_2.wav
           └─ 訓練集音訊檔_3.wav
```
+ dataset: 為資料夾，這個資料集名稱，名稱沒有限制，至少要看的出來找的到。
+ train & test: 為資料夾，名稱**至少分別包含關鍵字 "train"、"test"**，也可以是以此名稱延伸的檔名，載入時會自動辨別，否則會讀不到。淪為只讀取訓練集(這時就不會有命名要求)。

> 註: 有關音訊檔分配的部分，示範程式 "assort_data.py" 會協助隨機分配，使用限制後面會提及。

+ metadata.csv: 為 csv 檔，**檔案名稱只能叫 metadata**，土一點就直接用文字文件 txt 開啟即可。(嘀咕: 至少不用再安裝有的沒的) 要將模型輸入輸出的對應資料寫在這裡面。

> 註: 後面會示範撰寫格式。

+ 訓練集音訊檔: 要注意到的是，**若單一音訊檔的辨識結果可能不止一種時，在音訊檔路徑的描述上不可重複**，所以會複製一份音訊檔並更名(本專案就是從原檔名下去延伸)。

> 註: 示範程式 "update_data.py" 會協助完成比對與複製工作，使用限制後面會提及。

#### 看完後到這裡，就可以開始先動手建立初步的資料夾架構啦~~
* * *
#### 資料檔撰寫指南
#### 寫法格式
接著要闡述資料檔 metadata.csv 撰寫的留意事項。以本專案為例，僅需要輸出入兩個欄位就好。若是有因專案需要的追加其他欄位的，參考下方 "輸出" 的撰寫方式即可。
+ 輸入: 音訊檔路徑，**欄位名稱為 "file_name"，欄位名稱不可修改，後續程式載入時有套件會認這個名稱的欄位。**
+ 輸出: 辨識結果，欄位名稱為 "transcription"，欄位名稱可以更動，後續叫這個欄位時要認得出來。

> 因此，以文字文件 txt 開啟後，第一列寫的就是欄位名稱，格式如下
```
file_name,transcription
音訊檔路徑_1，辨識的文字內容_1
音訊檔路徑_2，辨識的文字內容_2
音訊檔路徑_3，辨識的文字內容_3
...往下以此類推...
```
預設以英文的半形逗點 "," 作間隔。

#### 細節留意
+ 若以文字文件 txt 編輯，編碼必須是 "UTF-8"，不可是 "ANSI"，載入時會報錯。
+ 每一筆資料 (每一排) 裡，是否有與預設分隔符號 (英文的半形逗點 ",") 重複的描述內容。**導致該筆資料的分段數被判定超過標頭欄位總數**，示範如下:
```
file_name,transcription
音訊檔路徑_1，辨識的文字內容_1
音訊檔路徑_2，辨識的,文字內容_2
音訊檔路徑_3，辨識的文字內容_3
...往下以此類推...
```
> 上述情形會出現報錯，並指出第 2 筆資料 (第 3 列) 有問題。
+ 不同筆資料 (每一排) 之間，是否有重複或非預期延伸格式的的音訊檔路徑。會導致僅一筆資料被讀取，其餘略過，形成資料量與音訊檔數量不符。

#### 看到這裡，開始著手製作資料檔吧 ! 畢竟校正資料的過程最費時了~~
* * *
## 輔助工具 (選讀)
輔助工具為自行撰寫，歡迎下載後依據自身需求修改調整程式碼，提升效率。
#### update_data.py
使用前提:
+ 自行準備撰寫完成的資料檔 (metadata.csv) 與音訊檔儲存的資料夾，並且通常資料檔 (metadata.csv) 就直接在資料夾下與音訊檔共存。

輸入資訊:
+ 資料檔 (metadata.csv) 路徑
+ 資料夾 (儲存音訊檔與資料檔的) 路徑

功能描述:
+ 針對延伸自特定音訊檔名的檔案，也就是當前不存在該路徑的資料，將原音訊檔複製，並於原來的名稱後面追加 "_" 與指定字元 (本例僅用數字)，例如:

> 資料檔描述

```
file_name,transcription
abc.wav，辨識的文字內容_1
abc_1.wav，辨識的文字內容_2  => 字尾有添加"_1"，為延伸檔名，但尚未存在該檔案
abcd.wav，辨識的文字內容_3
...往下以此類推...
```
> 結構樹狀圖 (處理前)

```
dataset
    ├─ test
    └─ train
           ├─ metadata.csv
           ├─ abc.wav
           └─ abcd.wav
```
***
> 程式處理後，**音訊檔 "abc.wav" 會被複製一份於同一個資料夾下，並重新命名為"abc_1.wav"**

***

> 結構樹狀圖 (處理後)

```
dataset
    ├─ test
    └─ train
           ├─ metadata.csv
           ├─ abc.wav
           ├─ abc_1.wav
           └─ abcd.wav
```
+ 檢查資料檔內是否有重複路徑並回報重複次數。

> 註: 名稱 "路徑" 重複出現 "數字" 次

+ 檢查資料格式是否有分段異常問題。

***

#### assort_data.py
使用前提:
+ 自行準備來源數據集資料夾 and 空的待分配資料夾。

輸入資訊:
+ 來源數據集資料夾路徑
+ 待分配資料夾路徑
+ 分配資料筆數或百分比（例如，100 或 10%）

功能描述:
+ 將來源數據集資料夾內的資料與檔案進行隨機重新分配，一部分維持原樣，另一部份則移動至待分配資料夾﹑提供訓練模型時，載入資料集前先行分配。

> 結構樹狀圖 (處理前)

```
dataset
    ├─ test
    └─ train
           ├─ metadata.csv
           ├─ abc.wav
           ├─ abc_1.wav
           ├─ abc_2.wav
           └─ abcd.wav
```
> 資料檔描述

```
file_name,transcription
abc.wav，辨識的文字內容_1
abc_1.wav，辨識的文字內容_2
abc_2.wav，辨識的文字內容_3
abcd.wav，辨識的文字內容_4
...往下以此類推...
```
***
> 程式處理後，**會依據指定需求隨機抽選，重新分配來源數據集下的資料至兩子集 (train and test)，並連帶移動相關音訊檔至對應的資料夾，同時更新兩邊的資料檔**

***

> 結構樹狀圖 (處理後)

```
dataset
    ├─ test
    │      ├─ metadata.csv
    │      ├─ abc_1.wav
    │      └─ abcd.wav
    └─ train
           ├─ metadata.csv
           ├─ abc.wav
           └─ abc_2.wav
```
> 子集 train 的資料檔描述

```
file_name,transcription
abc.wav，辨識的文字內容_1
abc_2.wav，辨識的文字內容_2
```
> 子集 test 的資料檔描述

```
file_name,transcription
abc_1.wav，辨識的文字內容_1
abcd.wav，辨識的文字內容_2
```
+ 檢查同子集下資料檔與音訊檔的對應，彼此是否有多餘的檔案
+ 檢查兩子集下資料檔，是否有重疊的資料
+ 檢查兩子集下音訊檔，是否有重疊的檔案

* * *
## 如何載入資料集

```
from datasets import load_dataset

# 載入資料集
dataset = load_dataset("audiofolder", data_dir="where_are_your_dataset", drop_labels=True)

# 印出來看看
print(f"\ndataset: \n {dataset}")

# 單一子資料集
print(f"\ndataset['train']:\n {dataset['train']}")
print(f"\ndataset['train'][0]:\n {dataset['train'][0]}")

# 另一子資料集
print(f"\ndataset['test']:\n {dataset['test']}")
print(f"\ndataset['test'][0]:\n {dataset['test'][0]}")
```
> 其中 "audiofolder" 不可異動，這樣才會自動分析音訊檔
> data_dir: 資料集路徑，
> drop_labels: 則是在載入後會自動汰掉追加欄位 "label"

參見 [Load audio data](https://huggingface.co/docs/datasets/audio_load#load-audio-data)

* * *
## 結語
最後叮嚀，資料總數問題要留意，若以文字文件 txt 開啟 csv 檔，總資料數為反白全部內容後，扣除標題欄 (-1) 和最後一筆資料的的換行符號  (-1)。要對應資料夾內，扣除資料檔 metadata.csv (-1) 後的數，若有不同則表示資料數間有差異。
> 總資料數 = 反白全部內容的總行數 -2 = 資料夾內檔案數 -1

* * *

#### 看到這裡，就完成了資料集從無到有的準備到匯入了，謝謝觀看~~

* * *

參考連結
+ 英，[Create an audio dataset](https://huggingface.co/docs/datasets/audio_dataset)
+ 英，[Fine-tuning the ASR model](https://huggingface.co/learn/audio-course/chapter5/fine-tuning#finetuning-the-asr-model)
+ 英，[Introduction to audio data](https://huggingface.co/learn/audio-course/chapter1/audio_data)
+ 英，[Load Dataset](https://huggingface.co/learn/audio-course/chapter5/fine-tuning#load-dataset)
+ 英，[Load audio data](https://huggingface.co/docs/datasets/audio_load#load-audio-data)
