# 微調 (Finetune) Whisper 模型
## 基本資訊
+ 模型
	+ 來源: https://huggingface.co/ADT109119/whisper-small-zh-TW
	+ 功能: 音訊轉繁體中文
+ 訓練資料: 
	+ 頻道: [千千進食中](https://www.youtube.com/@Chienseating)
	+ 影片清單: [水水來嘗鮮](https://www.youtube.com/playlist?list=PLWbKW1MoBjKLA2PuOwHL9AgO4yA-KIgRW)
	+ 訓練資料影片連結: [【千千進食中】饗饗全品項攻略！高樓景觀吃到飽！怎麼吃比較好？！](https://www.youtube.com/watch?v=HvugeIKJ9ok&list=PLWbKW1MoBjKLA2PuOwHL9AgO4yA-KIgRW&index=2&ab_channel=%E5%8D%83%E5%8D%83%E9%80%B2%E9%A3%9F%E4%B8%AD)
+ 顯卡: NVIDIA GeForce RTX 3070 Laptop GPU
```
NVIDIA-SMI 537.13
Driver Version: 537.13
CUDA Version: 12.2
```
+ 環境編輯器: Python IDLE 3.10.11 (is Python's Integrated Development and Learning Environment)
***

## 大綱
* #### 前置準備
* #### 下載模型
* #### 資料下載與預處理
* #### 訓練模型
***

## 前置準備

* #### 套件安裝
對於初來乍到的讀者，為了維持你想實作的**耐心籌碼**，將相關的套件先行安裝好，可以減少很多不必要的衍伸報錯，造成籌碼提前耗損。
> 出現難以解開的問題時，也可以早點放棄 ? (誤

使用 pip 安裝這些套件，或者直接將 requirements.txt 文件打開直接逐一套件複製出來，再自行安裝也行，也可以照如下步驟一次安裝：
1. 打開命令提示字元。
***
2. 使用 cd 命令來切換目錄，到存放 requirements.txt 文件的目錄。
3. 在目錄中執行以下命令安裝套件：
```
pip install -r requirements.txt
```
或者

2. 複製 requirements.txt 文件的路徑。
3. 在目錄中執行以下命令安裝套件：
```
pip install -r 文件的路徑
```

***
* #### CUDA
依據現在當下的時期，尚未有該硬體配置的相應 cuda 穩定的版本。作者當時臆測暨實際測試:

* * 是否有向前版本相容 ? => 不
* * 是否是下載的 Nvidia CUDA toolkit 的哪個元件漏裝 ? => 不
* * 是否是下載的 Nvidia CUDA toolkit 的元件全裝 ? => 不
* * 是否是下載的 Nvidia CUDA toolkit ，要用自訂義安裝 ? => 不
* * cudnn 沒安裝 ? => 不

... ...等等諸多問題

最後又 ~~ 回再度到 [PyTorch 官網](https://pytorch.org/get-started/locally/#start-locally)，如下點選:

```
PyTorch Build => Preview (Nightly)
Your OS => Windows
Package => Pip
Language => Python
Compute Platform => CUDA 12.1
```
得到的 Run this Command
> pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

**嘀咕: 前前後後拆裝 CUDA toolkit 和 cudnn 數次，結果最後根本完全沒安裝...**

***
* #### FFmpeg
一個免費的開源軟體庫，處理和操作音訊和影片。常被廣泛用於各種目的，包括轉碼、基本編輯以及過濾和 Stream 傳輸等任務。

1. 官網下載: [Download FFmpeg](https://ffmpeg.org/download.html)
2. 解壓縮檔案，放置的位置不能太隨便，要穩定且不會被"莫名其妙搬走"的
3. **設定環境變數**
4. 測試，在命令提示字元輸入
> ffmpeg

會洋洋灑灑的出現說明，反正沒報錯就行
***

## 下載模型
模型皆取自 [Hugging Face](https://huggingface.co/)，各式各樣各種類的模型，只要你電腦 hold 的住、裝得下 (模型除非有出較小的"尺寸"，他們通常有別稱和延伸命名，否則大部分幾 **"G"** 跑不掉，這還沒有考慮進開始使用或訓練時需要的記憶體)，花點時間下載模型下來體驗也會有不錯的探索體驗。

另外，若是真的跟我用相同的環境編輯器，別依賴程式裡引入套件直接下載，作者個人經驗，那個速度真~~的**堪憂**到懷疑人生。
> 作者有留意到，下載速度似乎會**隨著完成下載量增加而降速**，原因不明

那怎麼辦 ? 對，自行先透過其他管道下載，像是直接在命令提示字元輸入:
> transformers-cli download your_model_name

其中，your_model_name 就是在頁面裡，名稱旁有個明顯的複製按鈕，按下後就得到的複製內容。至於下載的東西到哪去了呢 ? 
> 通常位於使用者主目錄下的 .cache 文件夾中，具體路徑可能是 ~/.cache/huggingface/hub（Linux和macOS）
或
C:\Users\YourUsername\.cache\huggingface\hub（Windows）。

裡面有你曾經下載的各式模型，名稱仿複製內容。
***

## 資料下載與預處理


***

## 訓練模型


***

## 結語


***

參考連結
* 英，[]()

