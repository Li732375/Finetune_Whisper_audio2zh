# 訓練結果
## 為何這裡是空的?
因為訓練完的模型檔案過大，足足超過 2GB，加上單檔上限 100 MB，故改置放其他連結。

## 用 ssh 上傳？
### 前置設定
倘若依照當時線上主流 ssh key 設定教學，依序輸入指令，會發現**在目標位置 (C:\Users\使用者名稱\.ssh) 上，根本就沒有產生所謂的什麼 id_rsa、XXXXX.pub 檔**，頂多一個些微內容的 _host 檔。

### 那怎麼辦？
使用產生金鑰的指令 (ssh-keygen) 之前，先在目標位置新增一個空白文字文件檔，命名 id_rsa 並且**沒有副檔名**。然後再繼續依照教學指示進行 (過程應該會詢問是否要覆寫？，答**是**)，才會相對產生公鑰的 .pub 檔。

### 後來還是失敗？
是的，並且相對於原先的 https 傳輸，ssh 更快跳出傳輸失敗，內容如下:
```
C:\Users\XXXXXXXX\Desktop\...\Finetune_Whisper_audio2zh>git push origin-ssh XXXX:master
Enumerating objects: 31, done.
Counting objects: 100% (31/31), done.
Delta compression using up to 8 threads
Compressing objects: 100% (28/28), done.
client_loop: send disconnect: Connection reset by peer
fatal: sha1 file '<stdout>' write error: Broken pipe
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly

# origin-ssh: 路徑別名。查看指令 git remote -v 
```

## 又回到 https
傳輸失敗的問題還是回到原點，內容如下:
```
...第一種...

C:\Users\XXXXXXXX\Desktop\...\Finetune_Whisper_audio2zh>git push origin XXXX:master
Enumerating objects: 31, done.
Counting objects: 100% (31/31), done.
Delta compression using up to 8 threads
Compressing objects: 100% (28/28), done.
error: RPC failed; curl 6 OpenSSL SSL_read: Connection was reset, errno 10054
send-pack: unexpected disconnect while reading sideband packet
Writing objects: 100% (28/28), 2.02 GiB | 1.68 MiB/s, done.
Total 28 (delta 2), reused 0 (delta 0), pack-reused 0
fatal: the remote end hung up unexpectedly
Everything up-to-date


...第二種...

C:\Users\XXXXXXXX\Desktop\...\Finetune_Whisper_audio2zh>git push origin XXXX:master
Enumerating objects: 31, done.
Counting objects: 100% (31/31), done.
Delta compression using up to 8 threads
Compressing objects: 100% (28/28), done.
error: RPC failed; HTTP 408 curl 22 The requested URL returned error: 408
send-pack: unexpected disconnect while reading sideband packet
Writing objects: 100% (28/28), 2.02 GiB | 4.77 MiB/s, done.
Total 28 (delta 2), reused 0 (delta 0), pack-reused 0
fatal: the remote end hung up unexpectedly
Everything up-to-date


# origin: 路徑別名。查看指令 git remote -v 
```

## 最終方案
改至於 google 雲端[同名專案](https://drive.google.com/drive/folders/1-_27XKzwv20TLnnclIQ3ngqRRAe5N5vV?usp=sharing)後，回頭修改，再重新 commit。也就是為何這裡是空的了。

* * *

參考連結
+ 中，[Github SSH key 設定](https://ithelp.ithome.com.tw/articles/10320968?sc=pt)
