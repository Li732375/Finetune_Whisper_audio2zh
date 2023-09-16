"""
from datasets import load_dataset, DatasetDict

# 載入資料集
dataset = DatasetDict()
dataset['train'] = load_dataset("audiofolder", data_dir="audio_dataset")

print(f"\ndataset: \n{dataset}")
print(f"\ndataset['train']: \n{dataset['train']}")
print(f"\ndataset['train'][0]: \n{dataset['train'][0]}")
"""

from datetime import datetime

with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 程式開始時間: {current_time}\n")
    print(f"[log] 程式開始時間: {current_time}")


### Load Data

from datasets import load_dataset

with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 資料載入起始時間: {current_time}\n")
    print(f"[log] 資料載入起始時間: {current_time}")
    
# 載入資料集
dataset = load_dataset("audiofolder", data_dir="audio_dataset",
                       drop_labels=True)

with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 資料載入完成時間: {current_time}\n")
    print(f"[log] 資料載入完成時間: {current_time}")
    
#print(f"\ndataset: \n {dataset}")
#print(f"\ndataset['train']:\n {dataset['train']}")
#print(f"\ndataset['train'][0]:\n {dataset['train'][0]}")

#print(f"\ndataset['test']:\n {dataset['test']}")
#print(f"\ndataset['test'][0]:\n {dataset['test'][0]}")


### Pre-Process the Data

import torch

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("運算裝置：",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


#
from transformers import WhisperProcessor

model_name = "ADT109119/whisper-small-zh-TW"  # 模型名稱
processor = WhisperProcessor.from_pretrained(model_name, language="chinese",
                                             task="transcribe")


# 逐一樣本加載和重新取樣
# 使用特徵提取器從我們的1維音頻數組中計算出對數梅爾頻譜圖的輸入特徵。
# 使用分詞器將轉錄內容編碼為標籤ID。
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["transcription"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 資料預處理起始時間: {current_time}\n")
    print(f"[log] 資料預處理起始時間: {current_time}")

# 逐一樣本處理
common_voice = dataset.map(prepare_dataset,
                           remove_columns=dataset.column_names["train"])
#print(f"\ncommon_voice['train']:\n {common_voice['train']}")
#print(f"\ncommon_voice['train'][0]:\n {common_voice['train'][0]}")

with open("log.txt", "a", encoding="utf-8") as log_file:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 資料預處理完成時間: {current_time}\n")
    print(f"[log] 資料預處理完成時間: {current_time}")

# 
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# initialise the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

#
import evaluate

metric = evaluate.load("wer")


#將標籤 ID 中的-100替換為 pad_token_id，從而還原了在數據整合器中應用的步驟，
#以正確忽略損失中的填充標記。然後，它將預測的 ID 和標籤 ID 解碼為字符串。最
#後，它計算預測和參考標籤之間的 WER。在這里，我們可以選擇使用已經去除標點符
#號和大小寫的“標準化”轉錄和預測來進行評估
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str,
                                     references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm,
                               references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


### Load a Pre-Trained Checkpoint

# load the pre-trained Whisper small checkpoint
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(model.generate, language="chinese", task="transcribe",
                         use_cache=True)


### Define the Training Configuration

#
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="finetune-whisper-small",  # 保存模型權重的本地目錄，它也會是Hugging Face Hub上的模型存儲庫名稱。
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


#
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],  # 這裡的數據亦須自行生成
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


### Training
start_train_t = ""
end_train_t = ""

with open("log.txt", "a", encoding="utf-8") as log_file:
    start_train_t = datetime.now()
    current_time = start_train_t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 訓練開始時間: {current_time}\n")
    print(f"[log] 訓練開始時間: {current_time}")
    
# launch training
trainer.train()

with open("log.txt", "a", encoding="utf-8") as log_file:
    end_train_t = datetime.now()
    current_time = end_train_t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    log_file.write(f"[log] 訓練結束時間: {current_time}\n")
    print(f"[log] 訓練結束時間: {current_time}")

print("*******************************************")


with open("log.txt", "a", encoding="utf-8") as log_file:
    start_datetime = datetime.strptime(start_train_t, "%Y-%m-%d %H:%M:%S.%f")
    end_datetime = datetime.strptime(end_train_t, "%Y-%m-%d %H:%M:%S.%f")
    different_t = end_datetime - start_datetime
    
    # 將時間差轉換為秒數
    time_difference_seconds = time_difference.total_seconds()

    # 格式化時間差為指定格式
    time_difference_str = f"{int(time_difference_seconds // 3600):02}:{int((time_difference_seconds % 3600) // 60):02}:{time_difference_seconds % 60:.2f}"

    log_file.write(f"[log] 訓練期間共費時 {time_difference_str} s\n")
    print(f"訓練期間共費時 {time_difference_str} s")
    
print("\n新模型 fine_tuned_whisper_model 微調完畢")
