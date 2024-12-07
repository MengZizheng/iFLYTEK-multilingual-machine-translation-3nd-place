# 2024 iFLYTEK A.I. 开发者大赛：多语言机器翻译挑战赛

## 竞赛简介

本项目是参加 **2024 iFLYTEK A.I. 开发者大赛：多语言机器翻译挑战赛** 的解决方案，本方案**没有获奖**，但是其中的**SentencePiece**、**BART**以及**Seq2Seq训练**是有意义的。竞赛的任务是从中文翻译成多种语言，包括英语、德语、俄语、西班牙语、日语和哈萨克语。由于低资源语言的训练数据较少，我们的方案结合了多语言的双语数据和单语数据，采用了 BART 模型来完成翻译任务。

赛题链接：  
[多语言机器翻译挑战赛](https://challenge.xfyun.cn/topic/info?type=multilingual-machine-translation&option=ssgy)

数据链接：  
[赛题数据](https://challenge.xfyun.cn/topic/info?type=multilingual-machine-translation&option=stsj)

## 步骤概述

### 1. 构建 SentencePiece 分词器

我们首先使用 **SentencePiece** 训练分词器，处理来自双语和单语数据的多种语言。以下是训练分词器的代码：

```python
import sentencepiece as spm

# 使用 SentencePiece 训练分词器
spm.SentencePieceTrainer.Train(
    "--input=../xfdata/多语言机器翻译挑战赛训练集/train/en-zh.txt,\
../xfdata/多语言机器翻译挑战赛训练集/train/de-zh.txt,\
../xfdata/多语言机器翻译挑战赛训练集/train/ru-zh.txt,\
../xfdata/多语言机器翻译挑战赛训练集/train/es-zh.txt,\
../xfdata/多语言机器翻译挑战赛训练集/train/ja.txt,\
../xfdata/多语言机器翻译挑战赛训练集/train/kk.txt \
--model_prefix=../user_data/tokenizer \
--vocab_size=72000 \
--model_type=bpe \
--character_coverage=0.9995 \
--input_sentence_size=2800001 \
--shuffle_input_sentence=true \
--num_threads=16 \
--bos_piece=<s> \
--eos_piece=</s> \
--unk_piece=<unk> \
--pad_piece=<pad> \
--minloglevel=0"
)
```

训练完成后，我们将分词器保存并准备好用于模型训练。

### 2. 模型初始化

我们使用 **BART** 模型，结合自定义的配置和训练数据，来构建多语言机器翻译模型。

```python
from transformers import BartConfig, BartForConditionalGeneration

# 初始化 BART 模型配置
config = BartConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=128,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    d_model=512,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# 创建 BART 模型
model = BartForConditionalGeneration(config)
```

### 3. 数据处理和准备

我们使用 **NoisyTextDataset** 类处理数据，将文本数据转换为模型能够理解的格式。以下是数据准备的代码：

```python
from utils import NoisyTextDataset

# 加载数据集并转换为模型输入
texts = resampled_monolingual_data['text'].tolist()
dataset = NoisyTextDataset(texts, tokenizer)
```

### 4. 模型微调

在模型训练阶段，我们使用 **Seq2SeqTrainer** 来进行微调。微调时我们设置了适当的训练参数，例如学习率、批次大小和训练的 epochs 数量：

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=5000,
    save_total_limit=3,
    learning_rate=1e-4,
)

# 初始化 Seq2SeqTrainer 并开始训练
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 5. 评估和测试

在训练完成后，我们会使用验证集进行模型评估，并根据最终的测试集输出翻译结果，生成翻译质量报告。

## 6. 算力平台

为了高效训练模型，我们使用了 [onethingai](https://onethingai.com/invitation?code=wGZHFckZ) 提供的算力平台。该平台提供了强大的GPU资源，使我们能够在较短的时间内完成模型训练和微调。

## 7. 贡献者

- **团队名称**：小老正
- **成员**：[孟子正]
