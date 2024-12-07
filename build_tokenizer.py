import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os 
import json



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

print("SentencePiece 分词器训练完成。")


import sentencepiece as spm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
import json


# 加载已有的 SentencePiece 模型
sp = spm.SentencePieceProcessor()
sp.load("../user_data/tokenizer.model")

# 测试模型是否可以正常分词
print(sp.encode_as_pieces("你好，世界！This is a test sentence."))

# 获取词汇表大小
vocab_size = sp.get_piece_size()

# 将词汇表保存为文本文件
with open("../user_data/tokenizer_vocab.txt", "w", encoding="utf-8") as vocab_file:
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        vocab_file.write(f"{piece}\n")

print("词汇表已保存为文本格式。")

# 初始化空的 Tokenizer
tokenizer = Tokenizer(models.BPE())

# 配置 ByteLevel 预分词器和解码器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

# 定义特定用途的标记符
pad_token = "<pad>"
bos_token = "<s>"
eos_token = "</s>"
unk_token = "<unk>"
mask_token = "<mask>"

# 定义附加的特殊标记符
additional_special_tokens = ["<zh>", "<en>", "<de>", "<ru>", "<es>", "<ja>", "<kk>"]

# 加载词汇表并训练分词器
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=[pad_token, bos_token, eos_token, unk_token, mask_token] + additional_special_tokens
)
tokenizer.train(files=["../user_data/tokenizer_vocab.txt"], trainer=trainer)

# 保存分词器为 transformers 兼容格式
bart_tokenizer_path = "../user_data/bart_tokenizer"
os.makedirs(bart_tokenizer_path, exist_ok=True)
tokenizer.save(os.path.join(bart_tokenizer_path, "tokenizer.json"))
print("分词器已保存为 transformers 兼容格式。")

# 创建 tokenizer_config.json 配置文件
tokenizer_config = {
    "bos_token": bos_token,
    "eos_token": eos_token,
    "pad_token": pad_token,
    "unk_token": unk_token,
    "mask_token": mask_token,
    "additional_special_tokens": additional_special_tokens
}

# 将配置保存为 tokenizer_config.json
with open(os.path.join(bart_tokenizer_path, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(tokenizer_config, config_file, ensure_ascii=False, indent=4)

print("tokenizer_config.json 配置文件已保存。")
