#####################################################################################################################
###                                                   Step0                                                       ###
#####################################################################################################################
import pandas as pd
import os

# 数据集路径
data_files = {
    # 双语数据集，用\t分隔
    "en-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/en-zh.txt",
    "de-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/de-zh.txt",
    "ru-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/ru-zh.txt",
    "es-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/es-zh.txt",
    # 单语数据集
    "ja": "../xfdata/多语言机器翻译挑战赛训练集/train/ja.txt",
    "kk": "../xfdata/多语言机器翻译挑战赛训练集/train/kk.txt"
}

# 读取双语数据集
bilingual_data = {}
for key, path in data_files.items():
    if "-" in key:  # 双语数据集
        source_texts = []
        target_texts = []
        target_lang, source_lang = key.split('-')  # 获取源语言和目标语言的代码
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.strip().split('\t')
                if len(split_line) == 2:  # 确保只包含源和目标句子
                    # 在源文本和目标文本前添加语言标识符，并在开头和结尾添加<s>和</s>
                    source_texts.append(f"<{source_lang}> " + split_line[1] + " </s>")
                    target_texts.append(f"<{target_lang}> " + split_line[0] + " </s>")
        bilingual_data[key] = pd.DataFrame({'source': source_texts, 'target': target_texts})

# 读取单语数据集
monolingual_data = {}
for key, path in data_files.items():
    if "-" not in key:  # 单语数据集
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # 在每行文本前添加语言标识符，并在开头和结尾添加<s>和</s>
                texts.append(f"<{key}> " + line.strip() + " </s>")
        monolingual_data[key] = pd.DataFrame({'text': texts})

# 显示双语和单语数据集的前几行（可选）
for name, df in bilingual_data.items():
    print(f"Bilingual dataset ({name}):")
    print(df.head(), "\n")

for name, df in monolingual_data.items():
    print(f"Monolingual dataset ({name}):")
    print(df.head(), "\n")

# 拆分双语数据集
split_data = {k: [] for k in ["zh", "en", "de", "ru", "es"]}
for key, df in bilingual_data.items():
    target_lang, source_lang = key.split('-')
    # 将源语言和目标语言分别保存
    split_data[source_lang].extend(df['source'].tolist())
    split_data[target_lang].extend(df['target'].tolist())

# 合并拆分后的数据到单语数据集中
for lang, texts in split_data.items():
    if lang in monolingual_data:
        # 如果单语数据集已经存在该语言，则合并
        monolingual_data[lang] = pd.concat([monolingual_data[lang], pd.DataFrame({'text': texts})], ignore_index=True)
    else:
        # 如果单语数据集中没有该语言，则直接创建
        monolingual_data[lang] = pd.DataFrame({'text': texts})

os.makedirs("../user_data/step0", exist_ok=True)
# 形成最终的单语数据集
final_monolingual_data = pd.concat(monolingual_data.values(), ignore_index=True).drop_duplicates()
final_monolingual_data.to_csv("../user_data/step0/final_monolingual_data.csv", index=False)


# 修改函数，根据传入的 DataFrame 重新计算 new_counts
def resample_to_new_counts(df, language_column, alpha=0.7):
    # 提取语言标记列
    df['language'] = df[language_column].map(lambda x: x.split()[0])
    
    # 计算每种语言的原始数量和占比
    language_counts = df['language'].value_counts()
    total_count = len(df)
    language_percentage = language_counts / total_count
    
    # 使用 mBART 的采样策略计算权重
    weight_dict = (language_percentage ** alpha) / (language_percentage ** alpha).sum()
    
    # 计算新的采样数量
    new_counts = (weight_dict * total_count).astype(int)
    
    # 创建一个用于存放重采样后数据的列表
    resampled_data = []
    
    # 根据新采样数量对每种语言进行采样
    for language in language_counts.index:
        # 选择该语言对应的子集
        language_subset = df[df['language'] == language]
        
        if language in ['<zh>', '<en>']:
            # 对于中文和英文，执行无放回采样
            resampled_subset = language_subset.sample(n=new_counts[language], replace=False)
        else:
            # 对于其他语言，执行有放回采样
            resampled_subset = language_subset.sample(n=new_counts[language], replace=True)
        
        resampled_data.append(resampled_subset)
    
    # 将所有采样后的数据拼接为新的 DataFrame
    resampled_df = pd.concat(resampled_data).reset_index(drop=True)
    
    return resampled_df, new_counts

# 调用函数生成重采样后的 DataFrame 和新采样数量
resampled_monolingual_data, new_counts = resample_to_new_counts(final_monolingual_data, 'text')
resampled_monolingual_data.to_csv("../user_data/step0/resampled_monolingual_data.csv", index=False)



#####################################################################################################################
###                                                   Step1                                                       ###
#####################################################################################################################

import pandas as pd

# 数据集路径
data_files = {
    # 双语数据集，用\t分隔
    "en-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/en-zh.txt",
    "de-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/de-zh.txt",
    "ru-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/ru-zh.txt",
    "es-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/es-zh.txt",
    # 单语数据集
    "ja": "../xfdata/多语言机器翻译挑战赛训练集/train/ja.txt",
    "kk": "../xfdata/多语言机器翻译挑战赛训练集/train/kk.txt"
}

# 读取双语数据集
bilingual_data = {}
for key, path in data_files.items():
    if "-" in key:  # 双语数据集
        source_texts = []
        target_texts = []
        target_lang, source_lang = key.split('-')  # 获取源语言和目标语言的代码
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.strip().split('\t')
                if len(split_line) == 2:  # 确保只包含源和目标句子
                    # 在源文本和目标文本前添加语言标识符，并在开头和结尾添加<s>和</s>
                    source_texts.append(f"<{source_lang}> " + split_line[1] + " </s>")
                    target_texts.append(f"<{target_lang}> " + split_line[0] + " </s>")
        bilingual_data[key] = pd.DataFrame({'source': source_texts, 'target': target_texts})

os.makedirs("../user_data/step1", exist_ok=True)
for k, v in bilingual_data.items():
    v.to_csv(f"../user_data/step1/{k}.csv", index=False)


#####################################################################################################################
###                                               Step1：en                                                       ###
#####################################################################################################################
import os
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset


df = pd.read_csv("../user_data/step1/en-zh.csv")
dataset = Dataset.from_pandas(df)
# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


tokenized_dataset = dataset.map(tokenize_function, batched=True)
# 划分数据集
tokenized_train_dataset, tokenized_val_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42).values()
# 定义保存路径
output_dir = "../user_data/step1/en/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1/en/dataset 中")


#####################################################################################################################
###                                               Step1：es                                                       ###
#####################################################################################################################

import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset


df = pd.read_csv("../user_data/step1/es-zh.csv")
dataset = Dataset.from_pandas(df)
# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


tokenized_dataset = dataset.map(tokenize_function, batched=True)
# 划分数据集
tokenized_train_dataset, tokenized_val_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42).values()
# 定义保存路径
output_dir = "../user_data/step1/es/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1/es/dataset 中")


#####################################################################################################################
###                                               Step1：de                                                       ###
#####################################################################################################################
import os
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset


df = pd.read_csv("../user_data/step1/de-zh.csv")
dataset = Dataset.from_pandas(df)
# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


tokenized_dataset = dataset.map(tokenize_function, batched=True)
# 划分数据集
tokenized_train_dataset, tokenized_val_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42).values()
# 定义保存路径
output_dir = "../user_data/step1/de/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1/de/dataset 中")



#####################################################################################################################
###                                               Step1：ru                                                       ###
#####################################################################################################################
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset


df = pd.read_csv("../user_data/step1/ru-zh.csv")
dataset = Dataset.from_pandas(df)
# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


tokenized_dataset = dataset.map(tokenize_function, batched=True)
# 划分数据集
tokenized_train_dataset, tokenized_val_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42).values()
# 定义保存路径
output_dir = "../user_data/step1/ru/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1/ru/dataset 中")


#####################################################################################################################
###                                             Step2: ja                                                         ###
#####################################################################################################################
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
import os
from tqdm import tqdm 


# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# 构建双语数据集
# 读取文件，构造数据列表
source_data = []
with open("../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/中文/ja-zh.txt", "r", encoding="utf-8") as f:
    for line in f:
        source_text = line.strip()
        # 添加特殊token
        source_text = f"<zh> {source_text} </s>"
        source_data.append(source_text)
target_data = []
with open("../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/其他语言/ja-zh.txt", "r", encoding="utf-8") as f:
    for line in f:
        target_text = line.strip()
        # 添加特殊token
        target_text = f"<ja> {target_text} </s>"
        target_data.append(target_text)

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


data = []
for source_text, target_text in zip(source_data, target_data):
    data.append({"source": source_text, "target": target_text})
dataset = Dataset.from_list(data)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# 划分数据集
tokenized_train_dataset, tokenized_val_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42).values()
# 定义保存路径
output_dir = "../user_data/step1/ja/dataset"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1/ja/dataset 中")






















from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
import os


# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# 定义双语数据集路径
data_files = {
    "en-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/en-zh.txt",
    "de-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/de-zh.txt",
    "ru-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/ru-zh.txt",
    "es-zh": "../xfdata/多语言机器翻译挑战赛训练集/train/es-zh.txt"
}

# 加载和构建数据集
dataset_dict = {}

# 遍历双语数据集，将其加载并处理为适用于训练的数据集格式
for lang_pair, file_path in data_files.items():
    # 确定语言标签
    target_lang, source_lang = lang_pair.split("-")

    # 读取文件，构造数据列表
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            target_text, source_text = line.strip().split("\t")
            # 添加特殊token，确保 source 是中文， target 是目标语言
            source_text = f"<{source_lang}> {source_text}"
            target_text = f"<{target_lang}> {target_text}"
            data.append({"source": source_text, "target": target_text})

    # 创建Dataset对象
    dataset = Dataset.from_list(data)
    dataset_dict[lang_pair] = dataset

# 创建HFDatasetDict对象
hf_datasets = DatasetDict(dataset_dict)

# 打印数据集信息以确保正确加载
print(hf_datasets)

from transformers import BartConfig, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.model_selection import train_test_split
from datasets import DatasetDict

# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")


# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


# 对四个语言的数据集进行拼接并划分
train_datasets = []
val_datasets = []

for lang_pair in ["en-zh", "de-zh", "ru-zh", "es-zh"]:
    # 使用 datasets 的 train_test_split 方法
    split_dataset = hf_datasets[lang_pair].train_test_split(test_size=0.01)
    train_datasets.append(split_dataset["train"])
    val_datasets.append(split_dataset["test"])

from datasets import concatenate_datasets

# 合并所有训练集和验证集
combined_train_dataset = concatenate_datasets(train_datasets)
combined_val_dataset = concatenate_datasets(val_datasets)

# 对数据集进行 tokenization
tokenized_train_dataset = combined_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = combined_val_dataset.map(tokenize_function, batched=True)

# 定义保存路径
output_dir = "../user_data/step1"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "val"))

print("数据集已保存到 ../user_data/step1 中")


#####################################################################################################################
###                                                   Step2                                                       ###
#####################################################################################################################


from tqdm import tqdm 
from datasets import Dataset


tokenized_train_dataset_en = []
tokenized_train_dataset_de = []
tokenized_train_dataset_ru = []
tokenized_train_dataset_es = []
for example in tqdm(tokenized_train_dataset):
    if example["target"][: 4] == "<en>":
        tokenized_train_dataset_en.append(example)
    if example["target"][: 4] == "<de>":
        tokenized_train_dataset_de.append(example)
    if example["target"][: 4] == "<ru>":
        tokenized_train_dataset_ru.append(example)
    if example["target"][: 4] == "<es>":
        tokenized_train_dataset_es.append(example)

# 定义保存路径
output_dir = "../user_data/step2/train"
os.makedirs(output_dir, exist_ok=True)
tokenized_train_dataset_en = Dataset.from_list(tokenized_train_dataset_en)
tokenized_train_dataset_de = Dataset.from_list(tokenized_train_dataset_de)
tokenized_train_dataset_ru = Dataset.from_list(tokenized_train_dataset_ru)
tokenized_train_dataset_es = Dataset.from_list(tokenized_train_dataset_es)
# 保存数据集
tokenized_train_dataset_en.save_to_disk(os.path.join(output_dir, "en"))
tokenized_train_dataset_de.save_to_disk(os.path.join(output_dir, "de"))
tokenized_train_dataset_ru.save_to_disk(os.path.join(output_dir, "ru"))
tokenized_train_dataset_es.save_to_disk(os.path.join(output_dir, "es"))
print("数据集已保存到 ../user_data/step2/train 中")

tokenized_val_dataset_en = []
tokenized_val_dataset_de = []
tokenized_val_dataset_ru = []
tokenized_val_dataset_es = []
for example in tqdm(tokenized_val_dataset):
    if example["target"][: 4] == "<en>":
        tokenized_val_dataset_en.append(example)
    if example["target"][: 4] == "<de>":
        tokenized_val_dataset_de.append(example)
    if example["target"][: 4] == "<ru>":
        tokenized_val_dataset_ru.append(example)
    if example["target"][: 4] == "<es>":
        tokenized_val_dataset_es.append(example)
        
# 定义保存路径
output_dir = "../user_data/step2/val"
os.makedirs(output_dir, exist_ok=True)
tokenized_val_dataset_en = Dataset.from_list(tokenized_val_dataset_en)
tokenized_val_dataset_de = Dataset.from_list(tokenized_val_dataset_de)
tokenized_val_dataset_ru = Dataset.from_list(tokenized_val_dataset_ru)
tokenized_val_dataset_es = Dataset.from_list(tokenized_val_dataset_es)
# 保存数据集
tokenized_val_dataset_en.save_to_disk(os.path.join(output_dir, "en"))
tokenized_val_dataset_de.save_to_disk(os.path.join(output_dir, "de"))
tokenized_val_dataset_ru.save_to_disk(os.path.join(output_dir, "ru"))
tokenized_val_dataset_es.save_to_disk(os.path.join(output_dir, "es"))
print("数据集已保存到 ../user_data/step2/val 中")


#####################################################################################################################
###                                             Step3: 日语                                                       ###
#####################################################################################################################
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
import os
from opencc import OpenCC
from tqdm import tqdm 

# 创建一个转换器，s2t 表示从简体转为繁体
cc = OpenCC('t2s')

# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# 日语数据集路径
data_files = {
    "single": "../xfdata/多语言机器翻译挑战赛训练集/train/ja.txt",
    "double_source": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/中文/ja-zh.txt",
    "double_target": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/其他语言/ja-zh.txt"
    
}

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}


# 构建单语数据集
target_lang, source_lang = "ja", "zh"
data = []
with open(data_files["single"], "r", encoding="utf-8") as f:
    for line in tqdm(f, total=100000):
        # 转化为简体
        encoded_source_text = tokenizer.encode(line.strip())
        source_text = ["<zh>"]
        for token_id in encoded_source_text:
            token = tokenizer.decode(token_id)
            # 若是简体
            if '\u4e00' <= token <= '\u9fff':
                source_text.append(token)
            else:
                source_text.append("<mask>")
        source_text = "".join(source_text)
        target_text = line.strip()
        # 添加特殊token，确保target是目标语言
        target_text = f"<{target_lang}> {target_text}"    
        data.append({"source": source_text, "target": target_text})
    
single_dataset = Dataset.from_list(data)

split_dataset = single_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "ja/single_train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "ja/single_val"))


# 构建双语数据集
# 读取文件，构造数据列表
source_data = []
with open(data_files["double_source"], "r", encoding="utf-8") as f:
    for line in f:
        source_text = line.strip()
        # 添加特殊token
        source_text = f"<{source_lang}> {source_text}"
        source_data.append(source_text)
target_data = []
with open(data_files["double_target"], "r", encoding="utf-8") as f:
    for line in f:
        target_text = line.strip()
        # 添加特殊token
        target_text = f"<{target_lang}> {target_text}"
        target_data.append(target_text)
        
data = []
for source_text, target_text in zip(source_data, target_data):
    data.append({"source": source_text, "target": target_text})
dataset = Dataset.from_list(data)

split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "ja/train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "ja/val"))

print("数据集已保存到 ../user_data/step3 中")



# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}

split_dataset = single_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "ja/single_train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "ja/single_val"))

print("数据集已保存到 ../user_data/step3 中")



#####################################################################################################################
###                                             Step4: 哈萨克                                                     ###
#####################################################################################################################

from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
import os


# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

# 哈萨克数据集路径
data_files = {
    "single": "../xfdata/多语言机器翻译挑战赛训练集/train/kk.txt",
    "double_source": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/中文/kk-zh.txt",
    "double_target": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/val/其他语言/kk-zh.txt"
    
}

# 定义保存路径
output_dir = "../user_data/step4"
os.makedirs(output_dir, exist_ok=True)
# 构建单语数据集
target_lang, source_lang = "kk", "zh"
# 构建双语数据集
# 读取文件，构造数据列表
source_data = []
with open(data_files["double_source"], "r", encoding="utf-8") as f:
    for line in f:
        source_text = line.strip()
        # 添加特殊token
        source_text = f"<{source_lang}> {source_text}"
        source_data.append(source_text)
target_data = []
with open(data_files["double_target"], "r", encoding="utf-8") as f:
    for line in f:
        target_text = line.strip()
        # 添加特殊token
        target_text = f"<{target_lang}> {target_text}"
        target_data.append(target_text)
        
data = []
for source_text, target_text in zip(source_data, target_data):
    data.append({"source": source_text, "target": target_text})
dataset = Dataset.from_list(data)

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    target_texts = examples["target"]

    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Tokenize target texts without using as_target_tokenizer context
    labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # 将 labels 直接添加到 model_inputs
    model_inputs["labels"] = labels["input_ids"]

    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}

split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
# 保存 tokenized 数据集
tokenized_train_dataset.save_to_disk(os.path.join(output_dir, "kk/train"))
tokenized_val_dataset.save_to_disk(os.path.join(output_dir, "kk/val"))

print("数据集已保存到 ../user_data/step4 中")


#####################################################################################################################
###                                             Step5: 预测                                                       ###
#####################################################################################################################


from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
import os


# 数据集字典
dataset_dict = DatasetDict()
# 定义双语数据集路径
data_files = {
    "en-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/en-zh.txt",
    "de-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/de-zh.txt",
    "ru-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/ru-zh.txt",
    "es-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/es-zh.txt",
    "ja-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/ja-zh.txt",
    "kk-zh": "../xfdata/多语言机器翻译挑战赛数据集更新（以此测试集提交得分为准）/test/中文/kk-zh.txt",
}
# 加载分词器
tokenizer = PreTrainedTokenizerFast.from_pretrained("../user_data/bart_tokenizer")

for target_lang in ["en", "de", "ru", "es", "ja", "kk"]:
    # 读取文件，构造数据列表
    data = []
    with open(data_files[f"{target_lang}-zh"], "r", encoding="utf-8") as f:
        for line in f:
            source_text = line.strip()
            # 添加特殊token，确保 source 是中文
            source_text = f"<zh> {source_text}"
            data.append({"source": source_text})
    # 创建Dataset对象
    dataset = Dataset.from_list(data)
    dataset_dict[f"{target_lang}-zh"] = dataset

# Tokenize 函数
def tokenize_function(examples):
    source_texts = examples["source"]
    # Tokenize source texts
    model_inputs = tokenizer(source_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    # 转换成字典格式，便于 datasets 库使用
    return {key: value.tolist() for key, value in model_inputs.items()}

tokenized_dataset_dict = dataset_dict.map(tokenize_function)
tokenized_dataset_dict.save_to_disk("../user_data/step5")
print("数据集已保存到 ../user_data/step5 中")
