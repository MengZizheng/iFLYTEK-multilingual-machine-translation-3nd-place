import re
import random
from torch.utils.data import Dataset


class NoisyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, noise_factor=0.3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_factor = noise_factor

    def add_noise_to_text(self, text):
        try:
            # 使用 tokenizer 对文本进行编码为 input_ids
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if len(input_ids) < 3:  # 确保文本至少包含 <lang>, </s> 和其他内容
                return text  # 如果长度不够，返回原始文本
            
            # 处理开头和结尾特殊标记
            start_token = input_ids[0]  # <lang> 标记
            end_token = input_ids[-1] if input_ids[-1] == self.tokenizer.convert_tokens_to_ids("</s>") else None
            middle_tokens = input_ids[1:-1] if end_token is not None else input_ids[1:]  # 中间部分的 tokens
            
            # 确保至少保留一些有效内容，不能全部加噪
            num_tokens = len(middle_tokens)
            max_mask_num = int(num_tokens * self.noise_factor)
            max_mask_num = min(max_mask_num, num_tokens - 1)  # 至少保留一个有效 token

            # Token Masking
            if random.random() < 0.3 and max_mask_num > 0:
                mask_token_id = self.tokenizer.convert_tokens_to_ids("<mask>")
                mask_indices = random.sample(range(len(middle_tokens)), min(max_mask_num, len(middle_tokens)))
                for idx in mask_indices:
                    middle_tokens[idx] = mask_token_id

            # Token Deletion
            if random.random() < 0.2 and max_mask_num > 0:
                delete_indices = sorted(random.sample(range(len(middle_tokens)), min(max_mask_num, len(middle_tokens))), reverse=True)
                for idx in delete_indices:
                    del middle_tokens[idx]

            # Text Infilling
            if random.random() < 0.2 and max_mask_num > 0:
                infill_indices = sorted(random.sample(range(len(middle_tokens)), min(max_mask_num, len(middle_tokens))), reverse=True)
                for idx in infill_indices:
                    middle_tokens[idx] = mask_token_id

            # Sentence Permutation (基于标点符号)
            if random.random() < 0.2:
                decoded_middle_tokens = self.tokenizer.decode(middle_tokens)
                sentences = re.split(r'(?<=[。.!?！？])', decoded_middle_tokens)  # 以标点符号作为分隔符来分割句子
                sentences = [s.strip() for s in sentences if s.strip()]  # 去掉空白句子
                if len(sentences) > 1:  # 只有在有多个句子时才进行打乱
                    random.shuffle(sentences)
                    shuffled_text = ''.join(sentences)
                    middle_tokens = self.tokenizer.encode(shuffled_text, add_special_tokens=False)

            # Document Rotation
            if random.random() < 0.1 and len(middle_tokens) > 1:
                rotate_index = random.randint(1, len(middle_tokens) - 1)  # 确保旋转位置不为0，避免全部旋转
                middle_tokens = middle_tokens[rotate_index:] + middle_tokens[:rotate_index]

            # 组合成完整的文本
            final_tokens = [start_token] + middle_tokens
            if end_token is not None:
                final_tokens.append(end_token)

            return self.tokenizer.decode(final_tokens)
        
        except Exception as e:
            return text  # 返回原始文本以继续训练

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        original_text = self.texts[idx]
        
        # 添加异常处理，避免异常数据阻碍获取样本
        try:
            noisy_text = self.add_noise_to_text(original_text)

            # 使用 text_target 参数对目标文本进行编码
            encoded = self.tokenizer(
                noisy_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                text_target=original_text
            )

            # 去掉 batch 维度
            inputs = {key: val.squeeze() for key, val in encoded.items()}
            return inputs
        
        except Exception as e:
            # 如果在 __getitem__ 中遇到异常，打印错误并跳过
            return self.__getitem__((idx + 1) % len(self))  # 返回下一个样本，循环避免越界