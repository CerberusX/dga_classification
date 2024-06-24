import random
import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self, tokenized_urls, tokenizer, mask_probability=0.15):
        self.input_ids = []
        self.labels = []
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

        for tokens in tokenized_urls:
            input_id = tokens.copy()
            label = [-100] * len(input_id)

            for i, token in enumerate(tokens):
                if random.random() < self.mask_probability:
                    input_id[i] = self.tokenizer.mask_token_id
                    label[i] = tokens[i]

            self.input_ids.append(input_id)
            self.labels.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ClassificationDataset(Dataset):
    # Currently DGA dataset is used for classification, it can be replaced.
    def __init__(self, tokenized_urls, labels, tokenizer, max_length=128):
        self.tokenized_urls = tokenized_urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tokenized_urls)

    def __getitem__(self, idx):
        tokens = self.tokenized_urls[idx]
        input_ids = tokens.copy()
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def preprocess_urls(urls, tokenizer, max_length=128):
    tokenized_urls = [tokenizer.encode(url, add_special_tokens=True, truncation=True, max_length=max_length) for url in urls]
    return tokenized_urls
#test