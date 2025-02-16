import json
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def load_hellaswag(split: str, tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024) -> "HellaSwagDataset":
    """Loads the HellaSwag dataset and tokenizes it."""
    assert split in ["train", "val", "test"]
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Define paths relative to the data directory
    data_file = data_dir / f"hellaswag_{split}.jsonl"
    save_dir = data_dir / "preprocessed" / tokenizer.name_or_path / f"hellaswag_{split}_max_seq_length_{max_seq_length}"

    save_dir.mkdir(parents=True, exist_ok=True)  # Create the save directory
    processed_data_file = save_dir / f"hellaswag_{split}_processed.pt"
    # Download if the data file doesn't exist
    if not data_file.exists():
        print(f"Downloading HellaSwag {split} dataset...")
        #Use raw github files
        raw_train_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"
        raw_val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

        if split == "train":
            url = raw_train_url
        elif split == "val":
            url = raw_val_url
        else:
            raise ValueError("test split is not a raw github file")
        torch.hub.download_url_to_file(url, data_file)

    # Load and preprocess the data if it hasn't been done already
    if not processed_data_file.exists():
        print(f"Preprocessing HellaSwag {split} dataset...")
        with open(data_file, "r") as f:
            raw_data = [json.loads(line) for line in f]
        dataset = HellaSwagDataset(raw_data, tokenizer, max_seq_length)
        torch.save(dataset, processed_data_file)
    else: # Load preprocessed data
        print(f"Loading preprocessed HellaSwag {split} dataset...")
        dataset = torch.load(processed_data_file)
    return dataset

def load_raw_hellaswag(split: str, data_dir: str = "data") -> Sequence[dict]:
    """Loads raw (untokenized) HellaSwag data."""
    data_dir = Path(data_dir)
    data_file = data_dir / f"hellaswag_{split}.jsonl"

    if not data_file.exists():
        # Download logic (same as in load_hellaswag)
        print(f"Downloading HellaSwag {split} dataset...")
        raw_train_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"
        raw_val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

        if split == "train":
            url = raw_train_url
        elif split == "val":
            url = raw_val_url
        else:
            raise ValueError("test split is not a raw github file")
        torch.hub.download_url_to_file(url, data_file)
    with open(data_file, "r") as f:
        return [json.loads(line) for line in f]

def iterate_examples(split, data_dir="data"):
    """Yields HellaSwag examples one at a time.
        Helpful for evaluating without loading into memory."""
    data = load_raw_hellaswag(split, data_dir)
    for example in data:
        yield example

def render_example(example, tokenizer, max_seq_length: int = 1024) -> Tuple[torch.LongTensor,
                                                         torch.LongTensor,
                                                         torch.LongTensor,
                                                         int]:
    """Tokenizes a HellaSwag example."""
    ctx_a = example["ctx_a"]
    ctx_b = example["ctx_b"]
    endings = example["endings"]
    label = example["label"]

    # Concatenate context and endings, truncating at max_seq_length
    contexts = [f"{ctx_a} {ctx_b} {ending}" for ending in endings]
    encoded = tokenizer(contexts, add_special_tokens=True, truncation=True,
                        max_length=max_seq_length, padding="max_length",  # Pad during tokenization
                        return_tensors='pt')

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Return as a tuple of tensors, as expected in the original notebook
    return input_ids, attention_mask, label

def get_most_likely_row(input_ids, attention_mask, logits, return_probs = False):
    padding_id = 0
    mask = input_ids != padding_id
    mask = mask.type(attention_mask.dtype) * attention_mask
    logits = logits * mask.unsqueeze(-1) + -1e15 * (1 - mask.unsqueeze(-1))

    logits = logits.sum(dim=1)
    logits = logits / mask.sum(dim=1, keepdim=True)
    if return_probs:
        return torch.softmax(logits, dim=-1)

    most_likely = logits.argmax(dim=-1)

    return most_likely

class HellaSwagDataset(Dataset):
    """Tokenized HellaSwag dataset."""

    def __init__(self, raw_data: Sequence[dict], tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []

        for item in raw_data:
            input_ids, attention_mask, label = render_example(item, tokenizer, max_seq_length)
            # Store as individual tensors, not one large tensor
            for i in range(input_ids.shape[0]):
                self.data.append({
                    'input_ids': input_ids[i],
                    'attention_mask': attention_mask[i],
                    'label': torch.tensor(1 if i == label else 0)  # One-hot-like label
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
