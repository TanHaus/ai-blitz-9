import pandas as pd

train_df = pd.read_csv("../data/train.csv")
val_df = pd.read_csv("../data/val.csv")
test_df = pd.read_csv("../data/test.csv")

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenized_train = tokenizer(train_df["text"].to_list(), max_length=512, truncation=True, padding=True)
tokenized_val = tokenizer(val_df["text"].to_list(), max_length=512, truncation=True, padding=True)
tokenized_test = tokenizer(test_df["text"].to_list(), max_length=512, truncation=True, padding=True)

import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenized_text, labels):
        self.tokenized_text = tokenized_text
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.tokenized_text.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = TextDataset(tokenized_train, train_df["label"].to_list())
val_dataset = TextDataset(tokenized_val, val_df["label"].to_list())
test_dataset = TextDataset(tokenized_test, test_df["label"].to_list())

training_args = TrainingArguments(
    output_dir="results",
    logging_dir="logs",
    num_train_epochs=3,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=512,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
