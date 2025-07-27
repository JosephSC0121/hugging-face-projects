import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

data = {
     "text": [
         "I love you so much!",
         "This is absolutely disgusting!",
         "I'm so happy with my new phone!",
         "Why does this always break?",
         "I feel so alone right now."
     ],
     "label": [2, 7, 5, 1, 0]  
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

model_name = "boltuix/bert-emotion"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=13)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

def to_torch_format(example):
    return {
        "input_ids": torch.tensor(example["input_ids"]),
        "attention_mask": torch.tensor(example["attention_mask"]),
        "label": torch.tensor(example["label"])
    }

tokenized_dataset = tokenized_dataset.map(to_torch_format)

training_args = TrainingArguments(
    output_dir="./bert_emotion_results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_dir="./bert_emotion_logs",
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    learning_rate=3e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_bert_emotion")
tokenizer.save_pretrained("./fine_tuned_bert_emotion")

