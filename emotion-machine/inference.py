import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "./fine_tuned_bert_emotion"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

text = "I'm sad!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

labels = [
    "Sadness", "Anger", "Love", "Surprise", "Fear", "Happiness",
    "Neutral", "Disgust", "Shame", "Guilt", "Confusion", "Desire", "Sarcasm"
]

print(f"Predicted emotion for '{text}': {labels[predicted_class]}")
