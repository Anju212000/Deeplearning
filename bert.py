from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

def load_model(model_name="./sentiment_model"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model,tokenizer
def predict_sentiment(text):
    model,tokenizer=load_model()
    label_encoder = LabelEncoder()
    data=pd.read_csv("labels.csv")
    label_encoder.fit_transform(data["Class"])

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the model's device
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return label_encoder.inverse_transform(predictions.tolist())



