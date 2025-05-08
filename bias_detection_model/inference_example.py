
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./bias_detection_model")
    tokenizer = AutoTokenizer.from_pretrained("./bias_detection_model")
    with open("./bias_detection_model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

def predict(text):
    model, tokenizer, le = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = le.inverse_transform([torch.argmax(probs).cpu().numpy()])
    return pred_label[0]

if __name__ == "__main__":
    text = input("Enter text to analyze: ")
    print("Predicted bias:", predict(text))
