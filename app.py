from flask import Flask, request, jsonify
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Replace "kb4086/fakenewsdetector" with your actual HF username/repo name
model_path = "kb4086/fakenewsdetector"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_text(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Fake News" if prediction == 0 else "Real News"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400

    result = predict_text(data["text"])
    return jsonify({"prediction": result})

@app.route('/')
def home():
    return "Hello! Go to /predict to POST your text."


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # read from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
