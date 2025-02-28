from flask import Flask, request, jsonify, render_template_string
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

model_path = "kb4086/fakenewsdetector"  # Your HF model repo
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

@app.route('/')
def home():
    return "Hello! Go to /predict to POST your text, or /classify for a simple UI."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400

    result = predict_text(data["text"])
    return jsonify({"prediction": result})

# -------------- New UI route --------------
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        # Form submission
        user_text = request.form.get('text', '')
        result = predict_text(user_text)
        return render_template_string('''
            <h1>Fake News Detector</h1>
            <p><strong>Input:</strong> {{ text }}</p>
            <p><strong>Prediction:</strong> {{ result }}</p>
            <a href="/classify">Try another</a>
        ''', text=user_text, result=result)
    else:
        # Show form
        return render_template_string('''
            <h1>Fake News Detector</h1>
            <form method="POST">
                <label for="text">Enter News Article Text:</label><br><br>
                <textarea name="text" rows="8" cols="60" placeholder="Type or paste your news article here..."></textarea><br><br>
                <button type="submit">Submit</button>
            </form>
        ''')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
