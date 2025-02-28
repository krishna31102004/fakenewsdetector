import os
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Your Hugging Face model repo
model_path = "kb4086/fakenewsdetector"

# Load model/tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_text(text):
    """Return 'Fake News' or 'Real News' given input text."""
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


# -------------------------
# 1) Root route with GET/POST
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def classify_in_browser():
    """If GET, show an HTML form; if POST, show prediction immediately."""
    if request.method == 'POST':
        # The user submitted the form
        user_text = request.form.get('text', '')
        result = predict_text(user_text)
        # Return a quick HTML snippet showing the result
        return render_template_string('''
            <h1>Fake News Detector</h1>
            <p><strong>Input:</strong> {{ user_text }}</p>
            <p><strong>Prediction:</strong> {{ prediction }}</p>
            <hr>
            <a href="/">Try another</a>
        ''', user_text=user_text, prediction=result)
    else:
        # Just show the form for the user to enter text
        return render_template_string('''
            <h1>Fake News Detector</h1>
            <form method="POST">
                <label for="text">Enter news article text:</label><br><br>
                <textarea name="text" rows="8" cols="60"
                          placeholder="Type or paste your news article here..."></textarea><br><br>
                <button type="submit">Classify</button>
            </form>
            <hr>
            <p>You can also POST JSON to <code>/predict</code> if you prefer a programmatic approach.</p>
        ''')


# -------------------------
# 2) Programmatic endpoint
# -------------------------
@app.route('/predict', methods=['POST'])
def predict_json():
    """Accept JSON: {"text": "..."} and return JSON response."""
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400

    result = predict_text(data["text"])
    return jsonify({"prediction": result})


# -------------------------
# Run on the port that Render or other platforms provide
# -------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
