import os
import logging
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Set up basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Use your Hugging Face model repository ID
model_path = "kb4086/fakenewsdetector"

# Load the model and tokenizer with error handling
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    logging.info("Model loaded successfully from Hugging Face.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise e

def predict_text(text):
    """Returns 'Fake News' or 'Real News' for the given text."""
    try:
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
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return "Error during prediction"

# Root route: shows an HTML form for in-browser classification
@app.route('/', methods=['GET', 'POST'])
def classify_in_browser():
    if request.method == 'POST':
        user_text = request.form.get('text', '')
        if not user_text.strip():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Fake News Detector - Error</title>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                </head>
                <body>
                    <div class="container mt-5">
                        <h1>Error</h1>
                        <p>Please enter some text to classify.</p>
                        <a href="/" class="btn btn-primary">Go Back</a>
                    </div>
                </body>
                </html>
            ''')
        result = predict_text(user_text)
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fake News Detector - Result</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { padding-top: 50px; }
                    .container { max-width: 700px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Fake News Detector - Result</h1>
                    <div class="mb-3">
                        <label class="form-label"><strong>Input:</strong></label>
                        <div class="alert alert-secondary">{{ user_text }}</div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label"><strong>Prediction:</strong></label>
                        <div class="alert alert-info">{{ prediction }}</div>
                    </div>
                    <a href="/" class="btn btn-primary">Try Another</a>
                </div>
            </body>
            </html>
        ''', user_text=user_text, prediction=result)
    else:
        # GET: Show the form
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fake News Detector</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { padding-top: 50px; }
                    .container { max-width: 700px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="mb-4">Fake News Detector</h1>
                    <form method="POST">
                        <div class="mb-3">
                            <label for="text" class="form-label">Enter News Article Text:</label>
                            <textarea class="form-control" id="text" name="text" rows="8" placeholder="Type or paste your news article here..."></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary">Classify</button>
                    </form>
                    <hr>
                    <p>You can also use the API directly by sending a POST request to <code>/predict</code>.</p>
                </div>
            </body>
            </html>
        ''')

# JSON-based prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_json():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400
    result = predict_text(data["text"])
    return jsonify({"prediction": result})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
