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

# Enhanced HTML template components (Navigation Bar and Footer)
NAVBAR = """
<nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
  <div class="container">
    <a class="navbar-brand" href="/">Fake News Detector</a>
  </div>
</nav>
"""

FOOTER = """
<footer class="footer mt-5 py-3 bg-light">
  <div class="container text-center">
    <span class="text-muted">© 2025 Krishna Balaji | <a href="https://github.com/krishna31102004/fakenewsdetector" target="_blank">GitHub Repo</a></span>
  </div>
</footer>
"""

# Root route: shows an HTML form for in-browser classification with enhanced UI
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
                    <style>
                        body { padding-top: 70px; }
                        .container { max-width: 700px; }
                    </style>
                </head>
                <body>
                    ''' + NAVBAR + '''
                    <div class="container mt-5">
                        <div class="alert alert-danger" role="alert">
                            Please enter some text to classify.
                        </div>
                        <a href="/" class="btn btn-primary">Go Back</a>
                    </div>
                    ''' + FOOTER + '''
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
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
                    body { padding-top: 70px; }
                    .container { max-width: 700px; }
                </style>
            </head>
            <body>
                ''' + NAVBAR + '''
                <div class="container mt-5">
                    <div class="card">
                        <div class="card-header">
                            Fake News Detector - Result
                        </div>
                        <div class="card-body">
                            <h5 class="card-title">Your Input:</h5>
                            <p class="card-text">{{ user_text }}</p>
                            <h5 class="card-title">Prediction:</h5>
                            <p class="card-text"><span class="badge bg-info text-dark">{{ prediction }}</span></p>
                            <a href="/" class="btn btn-primary">Try Another</a>
                        </div>
                    </div>
                </div>
                ''' + FOOTER + '''
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
        ''', user_text=user_text, prediction=result)
    else:
        # GET: Show the form with enhanced styling
        return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Fake News Detector</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { padding-top: 70px; }
                    .container { max-width: 700px; }
                </style>
            </head>
            <body>
                ''' + NAVBAR + '''
                <div class="container">
                    <div class="card mt-4">
                        <div class="card-header">
                            Enter News Article Text
                        </div>
                        <div class="card-body">
                            <form method="POST">
                                <div class="mb-3">
                                    <textarea class="form-control" name="text" rows="8" placeholder="Type or paste your news article here..."></textarea>
                                </div>
                                <button type="submit" class="btn btn-primary">Classify</button>
                            </form>
                        </div>
                    </div>
                    <div class="mt-4">
                        <p>You can also use the API directly by sending a POST request to <code>/predict</code>.</p>
                    </div>
                </div>
                ''' + FOOTER + '''
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
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
