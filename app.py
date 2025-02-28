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

# Simple cache to store predictions for repeated texts
cache = {}

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

def predict_text_with_confidence(text):
    """
    Returns a tuple (label, confidence) for the given text.
    Confidence is a float between 0 and 1.
    Uses caching to avoid recomputation.
    """
    if text in cache:
        return cache[text]
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
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
        label = "Fake News" if prediction == 0 else "Real News"
        result = (label, confidence)
        cache[text] = result
        return result
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return ("Error during prediction", 0.0)

# Enhanced HTML template components (Navigation Bar and Footer)
NAVBAR = """
<nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
  <div class="container">
    <a class="navbar-brand" href="/">Fake News Detector</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav ms-auto">
        <li class="nav-item">
          <a class="nav-link" href="/about">About</a>
        </li>
      </ul>
    </div>
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

# HTML template for the form and results (enhanced with Bootstrap)
HTML_TEMPLATE = """
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
    {{ navbar|safe }}
    <div class="container">
        {% if error %}
        <div class="alert alert-danger mt-3" role="alert">
            {{ error }}
        </div>
        {% endif %}
        {% if user_text == "" and prediction is none %}
            <div class="card mt-4">
                <div class="card-header">
                    Enter News Article Text
                </div>
                <div class="card-body">
                    <form method="POST">
                        <div class="mb-3">
                            <textarea name="text" class="form-control" rows="8" placeholder="Type or paste your news article here..."></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary">Classify</button>
                    </form>
                </div>
            </div>
            <div class="mt-4">
                <p>You can also use the API directly by sending a POST request to <code>/predict</code>.</p>
            </div>
        {% else %}
            <div class="card mt-4">
                <div class="card-header">
                    Classification Result
                </div>
                <div class="card-body">
                    <h5 class="card-title">Your Input:</h5>
                    <p class="card-text">{{ user_text }}</p>
                    <h5 class="card-title">Prediction:</h5>
                    <p class="card-text">
                        <span class="badge bg-info text-dark">{{ prediction }}</span>
                        {% if confidence is not none %}
                        (Confidence: {{ '%.2f'|format(confidence * 100) }}%)
                        {% endif %}
                    </p>
                    <a href="/" class="btn btn-secondary">Classify Another</a>
                </div>
            </div>
        {% endif %}
    </div>
    {{ footer|safe }}
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Root route: shows an HTML form for in-browser classification with enhanced UI and confidence display
@app.route('/', methods=['GET', 'POST'])
def classify_in_browser():
    if request.method == 'POST':
        user_text = request.form.get('text', '')
        if not user_text.strip():
            return render_template_string(HTML_TEMPLATE, 
                                          navbar=NAVBAR, footer=FOOTER, 
                                          user_text="", prediction=None, confidence=None,
                                          error="Please enter some text to classify.")
        prediction, confidence = predict_text_with_confidence(user_text)
        return render_template_string(HTML_TEMPLATE, 
                                      navbar=NAVBAR, footer=FOOTER, 
                                      user_text=user_text, prediction=prediction, confidence=confidence,
                                      error=None)
    else:
        # GET: Show the form
        return render_template_string(HTML_TEMPLATE, 
                                      navbar=NAVBAR, footer=FOOTER, 
                                      user_text="", prediction=None, confidence=None,
                                      error=None)

# JSON-based prediction endpoint (now returns confidence as well)
@app.route('/predict', methods=['POST'])
def predict_json():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400
    prediction, confidence = predict_text_with_confidence(data["text"])
    return jsonify({"prediction": prediction, "confidence": confidence})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# About page route
@app.route('/about', methods=['GET'])
def about():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>About - Fake News Detector</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 70px; }
                .container { max-width: 700px; }
            </style>
        </head>
        <body>
            {{ navbar|safe }}
            <div class="container mt-5">
                <h1>About Fake News Detector</h1>
                <p>This project uses a fine-tuned DistilBERT model to classify news articles as "Fake News" or "Real News". The model is hosted on Hugging Face, and the API is built with Flask and deployed on Render.</p>
                <p>It includes a user-friendly web interface, programmatic endpoints, and enhanced error handling and logging. Future improvements include integrating interpretability tools, advanced hyperparameter tuning, and richer data augmentation.</p>
                <a href="/" class="btn btn-primary">Back to Home</a>
            </div>
            {{ footer|safe }}
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    ''', navbar=NAVBAR, footer=FOOTER)

# Custom error handler for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Page Not Found</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 70px; }
                .container { max-width: 700px; text-align: center; }
            </style>
        </head>
        <body>
            {{ navbar|safe }}
            <div class="container mt-5">
                <h1 class="display-4">404</h1>
                <p class="lead">Oops! The page you are looking for does not exist.</p>
                <a href="/" class="btn btn-primary">Go to Home</a>
            </div>
            {{ footer|safe }}
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    ''', navbar=NAVBAR, footer=FOOTER), 404

# Custom error handler for 500 errors
@app.errorhandler(500)
def internal_server_error(e):
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Server Error</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding-top: 70px; }
                .container { max-width: 700px; text-align: center; }
            </style>
        </head>
        <body>
            {{ navbar|safe }}
            <div class="container mt-5">
                <h1 class="display-4">500</h1>
                <p class="lead">Internal Server Error. Please try again later.</p>
                <a href="/" class="btn btn-primary">Go to Home</a>
            </div>
            {{ footer|safe }}
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    ''', navbar=NAVBAR, footer=FOOTER), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
