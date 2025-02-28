# Fake News Detector

A state-of-the-art fake news detection API built with Flask and DistilBERT. This project uses a fine-tuned DistilBERT model to classify news articles as "Fake News" or "Real News". The model is hosted on Hugging Face, and the Flask API is deployed on Render.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Testing](#local-testing)
  - [Deployed Service](#deployed-service)
  - [API Endpoints](#api-endpoints)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

## Overview

This project addresses the challenge of detecting fake news using modern NLP techniques. It fine-tunes a pre-trained DistilBERT model on a dataset of fake and real news articles. The resulting model is accessible via a REST API built with Flask.

## Features

- Fine-tuned DistilBERT model for fake news detection
- REST API built with Flask
- Model hosted on Hugging Face Hub
- Deployed on Render for public access
- Error analysis and logging integrated for model evaluation

## Technologies

- **Python 3.x**
- **Flask** – Web framework for building the API
- **Transformers** – Hugging Face library for DistilBERT
- **Torch** – For model inference
- **Git** – For version control
- **Render** – Cloud hosting platform
- **Hugging Face Hub** – Hosting the model files

## Installation

### Prerequisites

- Python 3.x installed
- Git installed
- A [Hugging Face account](https://huggingface.co/) for hosting the model (already set up)
- A [Render account](https://render.com/) if you want to deploy the service publicly

### Setup Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/krishna31102004/fakenewsdetector.git
   cd fakenewsdetector
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Update `app.py`** if needed to reference your Hugging Face model:

   ```python
   model_path = "kb4086/fakenewsdetector"  # Hugging Face repo ID
   tokenizer = DistilBertTokenizer.from_pretrained(model_path)
   model = DistilBertForSequenceClassification.from_pretrained(model_path)
   model.eval()
   ```

## Usage

### Local Testing

1. **Run the Flask app:**

   ```bash
   python app.py
   ```
   By default, it listens on `0.0.0.0` at a port specified by `PORT` (or 5000 if not set).

2. **Send a POST request to `/predict`:**

   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Breaking news: The President just announced new policies."}'
   ```

   You should get a JSON response:

   ```json
   {"prediction": "Fake News"}
   ```

### Deployed Service

The project is **live** at [Render](https://render.com/).  
You can access the public API here:
```
https://fakenewsdetector-pg8i.onrender.com
```
- **Root Route** (`GET /`): Shows a simple welcome message.
- **`POST /predict`**: Accepts a JSON payload with `"text"` and returns a prediction.

**Example**:
```bash
curl -X POST https://fakenewsdetector-pg8i.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The President just announced new policies."}'
```
Expected output:
```json
{"prediction":"Fake News"}
```

### API Endpoints

- **`GET /`**  
  Returns a brief welcome message.

- **`POST /predict`**  
  Accepts JSON with a `"text"` field, returns `{"prediction": "Fake News"}` or `{"prediction": "Real News"}`.

## Results

Preliminary evaluation shows high accuracy and F1 scores. Error analysis suggests the model handles a variety of political and general news articles well.

## Future Improvements

- **Better Explainability**: Integrate tools like LIME or SHAP to see which words influence predictions.
- **Hyperparameter Exploration**: Experiment with different learning rates, batch sizes, and epochs to see if performance can improve.
- **Data Augmentation**: Include more diverse or domain-specific datasets for broader coverage.
- **UI Enhancements**: Create a small web interface using Streamlit or Gradio for a more user-friendly demo.
- **Monitoring & Logging**: Add advanced logging or analytics to track usage and model performance in production.

## Contact

- **Name**: Krishna Balaji
- **GitHub**: [krishna31102004](https://github.com/krishna31102004)
- **Hugging Face**: [kb4086](https://huggingface.co/kb4086)
- **Email**: *kbalaji6@asu.edu*

