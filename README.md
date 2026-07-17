# Handwritten Digit Recognition API 🧠📝

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey.svg)
![Keras](https://img.shields.io/badge/Keras-3.12.2-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready, lightweight machine learning microservice that performs **Handwritten Digit Recognition (0-9)**. Built with Flask, Keras, and Pillow, it provides a robust HTTP API to process image uploads and base64-encoded images, running them through a trained Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 🚀 Features

- **Decoupled Architecture:** Clean separation of concerns between API routing (`app.py`) and Machine Learning inference (`services/recognizer.py`).
- **Flexible Inputs:** Accepts both `multipart/form-data` image file uploads and JSON payloads containing `base64` image data URIs.
- **Robust Security:** Built-in CORS support and a strict `MAX_CONTENT_LENGTH` (5MB) limit to prevent Out-Of-Memory (OOM) attacks from massive file uploads.
- **Advanced Preprocessing:** Automatically resizes (Lanczos resampling), converts to grayscale, and dynamically inverts light-background images to match the MNIST training distribution (white digits on black backgrounds).
- **Comprehensive Error Handling:** Graceful failure handling with detailed JSON error messages and appropriate HTTP status codes (400, 413, 500, 503).
- **Test Coverage:** Includes a suite of `pytest` unit tests ensuring API stability.

---

## 📁 Project Structure

```text
├── models/
│   └── best_model.h5         # Pre-trained Keras model weights
├── notebooks/
│   ├── MNIST_Train.ipynb     # Model training research notebook
│   └── app.ipynb             # Inference demonstration notebook
├── services/
│   └── recognizer.py         # ML Inference & Preprocessing logic
├── tests/
│   └── test_api.py           # Pytest suite for API endpoints
├── app.py                    # Flask API Server and routing
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Development & Testing dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Hand_written_Digit_recognization
```

### 2. Environment Setup (Windows PowerShell)
Create and activate a virtual environment:
```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
```

*(For Linux/macOS)*:
```bash
python3 -m venv .env
source .env/bin/activate
```

### 3. Install Dependencies
For **Production** only (lightweight, no Jupyter/Matplotlib):
```bash
pip install -r requirements.txt
```

For **Development & Testing** (includes pytest, notebooks, etc.):
```bash
pip install -r requirements-dev.txt
```

---

## 🚦 Running the Application

### Local Development
To run the Flask development server:
```bash
python app.py
```
The API will be available at `http://0.0.0.0:5000`.

### Production Deployment
Do not use the built-in Flask server in production. Use a WSGI server like Gunicorn (Linux/macOS):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```
*Note for Windows users:* Gunicorn is not supported on Windows. Use `waitress` instead:
```bash
pip install waitress
waitress-serve --listen=0.0.0.0:5000 app:app
```

---

## 🌐 API Reference

### `GET /`
Returns API metadata and available endpoints.

### `GET /predict`
Runs inference on a random sample image from the MNIST test dataset (useful for health checks).
**Response:**
```json
{
  "predicted": 7,
  "ground_truth": 7,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0]
}
```

### `POST /predict`
Process a user-provided image. 

#### Option A: File Upload (Form Data)
Use `multipart/form-data` with the key `file`.
```bash
curl -F "file=@/path/to/digit.png" http://localhost:5000/predict
```

#### Option B: JSON Base64
Send a JSON payload with the key `image` containing a base64 string or data URI.
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image": "data:image/png;base64,iVBORw0KGgo..."}' \
     http://localhost:5000/predict
```

**Response:**
```json
{
  "predicted": 3,
  "probabilities": [0.01, 0.0, 0.0, 0.95, 0.0, 0.01, 0.0, 0.03, 0.0, 0.0]
}
```

---

## 🧪 Testing

This project uses `pytest` for unit testing. Ensure you have installed the development requirements (`requirements-dev.txt`), then run:

```bash
pytest tests/
```
This will execute all endpoint tests, including large payload rejection and malformed base64 handling.

---

## ⚠️ Notes & Requirements
- **Model Requirement**: You must have a compatible Keras `.h5` model placed at `models/best_model.h5` for the service to successfully run predictions. If it is missing, the service will start but all `/predict` routes will return `503 Service Unavailable`.
