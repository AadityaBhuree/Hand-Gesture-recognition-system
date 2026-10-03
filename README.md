# Handwritten Digit Recognition Neural Laboratory & API 🧠📝

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey.svg)
![Keras](https://img.shields.io/badge/Keras-3.12.2-red.svg)
![Tests](https://img.shields.io/badge/10_Tests-Passing-10B981.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive machine learning laboratory and production-ready microservice that performs **Handwritten Digit Recognition (0–9)**. Built with Flask, Keras, and Pillow, it provides both an **interactive HTML5 drawing canvas UI** for real-time in-browser digit sketching and an HTTP REST API accepting multipart image files and base64 data URIs.

---

## 🚀 Features

- **Interactive HTML5 Drawing Canvas:** Beautiful cybernetic dark-mode drawing canvas with adjustable brush sizes, touch support, clear/invert controls, synthetic glyph presets, and live Softmax probability distribution bars.
- **Decoupled Architecture:** Clean separation of concerns between API routing and web serving (`app.py`), ML inference (`services/recognizer.py`), and interactive templates (`templates/index.html`).
- **Flexible Inputs:** Accepts browser canvas strokes, `multipart/form-data` image file uploads, and JSON payloads containing base64 data URIs.
- **Robust Security & Validation:** Built-in CORS support, a strict `MAX_CONTENT_LENGTH` (5MB) upload ceiling, and graceful rejection of corrupted or non-image payloads.
- **Advanced Preprocessing:** Automatically resizes (Lanczos resampling), converts to grayscale, and dynamically inverts light-background images to match the MNIST training distribution (white digits on black backgrounds).
- **Offline Fault Tolerance:** Resilient sample digit generation with synthetic glyph fallback if dataset downloads are unavailable.
- **100% Passing Automated Tests:** 10 comprehensive `pytest` test cases covering endpoint status, content negotiation, canvas base64 inference, corrupted payloads, and payload limits.

---

## 📁 Project Structure

```text
├── models/
│   └── best_model.h5         # Pre-trained Keras CNN model weights
├── notebooks/
│   ├── MNIST_Train.ipynb     # Model training research notebook
│   └── app.ipynb             # Inference demonstration notebook
├── services/
│   └── recognizer.py         # ML Inference & Preprocessing logic
├── templates/
│   └── index.html            # Interactive HTML5 Drawing Canvas UI
├── tests/
│   └── test_api.py           # Pytest suite (10 automated unit tests)
├── app.py                    # Flask server, canvas UI router, and REST API
├── pytest.ini                # Pytest configuration
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
pytest
```
This will execute all 10 automated unit tests, validating:
- HTML5 canvas UI serving and content negotiation (`Accept: application/json` vs. `text/html`)
- API schema endpoints (`GET /api`)
- Sample MNIST image inference and synthetic fallback (`GET /predict`)
- Multipart form file uploads with format validation
- Corrupted file rejection (graceful 400 Bad Request)
- Base64 data URI inference with 10-class probability distribution
- Malformed base64 and corrupted payload rejection
- 5MB maximum payload protection (413 Request Entity Too Large)

---

## ⚠️ Notes & Requirements
- **Model Requirement**: You must have a compatible Keras `.h5` model placed at `models/best_model.h5` for the service to successfully run predictions. If it is missing, the service will start but all `/predict` routes will return `503 Service Unavailable`.
