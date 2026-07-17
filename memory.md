# Codebase Intelligence & Memory

> **Project:** Handwritten Digit Recognition
> **Role:** Codebase Intelligence Agent

---

## 📌 PHASE 1 — REPOSITORY DISCOVERY

### Root Structure
- `app.py`: Minimal Flask API server serving predictions.
- `app.ipynb`: Jupyter notebook demonstrating model loading and prediction.
- `best_model.h5`: Trained Keras model weights (HDF5 format).
- `requirements.txt`: Python dependencies.
- `README.md`: Setup and API documentation.

---

## 🛠️ PHASE 2 — TECHNOLOGY DETECTION

### Backend / API Framework
- **Core:** Python
- **Web Server:** Flask (Waitress/Gunicorn for prod)

### Machine Learning Stack
- **Deep Learning Framework:** Keras (TensorFlow backend implied)
- **Data Processing:** NumPy
- **Image Processing:** Pillow (PIL)
- **Dataset:** MNIST (used for sample generation in app)

---

## 🎯 PHASE 3 — PROJECT PURPOSE ANALYSIS

### Purpose & Business Problem
A microservice API designed to recognize handwritten digits (0-9). It solves the problem of digitizing handwritten numbers by exposing a lightweight HTTP endpoint that accepts images and returns the predicted number and confidence probabilities.

### User Workflow
1. Client sends a GET request to `/predict` for a sample MNIST evaluation.
2. Client sends a POST request with `multipart/form-data` (image file) or JSON (`base64` encoded image) to `/predict`.
3. The API processes the image and returns a JSON response containing the `predicted` label.

---

## 🏗️ PHASE 4 — ARCHITECTURE ANALYSIS

```text
[ Client (cURL / Postman / App) ]
       │ (REST / JSON / Multipart)
       ▼
[ Flask API (app.py) ]
       │
       ├─► [ Pillow / NumPy (Image Preprocessing) ]
       │        (Resize 28x28, Invert, Normalize)
       │
       ▼
[ Keras Model (best_model.h5) ]
       │
       ▼
[ Output Generation (Argmax & Probabilities) ]
       │
       ▼
[ JSON Response ]
```

---

## 🔄 PHASE 9 — DATA FLOW ANALYSIS

**Inference Data Flow:**
1. Base64 or Image File is received at the `/predict` endpoint.
2. Pillow (`PIL`) converts the image to Grayscale (`L`).
3. Image is resized to `28x28` using Lanczos resampling.
4. Converted to a NumPy array. Background is dynamically inverted if the mean pixel value > 127 (ensuring white digit on dark background).
5. Array is normalized (divided by 255.0) and reshaped to `(1, 28, 28, 1)`.
6. Keras `model.predict()` executes inference.
7. `np.argmax()` determines the highest probability class (0-9).
8. Results returned to user via JSON.
