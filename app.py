from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import io
import base64
from PIL import Image

# Import the new service
from services.recognizer import DigitRecognizerService

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Limit upload size to 5MB to prevent OOM attacks
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Initialize the recognizer service
recognizer = DigitRecognizerService()

# We still need the sample for the GET /predict route
# We load it lazily to avoid loading keras dataset on startup if not needed
SAMPLE_IMG = None
SAMPLE_LABEL = None

def get_sample_image():
    global SAMPLE_IMG, SAMPLE_LABEL
    if SAMPLE_IMG is None:
        try:
            from keras.datasets import mnist  # pyright: ignore[reportMissingImports] # type: ignore
            (_, _), (x_test, y_test) = mnist.load_data()
            SAMPLE_IMG = x_test[0]
            SAMPLE_LABEL = int(y_test[0])
        except Exception as e:
            print(f"Failed to load MNIST sample from keras.datasets: {e}")
            # Fallback to an offline synthetic 28x28 sample digit (e.g. digit 7)
            import numpy as np
            synthetic = np.zeros((28, 28), dtype=np.uint8)
            synthetic[5:7, 6:22] = 255  # top bar
            for i in range(16):
                synthetic[7 + i, 20 - i] = 255  # diagonal
            SAMPLE_IMG = synthetic
            SAMPLE_LABEL = 7
    return SAMPLE_IMG, SAMPLE_LABEL


@app.route('/')
def index():
    accept = request.headers.get('Accept', '')
    if request.args.get('format') == 'json' or ('application/json' in accept and 'text/html' not in accept):
        return api_spec()
    return render_template('index.html')


@app.route('/api')
def api_spec():
    return jsonify({
        'message': 'Handwritten Digit Recognition API',
        'endpoints': {
            'GET /': 'Interactive HTML5 Drawing Canvas and Telemetry UI.',
            'GET /api': 'Returns this API specification.',
            'GET /predict': 'Returns prediction for a sample MNIST image.',
            'POST /predict': 'Submit an image file (form-data "file") or JSON with a base64 string under key "image".'
        }
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        if recognizer.model is None:
            return jsonify({'error': 'Model not loaded.'}), 503
            
        img_arr, label = get_sample_image()
        if img_arr is None:
            return jsonify({'error': 'Failed to load sample image data.'}), 500
            
        try:
            # Convert the sample numpy array to a PIL Image so we can use our service
            img = Image.fromarray(img_arr).convert('L')
            result = recognizer.predict(img)
            return jsonify({
                'predicted': result['predicted'],
                'ground_truth': label,
                'probabilities': result['probabilities']
            })
        except Exception as e:
            return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

    # POST Method
    if recognizer.model is None:
        return jsonify({'error': 'Model not loaded. Service unavailable.'}), 503

    img = None
    
    # 1. Handle file upload (multipart/form-data)
    if 'file' in request.files:
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No selected file.'}), 400
        try:
            img = Image.open(f.stream).convert('L')
        except Exception as e:
            return jsonify({'error': 'Cannot process the uploaded file as an image.', 'details': str(e)}), 400
            
    # 2. Handle JSON payload with base64 string
    elif request.is_json or request.form:
        data = request.get_json(silent=True) or request.form.to_dict()
        if data and 'image' in data:
            b64 = data['image']
            
            # Strip data URI header if present
            if b64.startswith('data:'):
                try:
                    b64 = b64.split(',', 1)[1]
                except IndexError:
                    return jsonify({'error': 'Invalid base64 string format (missing comma in data URI).'}), 400
                    
            try:
                decoded = base64.b64decode(b64, validate=True)
                img = Image.open(io.BytesIO(decoded)).convert('L')
            except Exception as e:
                return jsonify({'error': 'Failed to decode base64 image.', 'details': str(e)}), 400

    if img is None:
        return jsonify({'error': "No image provided. Send multipart 'file' or JSON 'image' base64."}), 400

    # Run inference via the service
    try:
        result = recognizer.predict(img)
        return jsonify({
            'predicted': result['predicted'],
            'probabilities': result['probabilities']
        })
    except Exception as e:
        # In a real app, log the exception traceback here
        print(f"Inference error: {e}")
        return jsonify({'error': 'Failed to process image during inference.', 'details': str(e)}), 500

# Global error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 5MB.'}), 413

if __name__ == '__main__':
    # Using a production-ready approach, debug is False
    app.run(host='0.0.0.0', port=5000, debug=False)
