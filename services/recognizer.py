import os
import numpy as np
from PIL import Image

class DigitRecognizerService:
    def __init__(self, model_path='models/best_model.h5'):
        self.model = None
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        """Loads the Keras model."""
        if os.path.exists(self.model_path):
            try:
                # Import keras here to avoid slow startup if service isn't used immediately
                from keras.models import load_model
                self.model = load_model(self.model_path)
                print(f"Model loaded from {self.model_path}.")
            except Exception as e:
                print(f"Failed loading model: {e}")
        else:
            print(f"Model file {self.model_path} not found; predictions will be unavailable.")

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Preprocesses the PIL Image for the MNIST model."""
        # Ensure image is grayscale
        if img.mode != 'L':
            img = img.convert('L')
            
        # Resize to 28x28 using high-quality downsampling
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img).astype('float32')
        
        # If the background is light, invert it (MNIST expects white digit on black background)
        # Using a simple median check or mean check
        if arr.mean() > 127:
            arr = 255 - arr
            
        # Normalize and reshape to match model input shape (batch_size, height, width, channels)
        arr = arr / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr

    def predict(self, img: Image.Image) -> dict:
        """Runs inference on the provided image and returns the prediction."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
            
        x = self.preprocess_image(img)
        preds = self.model.predict(x)
        label = int(np.argmax(preds, axis=1)[0])
        probs = preds[0].tolist()
        
        return {
            'predicted': label,
            'probabilities': probs
        }
