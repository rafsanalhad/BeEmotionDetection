# app.py
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
from pathlib import Path

app = Flask(__name__)

# Gunakan path relatif untuk model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.keras'
emotion_model = load_model(str(MODEL_PATH))

def get_temp_path(filename):
    temp_dir = Path("/tmp")
    temp_dir.mkdir(exist_ok=True)
    return str(temp_dir / filename)

def facecrop(image_path):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
        
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face_path = get_temp_path('cropped_face.jpg')
        cv2.imwrite(face_path, face)
        return face_path
    return None

@app.route('/')
def home():
    return "Selamat datang di API deteksi emosi!"

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400

        file_path = get_temp_path('uploaded_image.jpg')
        file.save(file_path)

        cropped_face_path = facecrop(file_path)
        if not cropped_face_path:
            return jsonify({'error': 'Tidak ada wajah terdeteksi pada gambar'}), 400

        img = image.load_img(cropped_face_path, color_mode="grayscale", target_size=(48, 48))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        predictions = emotion_model.predict(x)[0]
        emotions = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
        result = {emotion: round(float(predictions[i]) * 100, 1) for i, emotion in enumerate(emotions)}

        max_emotion = max(result, key=result.get)
        max_percentage = result[max_emotion]

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)

        return jsonify({
            'result': result,
            'max_emotion': max_emotion,
            'max_percentage': max_percentage
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel membutuhkan app sebagai handler
app.debug = False