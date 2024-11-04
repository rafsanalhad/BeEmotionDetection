from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained emotion model
emotion_model = load_model('model.keras')  # Ganti path sesuai lokasi model

def facecrop(image_path):
    # Load Haar Cascade untuk deteksi wajah
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face_path = 'cropped_face.jpg'
        cv2.imwrite(face_path, face)
        return face_path
    return None

@app.route('/')
def home():
    return "Selamat datang di API deteksi emosi!"

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

    file = request.files['file']
    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    # Crop wajah dari gambar yang diunggah
    cropped_face_path = facecrop(file_path)
    if not cropped_face_path:
        return jsonify({'error': 'Tidak ada wajah terdeteksi pada gambar'}), 400

    # Preprocess wajah untuk prediksi
    img = image.load_img(cropped_face_path, color_mode="grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # Prediksi emosi
    predictions = emotion_model.predict(x)[0]
    emotions = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']
    result = {emotion: round(float(predictions[i]) * 100, 1) for i, emotion in enumerate(emotions)}

    # Temukan emosi dengan persentase terbesar
    max_emotion = max(result, key=result.get)
    max_percentage = result[max_emotion]

    # Hapus file yang dibuat
    os.remove(file_path)
    if os.path.exists(cropped_face_path):
        os.remove(cropped_face_path)

    # Kembalikan hasil prediksi dalam format JSON
    return jsonify({'result': result, 'max_emotion': max_emotion, 'max_percentage': max_percentage})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)