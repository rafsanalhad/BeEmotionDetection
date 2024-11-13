from flask import Flask, request, jsonify
from flask_cors import CORS  # Tambahkan ini
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)  # Tambahkan ini
emotion_model = load_model("model.keras")

def facecrop(image_path):
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Failed to load image")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No faces detected")
            return None
            
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_path = "cropped_face.jpg"
            cv2.imwrite(face_path, face)
            return face_path
            
        return None
    except Exception as e:
        print(f"Error in facecrop: {str(e)}")
        return None

@app.route("/")
def home():
    return "Selamat datang di API deteksi emosi!"

@app.route("/predict", methods=["POST"])
def predict_emotion():
    try:
        print("Request received")  # Debug log
        
        if "file" not in request.files:
            print("No file in request")  # Debug log
            return jsonify({"error": "Tidak ada file gambar yang diunggah"}), 400
            
        file = request.files["file"]
        if file.filename == '':
            print("No filename")  # Debug log
            return jsonify({"error": "Nama file kosong"}), 400

        # Simpan file dengan nama unik berdasarkan timestamp
        import time
        timestamp = int(time.time())
        file_path = f"uploaded_image_{timestamp}.jpg"
        
        print(f"Saving file to {file_path}")  # Debug log
        file.save(file_path)
        
        print("Processing image")  # Debug log
        cropped_face_path = facecrop(file_path)
        
        if not cropped_face_path:
            print("No face detected")  # Debug log
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": "Tidak ada wajah terdeteksi pada gambar"}), 400

        print("Loading cropped image")  # Debug log
        img = image.load_img(
            cropped_face_path, color_mode="grayscale", target_size=(48, 48)
        )
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        print("Making prediction")  # Debug log
        predictions = emotion_model.predict(x)[0]
        emotions = ["Marah", "Jijik", "Takut", "Senang", "Sedih", "Terkejut", "Netral"]
        result = {
            emotion: round(float(predictions[i]) * 100, 1)
            for i, emotion in enumerate(emotions)
        }
        
        max_emotion = max(result, key=result.get)
        max_percentage = result[max_emotion]

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)

        print(f"Prediction complete: {max_emotion} ({max_percentage}%)")  # Debug log
        return jsonify({
            "result": result,
            "max_emotion": max_emotion,
            "max_percentage": max_percentage
        })

    except Exception as e:
        print(f"Error in predict_emotion: {str(e)}")  # Debug log
        # Cleanup in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        if 'cropped_face_path' in locals() and os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)