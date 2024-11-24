from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Tambahkan ini
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import time
#import mysqlclient
app = Flask(__name__)

# Konfigurasi MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/db_dineserve'  # Ganti db_name dengan nama database Anda
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/' 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    profile_picture = db.Column(db.String(120), nullable=True) 

class Reservation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    time = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    guest_count = db.Column(db.Integer, nullable=False)
    table_preference = db.Column(db.String(100), nullable=True)

    user = db.relationship('User', backref=db.backref('reservations', lazy=True))

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.String(500), nullable=True)

    user = db.relationship('User', backref=db.backref('reviews', lazy=True))

with app.app_context():
    db.create_all()

CORS(app)
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

@app.route("/signup", methods=["GET", "POST"])
def signup():
    # Ambil data dari request
    username = request.json.get("username")
    email = request.json.get("email")
    password = request.json.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    # Cek apakah pengguna sudah ada
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "Email sudah terdaftar"}), 400

    # Hash password dengan metode yang valid
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    # Simpan data pengguna baru
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User created successfully!"}), 201

@app.route("/login", methods=["GET", "POST"])
def login():
    email = request.json.get("email")
    password = request.json.get("password")

    # Cari pengguna berdasarkan email
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        return jsonify({"id": user.id, "user": user.username})
    else:
        return jsonify({"error": "Email atau password salah"}), 400

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/profil", methods=["GET"])
def get_profile():
    try:
        # Mendapatkan data pengguna dari query string (misalnya: user_id=1)
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = User.query.filter_by(username=user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Mengembalikan informasi pengguna termasuk gambar profil (jika ada)
        profile_picture = user.profile_picture if user.profile_picture else 'default.jpg'
        return jsonify({
            "username": user.username,
            "email": user.email,
            "profile_picture": f"{profile_picture}"  # Path gambar profil
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/profil/update", methods=["POST"])
def update_profile():
    try:
        # Ambil data dari form
        user_id = request.form.get('user_id')
        username = request.form.get('username')
        email = request.form.get('email')
        new_password = request.form.get('password')
        
        # Validasi User ID
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Cari user berdasarkan username atau id
        user = User.query.filter(
            (User.username == user_id) | (User.id == user_id)
        ).first()
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Cek apakah username baru sudah digunakan (jika username diubah)
        if username and username != user.username:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                return jsonify({"error": "Username already taken"}), 400

        # Cek apakah email baru sudah digunakan (jika email diubah)
        if email and email != user.email:
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                return jsonify({"error": "Email already taken"}), 400

        # Update data profil
        if username:
            user.username = username
        if email:
            user.email = email
        if new_password:
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')

        # Handling file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Buat folder uploads jika belum ada
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Generate unique filename
                filename = secure_filename(f"profile_{user.id}_{int(time.time())}.{file.filename.rsplit('.', 1)[1].lower()}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Hapus file profil lama jika ada
                if user.profile_picture and user.profile_picture != 'default.jpg':
                    old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_picture)
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                
                # Simpan file baru
                file.save(file_path)
                user.profile_picture = filename

        # Simpan perubahan ke database
        db.session.commit()
        
        return jsonify({
            "message": "Profile updated successfully!",
            "user": {
                "username": user.username,
                "email": user.email,
                "profile_picture": user.profile_picture
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        print(f"Error updating profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Menyajikan gambar profil yang diupload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/reservations", methods=["POST"])
def create_reservation():
    try:
        user_id = request.json.get('user_id')
        date = request.json.get('date')
        time = request.json.get('time')
        name = request.json.get('name')
        phone = request.json.get('phone')
        email = request.json.get('email')
        guest_count = request.json.get('guest_count')
        table_preference = request.json.get('table_preference')

        if not user_id or not date or not time:
            return jsonify({"error": "User ID, date, and time are required"}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        new_reservation = Reservation(user_id=user_id, date=date, time=time, name = name, phone=phone, email=email, guest_count=guest_count, table_preference=table_preference)
        db.session.add(new_reservation)
        db.session.commit()

        return jsonify({"message": "Reservation created successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/reservations", methods=["GET"])
def get_reservations():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        reservations = Reservation.query.filter_by(user_id=user_id).all()
        if not reservations:
            return jsonify({"message": "No reservations found"}), 404
        result = [
            {"user_id": user_id, "date": r.date, "time": r.time, "name": r.name, "phone": r.phone, "email": r.email, "guest_count": r.guest_count, "table_preference": r.table_preference}
            for r in reservations
        ]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/review", methods=["POST"])
def create_review():
    try:
        # Ambil data review dari request
        user_id = request.json.get("user_id")
        rating = request.json.get("rating")
        comment = request.json.get("comment")

        if not user_id or not rating:
            return jsonify({"error": "User ID dan rating diperlukan"}), 400

        # Validasi rating
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating harus antara 1 hingga 5"}), 400

        # Simpan review ke dalam database
        new_review = Review(user_id=user_id, rating=rating, comment=comment)
        db.session.add(new_review)
        db.session.commit()

        return jsonify({"message": "Review berhasil ditambahkan!"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@app.route("/reviews", methods=["GET"])
def get_reviews():
    try:
        reviews = Review.query.all()  # Fetch all reviews from the database
        if not reviews:
            return jsonify({"message": "No reviews found"}), 404

        result = []
        for review in reviews:
            user = User.query.get(review.user_id)
            review_data = {
                "id": review.id,
                "user_id": review.user_id,
                "rating": review.rating,
                "comment": review.comment,
                "username": user.username,  # Include the username of the reviewer
                "email": user.email  # You can also include the email if needed
            }
            result.append(review_data)

        return jsonify({"reviews": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)