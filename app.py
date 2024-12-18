from flask import Flask, request, jsonify, send_from_directory, render_template
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
from flask_midtrans import Midtrans
from dotenv import load_dotenv
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
#import mysqlclient
load_dotenv()

app = Flask(__name__, template_folder='./templates')

# Konfigurasi dari .env
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS') == 'True'
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['ALLOWED_EXTENSIONS'] = set(os.getenv('ALLOWED_EXTENSIONS').split(','))

app.config['MIDTRANS_CLIENT_KEY'] = os.getenv('MIDTRANS_CLIENT_KEY')
app.config['MIDTRANS_SERVER_KEY'] = os.getenv('MIDTRANS_SERVER_KEY')
app.config['MIDTRANS_IS_PRODUCTION'] = os.getenv('MIDTRANS_IS_PRODUCTION') == 'True'

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

db = SQLAlchemy(app)
mail = Mail(app)
serializer = URLSafeTimedSerializer(os.getenv('SECRET_KEY'))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    profile_picture = db.Column(db.String(120), nullable=True)
    role = db.Column(db.String(50), nullable=True) 

class Reservation(db.Model):
    id = db.Column(db.String(100), nullable=False, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    time = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    guest_count = db.Column(db.Integer, nullable=False)
    transaction_id = db.Column(db.String(100), nullable=False)
    table_id = db.Column(db.Integer, db.ForeignKey('table.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('reservations', lazy=True))

class Transaction(db.Model):
    transaction_id = db.Column(db.String(100), primary_key=True, unique=True, nullable=False)
    reservation_id = db.Column(db.String(100), db.ForeignKey('reservation.id'), nullable=False)
    transaction_time = db.Column(db.DateTime, nullable=False)
    transaction_status = db.Column(db.String(50), nullable=False)
    payment_type = db.Column(db.String(50), nullable=False)
    order_id = db.Column(db.String(100), unique=True, nullable=False)
    status = db.Column(db.String(50), nullable=False)

    reservation = db.relationship('Reservation', backref=db.backref('payment', lazy=True))

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.String(500), nullable=True)
    date= db.Column(db.DateTime, nullable=False, default=datetime.now)

    user = db.relationship('User', backref=db.backref('reviews', lazy=True))


class Table(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    table_number = db.Column(db.String(10), unique=True, nullable=False) 
    capacity = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(50), nullable=True)
    reservations = db.relationship('Reservation', backref='table', lazy=True)

class Refund(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reservation_id = db.Column(db.String(100), nullable=False)
    transaction_id = db.Column(db.String(100), unique=True, nullable=False)
    transaction_time = db.Column(db.DateTime, nullable=False)
    transaction_status = db.Column(db.String(50), nullable=False)
    payment_type = db.Column(db.String(50), nullable=False)
    order_id = db.Column(db.String(100), unique=True, nullable=False)
    status = db.Column(db.String(50), nullable=False)

    reservation = db.relationship('User', backref=db.backref('refund', lazy=True))

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), nullable=False, default="unread")  # status: 'unread' or 'read'
    timestamp = db.Column(db.DateTime, default=datetime.now)

    user = db.relationship('User', backref=db.backref('notifications', lazy=True))

def seed_tables():
    if Table.query.count() > 0:
        print("Data sudah ada di tabel, tidak menambahkan data baru.")
        return

    tables = [
        {"table_number": "A1", "capacity": 6, "location": "Indoor - Near Entrance"},
        {"table_number": "A2", "capacity": 6, "location": "Indoor - Center of Hall"},
        {"table_number": "A3", "capacity": 6, "location": "Indoor - Corner Section"},
        {"table_number": "B1", "capacity": 6, "location": "Outdoor - Near Garden"},
        {"table_number": "B2", "capacity": 6, "location": "Outdoor - Patio Section"},
        {"table_number": "B3", "capacity": 6, "location": "Outdoor - Near Fountain"},
        {"table_number": "C1", "capacity": 6, "location": "Near Window - Garden View"},
        {"table_number": "C2", "capacity": 6, "location": "Near Window - Street View"},
        {"table_number": "D1", "capacity": 6, "location": "Private Room - VIP Section"},
        {"table_number": "D2", "capacity": 6, "location": "Private Room - Family Section"},
    ]

    for table_data in tables:
        table = Table(
            table_number=table_data["table_number"],
            capacity=table_data["capacity"],
            location=table_data["location"]
        )
        db.session.add(table)

    db.session.commit()
    print("10 tables have been added to the database.")


with app.app_context():
    db.create_all()
    try:
        seed_tables()
    except Exception as e:
        print(f"Seeder error: {e}")


CORS(app)

def load_emotion_model(model_path):
    """
    Memuat model yang telah dilatih
    """
    return load_model(model_path)

def get_emotion_prediction(model, image):
    """
    Memprediksi emosi dari gambar
    """
    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Preprocess gambar
    preprocessed_image = preprocess_image(image)

    # Set steps_per_execution to 1 before prediction
    model.steps_per_execution = 1

    # Prediksi
    predictions = model.predict(preprocessed_image)

    # Ambil indeks dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions[0])

    # Ambil label emosi dan probabilitasnya
    emotion = class_labels[predicted_class]
    probability = predictions[0][predicted_class]

    return emotion, probability

def preprocess_image(image):
    """
    Memproses gambar untuk prediksi
    """
    # Pastikan gambar dalam mode grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize ke ukuran 48x48
    resized = cv2.resize(gray, (48, 48))
    
    # Normalisasi
    normalized = resized / 255.0
    
    # Reshape untuk model
    preprocessed = normalized.reshape((1, 48, 48, 1))
    return preprocessed

# instance
midtrans = Midtrans(app)

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
        
        # Crop wajah menggunakan fungsi asli
        cropped_face_path = facecrop(file_path)
        
        if cropped_face_path is None:
            print("No faces detected")  # Debug log
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": "Tidak ada wajah terdeteksi pada gambar"}), 400
        
        # Muat model
        model = load_emotion_model("best_emotion_model_v2.keras")  # Sesuaikan path model jika berbeda
        
        # Baca gambar yang dicrop
        cropped_img = cv2.imread(cropped_face_path)
        
        # Prediksi emosi
        emotion, probability = get_emotion_prediction(model, cropped_img)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)

        print(f"Prediction complete: {emotion} ({probability:.2f})")  # Debug log
        return jsonify({
            "max_emotion": emotion,
            "max_percentage": round(float(probability) * 100, 1)
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
    existing_username = User.query.filter_by(username=username).first()
    if existing_username:
        return jsonify({"error": "Username sudah terdaftar"}), 400

    # Hash password dengan metode yang valid
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    # Simpan data pengguna baru
    new_user = User(username=username, email=email, password=hashed_password, role='member')
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
        return jsonify({"id": user.id, "user": user.username, "role": user.role}), 200
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
        id= request.json.get('id')
        user_id = request.json.get('user_id')
        date = request.json.get('date')
        time = request.json.get('time')
        name = request.json.get('name')
        phone = request.json.get('phone')
        email = request.json.get('email')
        guest_count = request.json.get('guest_count')
        table_id = request.json.get('table_id')
        transaction_id = request.json.get('transaction_id')

        if not user_id or not date or not time:
            return jsonify({"error": "User ID, date, and time are required"}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        new_reservation = Reservation(id=id, user_id=user_id, date=date, time=time, name = name, phone=phone, email=email, guest_count=guest_count, table_id=table_id, transaction_id=transaction_id)
        new_notification = Notification(user_id=user_id, message=f"Reservation telah berhasil di tanggal {date} dan jam {time}. Reservasi ID: {id}")
        db.session.add(new_reservation)
        db.session.add(new_notification)
        db.session.commit()

        return jsonify({"message": "Reservation created successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getAllTables', methods=['GET'])
def get_all_tables():
    tables = Table.query.all()

    if not tables:
        return jsonify({'message': 'No tables found'}), 404

    tables_data = []
    for table in tables:
        table_data = {
            'id': table.id,
            'table_number': table.table_number,
            'capacity': table.capacity,
            'location': table.location
        }
        tables_data.append(table_data)

    return jsonify({'tables': tables_data}), 200
    
@app.route("/reservations", methods=["GET"])
def get_reservations():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        if user.role == 'admin':
            # Sort descending by date or id
            reservations = Reservation.query.order_by(Reservation.date.desc()).all()
        else:
            reservations = Reservation.query.filter_by(user_id=user_id).order_by(Reservation.date.desc()).all()

        reservations_data = [
            {   
                "id": r.id,
                "user_id": r.user_id,
                "username": r.user.username,
                "date": r.date,
                "time": r.time,
                "name": r.name,
                "phone": r.phone,
                "email": r.email,
                "guest_count": r.guest_count,
                "table_id": r.table_id,
                "table_number": r.table.table_number,
            }
            for r in reservations
        ]
        return jsonify(reservations_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/review", methods=["POST"])
def create_review():
    try:
        # Ambil data review dari request
        user_id = request.json.get("user_id")
        rating = request.json.get("rating")
        comment = request.json.get("comment")
        date= datetime.now()

        if not user_id or not rating:
            return jsonify({"error": "User ID dan rating diperlukan"}), 400

        # Validasi rating
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating harus antara 1 hingga 5"}), 400

        # Simpan review ke dalam database
        new_review = Review(user_id=user_id, rating=rating, comment=comment, date=date)
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
                "email": user.email,  # You can also include the email if needed
                "date": review.date
            }
            result.append(review_data)

        return jsonify({"reviews": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/payment', methods=["POST"])
def hello_world():
    id = request.json.get("id")
    production_name = request.json.get("production_name")
    price = request.json.get("price")
    param = {
        "transaction_details": {
            "order_id": id,
            "gross_amount": price
        }, "credit_card": {
            "secure": True
        }
    }

    # midtrans.snap or midtrans.core
    # https://github.com/Midtrans/midtrans-python-client
    response = midtrans.snap.create_transaction(param)

    response_data = {
        'token': response['token'],
    }
    # >> response
    #  {'token': 'thistoken', 'redirect_url': 'http://midtrans..'}
    return jsonify(response_data), 200

@app.route('/payment/finish', methods=['POST'])
def payment_finish():
    if request.method == 'POST':
        # Tangani data callback dari Midtrans
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400
        
        # Ambil data penting dari JSON
        transaction_status = data.get('transaction_status')
        transaction_id = data.get('transaction_id')
        transaction_time = datetime.now()
        reservation_id = data.get('reservation_id')
        payment_type = data.get('payment_type')
        order_id = data.get('order_id')
        status = data.get('status')

        # Validasi data yang diperlukan
        if not transaction_id or not order_id:
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

        try:
            # Simpan transaksi ke database
            new_transaction = Transaction(
                transaction_id=transaction_id,
                transaction_time=transaction_time,
                reservation_id=reservation_id,
                transaction_status=transaction_status,
                payment_type=payment_type,
                order_id=order_id,
                status=status,
            )
            db.session.add(new_transaction)
            db.session.commit()

            return jsonify({'status': 'success', 'message': 'Transaction saved successfully'}), 200

        except Exception as e:
            db.session.rollback()
            # Menambahkan traceback untuk membantu debugging
            print("Error occurred during transaction processing:")
            print(str(e))  # Log error message
            return jsonify({'status': 'error', 'message': str(e)}), 500

    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@app.route('/check_reservation', methods=['POST'])
def check_reservation():
    data = request.get_json()

    if not all(key in data for key in ['date', 'time', 'table_id']):
        return jsonify({'error': 'Missing required parameters'}), 400

    date = data['date']
    time = data['time']
    table_id = data['table_id']

    reservation = Reservation.query.filter_by(date=date, time=time, table_id=table_id).first()

    if reservation:
        return jsonify({'message': 'Sorry, this table is already reserved at this time and date.'}), 400
    else:
        return jsonify({'message': 'Table is available for reservation.'}), 200

@app.route('/reservations/<string:reservation_id>', methods=['DELETE'])
def delete_reservation(reservation_id):
    try:
        print(reservation_id)
        # Cari reservasi berdasarkan ID
        reservation = Reservation.query.get(reservation_id)
        if not reservation:
            return jsonify({"message": "Reservation not found"}), 404

        # Validasi keberadaan transaksi
        if not reservation.transaction_id:
            return jsonify({"message": "No transaction associated with this reservation"}), 400
        
        transaction = Transaction.query.get(reservation.transaction_id)
        if not transaction:
            return jsonify({"message": "Transaction not found"}), 404

        new_notification = Notification(user_id=reservation.user_id, message=f"Reservasi {reservation_id} telah berhasil dibatalkan. Silahkan menunggu informasi pengembalian dana")

        # Buat refund baru
        new_refund = Refund(
            transaction_id=transaction.transaction_id,
            reservation_id=reservation_id,
            user_id=reservation.user_id,
            transaction_time=transaction.transaction_time,
            transaction_status=transaction.transaction_status,
            payment_type=transaction.payment_type,
            order_id=transaction.order_id,
            status="Belum diproses",
        )
        db.session.add(new_refund)
        db.session.add(new_notification) 

        # Hapus reservasi dan transaksi
        db.session.delete(transaction)
        db.session.delete(reservation)
        db.session.commit()

        return jsonify({"message": "Reservation cancelled successfully"}), 200

    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

@app.route('/refunds', methods=['GET'])
def get_all_refunds():
    try:
        # Mengurutkan data refund secara descending berdasarkan transaction_time
        refunds = Refund.query.order_by(Refund.transaction_time.desc()).all()

        result = []
        for refund in refunds:
            result.append({
                'id': refund.id,
                'user_id': refund.user_id,
                'reservation_id': refund.reservation_id,
                'transaction_id': refund.transaction_id,
                'transaction_time': refund.transaction_time,
                'transaction_status': refund.transaction_status,
                'payment_type': refund.payment_type,
                'order_id': refund.order_id,
                'status': refund.status
            })

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/refunds/<int:id>', methods=['GET', 'PUT'])
def manage_refund(id):
    refund = Refund.query.get(id)
    if not refund:
        return jsonify({'error': 'Refund not found'}), 404

    if request.method == 'GET':
        return jsonify({
            'id': refund.id,
            'user_id': refund.user_id,
            'reservation_id': refund.reservation_id,
            'transaction_id': refund.transaction_id,
            'transaction_time': refund.transaction_time,
            'transaction_status': refund.transaction_status,
            'payment_type': refund.payment_type,
            'order_id': refund.order_id,
            'status': refund.status
        }), 200

    if request.method == 'PUT':
        data = request.json
        if 'status' not in data or data['status'] not in ['Diterima', 'Ditolak']:
            return jsonify({'error': 'Invalid status'}), 400
        
        if refund.status != 'Belum diproses':
            return jsonify({'error': 'Refund has already been processed'}), 400

        refund.status = data['status']
        print('Refund status:', refund.status)
        new_notifcation = Notification(user_id=refund.user_id, message=f"Refund dengan id {refund.id} telah {refund.status}")
        db.session.add(new_notifcation)
        db.session.commit()
        return jsonify({'message': f'Refund {data["status"]} successfully'}), 200
    
@app.route('/notifications/<int:id>', methods=['GET'])
def get_notification_by_id(id):
    try:
        # Mengurutkan data notifikasi secara descending berdasarkan timestamp
        notifications = Notification.query.filter_by(user_id=id).order_by(Notification.timestamp.desc()).all()

        if not notifications:
            return jsonify({'error': 'Notifications not found'}), 404

        notification_data = []
        for notification in notifications:
            notificationTemp = {
                'id': notification.id,
                'user_id': notification.user_id,
                'message': notification.message,
                'status': notification.status,
                'timestamp': notification.timestamp
            }
            notification_data.append(notificationTemp)

        return jsonify({'notif': notification_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Forgot Password Endpoint
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    if not request.is_json:
        return jsonify({'error': 'Request must be in JSON format.'}), 400
    
    data = request.json
    if not data:
        return jsonify({'error': 'Request body is missing.'}), 400

    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email is required.'}), 400

    # user = User.query.filter_by(email=email).first()
    # if not user:
    #     return jsonify({'error': 'User with this email does not exist.'}), 404

    # Generate password reset token
    token = serializer.dumps(email, salt='password-reset-salt')
    reset_url = f"{os.getenv('BASE_URL')}/reset-password/{token}"

    # Send email
    try:
        msg = Message('Password Reset Request', sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Click the link to reset your password: {reset_url}"
        mail.send(msg)
        return jsonify({'message': 'Password reset email sent.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to send email: {str(e)}'}), 500


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'GET':
        return render_template('reset_password_form.html', token=token, message=None, error=None)
    
    if request.method == 'POST':
        try:
            # Verifikasi token
            email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
        except Exception:
            # Jika token tidak valid
            return render_template('reset_password_form.html', token=token, message=None, error="Invalid or expired token.")
        
        data = request.form
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        # Validasi password lama
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, old_password):
            return render_template('reset_password_form.html', token=token, message=None, error="Old password is incorrect.")
        
        # Validasi password baru
        if not new_password:
            return render_template('reset_password_form.html', token=token, message=None, error="New password is required.")

        # Update password pengguna
        user.password = generate_password_hash(new_password)
        db.session.commit()

        # Return success message if password reset is successful
        return render_template('reset_password_form.html', token=token, message="Password reset successfully.", error=None)





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)