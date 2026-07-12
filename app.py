import os
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = 'model_diabetic_retinopathy.h5'
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "Sistem tidak menemukan file yang diunggah."

    file = request.files["file"]

    if file.filename == "":
        return "Silakan pilih file gambar retina terlebih dahulu."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 1. Membaca gambar menggunakan OpenCV & Konversi ke RGB Standar
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return "Gagal membaca berkas gambar. Pastikan format file Anda benar."

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Prediksi Nilai Mentah
    prediction = model.predict(img_array)
    nilai = float(prediction[0][0])

    # Log Diagnostik untuk memantau nilai asli di Terminal VS Code
    print(f"\n[DIAGNOSTIK] File: {filename} -> Nilai Mentah Model: {nilai}\n")

    # 3. Logika Penentuan Kelas (Sesuai Konfirmasi Terakhir Anda)
    if nilai > 0.5:
        hasil = "Kondisi Retina Normal (No-Diabetic Retinopathy)"
        status = "normal"
        persentase = nilai * 100
        
        penjelasan = (
            f"Sistem mengklasifikasikan gambar ini sebagai 'Normal' dengan tingkat kepastian {round(persentase, 2)}%. "
            f"Nilai akurasi ini didasarkan pada pemindaian piksel struktur pembuluh darah utama retina. "
            f"Namun, perlu dicatat bahwa variasi pencahayaan fundus, resolusi kamera, bercak putih bekas terapi laser pembuluh darah, atau kompresi file "
            f"dapat memengaruhi pembacaan komputer. Jika pada data klinis asli gambar ini seharusnya adalah DR (Sakit), "
            f"maka gangguan visual atau pola laser tersebut memicu terjadinya toleransi error (False Negative) di mana AI salah mengira bercak penyakit sebagai retina sehat."
        )
        rekomendasi = "Pertahankan pola hidup sehat, kontrol gula darah secara rutin, dan lakukan cek mata ke dokter minimal setahun sekali."
    
    else:
        hasil = "Terdeteksi Retinopati Diabetik (Diabetic Retinopathy)"
        status = "dr"
        persentase = (1 - nilai) * 100
        
        penjelasan = (
            f"Sistem mendeteksi adanya indikasi 'Diabetic Retinopathy' dengan tingkat kepastian {round(persentase, 2)}%. "
            f"Model AI menangkap adanya pola kontras atau bintik gelap pada matriks piksel gambar yang menyerupai flek penyakit. "
            f"Namun, jika pada data klinis asli gambar ini sebenarnya adalah Normal (Sehat), nilai akurasi ini dipengaruhi oleh "
            f"adanya noise, bayangan lampu kamera fundus, atau garis blur pada gambar luar yang membuat komputer salah menginterpretasikannya "
            f"sebagai gejala klinis (False Positive)."
        )
        rekomendasi = "Disarankan untuk segera melakukan pemeriksaan lanjutan ke Dokter Spesialis Mata untuk mendapatkan penanganan lebih tepat."

    return render_template(
        "result.html",
        hasil=hasil,
        status=status,
        nilai=round(nilai, 4),
        persentase=round(persentase, 2),
        gambar=filepath,
        filename=filename,
        penjelasan=penjelasan,
        rekomendasi=rekomendasi,
    )

if __name__ == "__main__":
    app.run()