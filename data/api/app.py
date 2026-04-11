"""
==============================================================
FLASK API - Deteksi Stunting Pada Balita
==============================================================
Endpoint  : POST /predict
Input     : JK, Usia_Bulan, Berat, Tinggi
Output    : Status_Stunting, Probabilitas
Port      : 5000
==============================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)  # izinkan request dari Laravel

# ──────────────────────────────────────────────
# PATH FILE
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, '..', '..', 'models')
REF_DIR    = os.path.join(BASE_DIR, '..', 'references')

MODEL_PATH   = os.path.join(MODEL_DIR, 'model_stunting.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder_stunting.pkl')
BOYS_PATH    = os.path.join(REF_DIR, 'tab_lhfa_boys_p_0_5.csv')
GIRLS_PATH   = os.path.join(REF_DIR, 'tab_lhfa_girls_p_0_5.csv')

# ──────────────────────────────────────────────
# LOAD MODEL & REFERENSI WHO SAAT STARTUP
# ──────────────────────────────────────────────
print("Loading model...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Load tabel WHO TB/U
who_boys  = pd.read_csv(BOYS_PATH)
who_girls = pd.read_csv(GIRLS_PATH)

print("✅ Model dan tabel WHO berhasil dimuat!")


# ──────────────────────────────────────────────
# FUNGSI HITUNG ZS TB/U (untuk referensi saja)
# tidak dipakai sebagai fitur model
# ──────────────────────────────────────────────
def hitung_zs_tbu(jk, usia_bulan, tinggi):
    """
    Hitung Z-Score TB/U menggunakan tabel WHO.
    Digunakan sebagai referensi validasi, bukan fitur model.
    """
    try:
        tabel = who_boys if jk == 1 else who_girls
        row   = tabel[tabel['Month'] == usia_bulan]
        if row.empty:
            return None
        M  = row['M'].values[0]
        SD = row['SD'].values[0]
        return round((tinggi - M) / SD, 2)
    except:
        return None


# ──────────────────────────────────────────────
# FUNGSI VALIDASI INPUT
# ──────────────────────────────────────────────
def validasi_input(data):
    errors = []

    # Cek field wajib
    required = ['JK', 'Usia_Bulan', 'Berat', 'Tinggi']
    for field in required:
        if field not in data:
            errors.append(f"Field '{field}' wajib diisi.")

    if errors:
        return errors

    # Validasi JK
    if data['JK'] not in [0, 1]:
        errors.append("JK harus 0 (Perempuan) atau 1 (Laki-laki).")

    # Validasi Usia
    usia = data['Usia_Bulan']
    if not (0 <= usia <= 60):
        errors.append("Usia_Bulan harus antara 0-60 bulan.")

    # Validasi Berat
    berat = data['Berat']
    if not (1 <= berat <= 30):
        errors.append("Berat harus antara 1-30 kg.")

    # Validasi Tinggi
    tinggi = data['Tinggi']
    if not (40 <= tinggi <= 120):
        errors.append("Tinggi harus antara 40-120 cm.")

    return errors


# ──────────────────────────────────────────────
# ENDPOINT: CEK STATUS API
# ──────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status' : 'ok',
        'message': 'Flask API Deteksi Stunting aktif',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Prediksi status stunting balita'
        }
    })


# ──────────────────────────────────────────────
# ENDPOINT: PREDIKSI STUNTING
# ──────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request Laravel
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'message': 'Request body tidak boleh kosong.'
            }), 400

        # Validasi input
        errors = validasi_input(data)
        if errors:
            return jsonify({
                'success': False,
                'message': 'Input tidak valid.',
                'errors' : errors
            }), 422

        # Ambil nilai input
        jk          = int(data['JK'])
        usia_bulan  = int(data['Usia_Bulan'])
        berat       = float(data['Berat'])
        tinggi      = float(data['Tinggi'])

        # Hitung fitur turunan
        bmi         = round(berat / ((tinggi / 100) ** 2), 4)
        rasio_bb_tb = round(berat / tinggi, 4)

        # Susun input untuk model
        input_model = pd.DataFrame([{
            'JK'         : jk,
            'Usia_Bulan' : usia_bulan,
            'Berat'      : berat,
            'Tinggi'     : tinggi,
            'BMI'        : bmi,
            'Rasio_BB_TB': rasio_bb_tb
        }])

        # Prediksi
        hasil_encoded = model.predict(input_model)[0]
        hasil_label   = le.inverse_transform([hasil_encoded])[0]
        probabilitas  = model.predict_proba(input_model)[0]

        # Hitung ZS TB/U sebagai referensi
        zs_tbu = hitung_zs_tbu(jk, usia_bulan, tinggi)

        # Susun probabilitas per kelas
        prob_dict = {
            kelas: round(float(prob) * 100, 2)
            for kelas, prob in zip(le.classes_, probabilitas)
        }

        return jsonify({
            'success': True,
            'data': {
                'input': {
                    'JK'        : 'Laki-laki' if jk == 1 else 'Perempuan',
                    'Usia_Bulan': usia_bulan,
                    'Berat'     : berat,
                    'Tinggi'    : tinggi,
                },
                'hasil': {
                    'Status_Stunting': hasil_label,
                    'ZS_TBU'         : zs_tbu,
                    'Probabilitas'   : prob_dict,
                    'Confidence'     : round(float(max(probabilitas)) * 100, 2)
                }
            }
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Terjadi kesalahan pada server: {str(e)}'
        }), 500


# ──────────────────────────────────────────────
# RUN SERVER
# ──────────────────────────────────────────────
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True   # ganti False saat production/hosting
    )