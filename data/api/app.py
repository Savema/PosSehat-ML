from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# --- SETUP PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pastikan nama file model sesuai dengan yang kamu simpan (model_rf_stunting.pkl)
model_path = os.path.join(BASE_DIR, '..', '..', 'models', 'model_rf_stunting.pkl')
ref_path_boys = os.path.join(BASE_DIR, '..', 'references', 'tab_lhfa_boys_p_0_5.csv')
ref_path_girls = os.path.join(BASE_DIR, '..', 'references', 'tab_lhfa_girls_p_0_5.csv')

# --- LOAD MODEL & DATA REFERENSI ---
try:
    model = joblib.load(model_path)
    df_ref_boys = pd.read_csv(ref_path_boys)
    df_ref_girls = pd.read_csv(ref_path_girls)
    print("✅ Model dan Referensi WHO berhasil dimuat!")
except Exception as e:
    print(f"❌ Error saat memuat file: {e}")

def hitung_zscore_who(tinggi, usia, jk):
    df_ref = df_ref_boys if int(jk) == 1 else df_ref_girls
    
    # Ambil baris dengan bulan terdekat (agar lebih stabil)
    usia_int = int(round(usia))
    ref_row = df_ref.iloc[(df_ref['Month'] - usia_int).abs().argsort()[:1]].iloc[0]
    
    L = ref_row['L']
    M = ref_row['M']
    S = ref_row['S']
    
    # Rumus LMS WHO (Sama dengan saat training)
    z_score = ((tinggi / M) ** L - 1) / (L * S)
    return z_score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Ambil data input
        jk = int(data['JK'])
        usia = float(data['Usia']) # Pakai float agar bisa handle desimal
        berat = float(data['Berat'])
        tinggi = float(data['Tinggi'])
        lila = float(data['LiLA'])

        # 1. Hitung Z-score (Untuk dikembalikan ke Frontend sebagai info tambahan)
        zscore_val = hitung_zscore_who(tinggi, usia, jk)
        
        # 2. Buat DataFrame Input (Urutan HARUS sama dengan X_train)
        X_input = pd.DataFrame([[jk, usia, berat, tinggi, lila]], 
                               columns=['JK', 'Usia', 'Berat', 'Tinggi', 'LiLA'])
        
        # 3. Prediksi
        y_pred = model.predict(X_input)
        
        # Mapping hasil agar user tidak bingung (Opsional)
        kategori = {0: "Sangat Pendek", 1: "Pendek", 2: "Normal"}
        hasil_text = kategori.get(int(y_pred[0]), "Tidak Diketahui")
        
        return jsonify({
            'Status_Stunting': int(y_pred[0]),
            'Zscore': float(zscore_val)
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Pakai host 0.0.0.0 agar bisa diakses dari HP atau device lain di jaringan yang sama
    app.run(debug=True, host='0.0.0.0', port=5000)