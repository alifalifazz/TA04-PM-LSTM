from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import os

# Inisialisasi Flask
app = Flask(__name__)

# Pastikan folder static ada
os.makedirs("static", exist_ok=True)

# Muat model dan scaler
model = load_model('models/lstm_model.h5')
scaler = joblib.load('models/scaler.pkl')

# === 🔹 Muat dan gabungkan dataset Train + Test ===
train_data = pd.read_csv('dataset/DailyDelhiClimateTrain.csv')
test_data = pd.read_csv('dataset/DailyDelhiClimateTest.csv')

# Gabungkan dua dataset jadi satu
data = pd.concat([train_data, test_data])

# Ubah kolom 'date' ke datetime dan set jadi index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Ambil kolom suhu rata-rata
df = data[['meantemp']].copy()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input suhu dari pengguna
        suhu_input = float(request.form['suhu'])

        # Skala seluruh data historis
        scaled_data = scaler.transform(df.values)

        # Ambil 60 data terakhir untuk urutan LSTM
        timesteps = 60
        last_sequence = scaled_data[-timesteps:].flatten()

        # Ganti nilai terakhir dengan input suhu pengguna
        last_sequence[-1] = scaler.transform([[suhu_input]])[0][0]

        # Ubah ke bentuk [1, timesteps, 1]
        X_input = np.reshape(last_sequence, (1, timesteps, 1))

        # Prediksi suhu hari berikutnya
        pred_scaled = model.predict(X_input)
        hasil = scaler.inverse_transform(pred_scaled)[0][0]

        # === 🔹 VISUALISASI SELURUH DATA ===
        plt.figure(figsize=(12, 6))

        # Garis biru = seluruh data suhu aktual dari dataset
        plt.plot(df.index, df['meantemp'], label='Suhu Aktual (°C)', color='royalblue', linewidth=2)

        # Garis merah = prediksi 1 hari ke depan (setelah data terakhir)
        plt.plot([df.index[-1], df.index[-1] + pd.Timedelta(days=1)],
                 [df['meantemp'].iloc[-1], hasil],
                 'r--o', label='Prediksi Hari Berikutnya', color='red', linewidth=2, markersize=7)

        plt.title('Grafik Suhu Aktual vs Prediksi Hari Berikutnya (LSTM)',
                  fontsize=13, fontweight='bold', color='#023e8a')
        plt.xlabel('Tanggal')
        plt.ylabel('Suhu (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Simpan plot ke folder static
        plot_path = 'static/plot.png'
        plt.savefig(plot_path)
        plt.close()

        # Kirim hasil ke template
        return render_template(
            'result.html',
            prediksi=f"{hasil:.2f}",
            input_user=f"{suhu_input:.2f}",
            plot_url='/static/plot.png'
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
