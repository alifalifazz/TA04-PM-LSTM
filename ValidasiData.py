import pandas as pd
import matplotlib.pyplot as plt

# Muat dua dataset
train = pd.read_csv("dataset/DailyDelhiClimateTrain.csv")
test = pd.read_csv("dataset/DailyDelhiClimateTest.csv")

# Gabungkan keduanya agar data lengkap
df = pd.concat([train, test])

# Konversi kolom tanggal menjadi datetime dan set sebagai index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Tampilkan 5 data pertama
print("5 Data Teratas:")
print(df.head())

# Buat plot garis dari kolom target (suhu rata-rata harian)
plt.figure(figsize=(12, 6))
plt.plot(df['meantemp'], color='orange')
plt.title('Grafik Suhu Udara Harian di Delhi (Train + Test Dataset)')
plt.xlabel('Tanggal')
plt.ylabel('Suhu Rata-rata (°C)')
plt.grid(True)
plt.show()
