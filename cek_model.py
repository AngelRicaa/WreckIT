import pickle

with open("buzzer_model.pkl", "rb") as f:
    saved = pickle.load(f)

model    = saved['model']
features = saved['features']

print("Model berhasil dibuka!")
print(f"Jumlah pohon keputusan: {model.n_estimators}")
print(f"\nFitur yang dipakai ({len(features)}):")
for i, f in enumerate(features, 1):
    print(f"  {i}. {f}")

print(f"\nKelas yang bisa diprediksi: {model.classes_}")
print("  0 = Normal")
print("  1 = Mencurigakan") 
print("  2 = Buzzer")