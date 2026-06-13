import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import classification_report
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  WRECK-IT BUZZER DETECTOR — AI TRAINING")
print("=" * 60)
print("\n-> Memuat dataset...")

df = pd.read_csv("account_profile_labeled.csv")

print(f"  -> {len(df)} akun berhasil dimuat")
print(f"  -> Kolom: {list(df.columns)}")

# converting label data set, normal jadi 0, suspicious jadi 1, buzzer jadi 2

print("\n-> Menyiapkan label...")

LABEL_MAP = {'normal': 0, 'suspicious': 1, 'buzzer': 2}
df['label'] = df['class'].map(LABEL_MAP)

dist = df['label'].value_counts().sort_index()
print(f"  -> Normal      : {dist.get(0, 0)} akun")
print(f"  -> Suspicious  : {dist.get(1, 0)} akun")
print(f"  -> Buzzer      : {dist.get(2, 0)} akun")

print("\n-> Membuat fitur...")

def parse_tanggal(teks):
    try:
        return datetime.strptime(str(teks).strip(), '%a %b %d %H:%M:%S +0000 %Y')
    except:
        return None

TANGGAL_REFERENSI = datetime(2024, 1, 1)

df['created_dt']       = df['created_at'].apply(parse_tanggal)
df['account_age_days'] = df['created_dt'].apply(
    lambda x: (TANGGAL_REFERENSI - x).days if x is not None else None
)
df['account_age_days'] = df['account_age_days'].fillna(
    df['account_age_days'].median()
)

for kolom in ['followers_count', 'friends_count', 'statuses_count', 'listed_count']:
    df[kolom] = pd.to_numeric(df[kolom], errors='coerce').fillna(0)

df['ff_ratio']        = df['friends_count'] / (df['followers_count'] + 1)
df['tweets_per_day']  = df['statuses_count'] / (df['account_age_days'] + 1)
df['has_description'] = df['description'].apply(
    lambda x: 0 if str(x).strip() in ['', 'nan'] else 1
)
df['verified_int']    = df['verified'].apply(
    lambda x: 1 if str(x).strip() == 'True' else 0
)
df['is_new_account']  = (df['account_age_days'] < 730).astype(int)
df['low_followers']   = (df['followers_count'] < 500).astype(int)
df['high_ff_ratio']   = (df['ff_ratio'] > 3).astype(int)
df['no_listed']       = (df['listed_count'] == 0).astype(int)

FITUR = [
    'followers_count',   # jumlah follower
    'friends_count',     # jumlah following
    'statuses_count',    # jumlah tweet
    'listed_count',      # berapa kali masuk Twitter list
    'account_age_days',  # umur akun dalam hari
    'ff_ratio',          # rasio following/follower
    'tweets_per_day',    # rata-rata tweet per hari
    'has_description',   # punya bio? (1=ya, 0=tidak)
    'verified_int',      # terverifikasi? (1=ya, 0=tidak)
    'is_new_account',    # akun < 2 tahun? (1=ya, 0=tidak)
    'low_followers',     # follower < 500? (1=ya, 0=tidak)
    'high_ff_ratio',     # rasio following/follower > 3?
    'no_listed',         # tidak pernah masuk list? (1=ya, 0=tidak)
]

X = df[FITUR]
y = df['label']

print(f"  -> {len(FITUR)} fitur berhasil dibuat")

print("\n-> Menyeimbangkan data...")

jumlah_terbanyak = y.value_counts().max()

frames = []
for kelas in sorted(y.unique()):
    data_kelas   = df[y == kelas]
    data_duplikat = resample(
        data_kelas,
        replace=True,
        n_samples=jumlah_terbanyak,
        random_state=42
    )
    frames.append(data_duplikat)

df_seimbang = pd.concat(frames)
X_seimbang  = df_seimbang[FITUR]
y_seimbang  = df_seimbang['label']

print(f"  -> Setiap kelas: {jumlah_terbanyak} data")
print(f"  -> Total setelah balancing: {len(X_seimbang)} baris")


print("\n-> Melatih model...")

model = RandomForestClassifier(
    n_estimators=80,        
    max_depth=7,            
    min_samples_leaf=2,     
    random_state=42,        
    class_weight='balanced' 
)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skor_cv = cross_val_score(model, X_seimbang, y_seimbang, cv=cv, scoring='f1_weighted')

model.fit(X_seimbang, y_seimbang)

print(f"\n  F1 Score: {skor_cv.mean():.2f} (+/- {skor_cv.std():.2f})")
if skor_cv.mean() >= 0.85:
    print(f"  Hasil: BAGUS")
elif skor_cv.mean() >= 0.70:
    print(f"  Hasil: CUKUP")
else:
    print(f"  Hasil: PERLU DIPERBAIKI (tambah lebih banyak data)")

print("\n[-> Evaluasi per kelas...")

prediksi   = model.predict(X)
NAMA_KELAS = ['Normal', 'Suspicious', 'Buzzer']

print()
print(classification_report(y, prediksi, target_names=NAMA_KELAS))

print("  Fitur paling berpengaruh:")
print("  " + "-" * 45)
importances = sorted(zip(FITUR, model.feature_importances_), key=lambda x: -x[1])
for fitur, imp in importances[:8]:
    bar = "x" * int(imp * 50)
    print(f"  {fitur:<22} {bar} {imp:.3f}")


print("\n-> Menyimpan model...")

MODEL_FILE = "buzzer_model.pkl"

with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        'model':     model,
        'features':  FITUR,
        'classes':   NAMA_KELAS,
        'label_map': LABEL_MAP,
        'f1_score':  float(skor_cv.mean()),
    }, f)

print(f"  -> Tersimpan: {MODEL_FILE}")


print("\n" + "=" * 60)
print("  TRAINING SELESAI!")
print("=" * 60)
print(f"""
  Ringkasan:
  - Dataset  : {len(df)} akun (Normal / Suspicious / Buzzer)
  - Fitur    : {len(FITUR)} fitur per akun
  - Algoritma: Random Forest (80 pohon, max depth 7)
  - F1 Score : {skor_cv.mean():.2f} / 1.00
""")
