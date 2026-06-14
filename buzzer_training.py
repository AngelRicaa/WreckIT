import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  WRECK-IT BUZZER DETECTOR - AI TRAINING")
print("  2 Kelas: Normal vs Buzzer")
print("=" * 60)

print("\n-> Memuat dataset...")

df1 = pd.read_csv("account_profile_labeled.csv")
df2 = pd.read_excel("bot_detection_dataset.xlsx")

print(f"  -> Dataset 1 (labelingmu)  : {len(df1)} akun")
print(f"  -> Dataset 2 (bot_detection): {len(df2)} akun")

# converting data set, normal jadi 0, buzzer jadi 1

print("\n-> Menyiapkan label...")

df1 = df1[df1['class'] != 'suspicious'].copy()
df1['label'] = df1['class'].map({'normal': 0, 'buzzer': 1})

print(f"  -> Dataset 1 setelah hapus suspicious: {len(df1)} akun")
print(f"     Normal: {(df1['label']==0).sum()} | Buzzer: {(df1['label']==1).sum()}")
print(f"  -> Dataset 2: Normal: {(df2['label']==0).sum()} | Buzzer: {(df2['label']==1).sum()}")

print("\n-> Membuat fitur...")

def parse_tanggal(teks):
    try:
        return datetime.strptime(str(teks).strip(), '%a %b %d %H:%M:%S +0000 %Y')
    except:
        return None

TANGGAL_REFERENSI = datetime(2024, 1, 1)

def build_features(df):
    df = df.copy()
    for kolom in ['followers_count', 'friends_count', 'statuses_count', 'listed_count']:
        df[kolom] = pd.to_numeric(df[kolom], errors='coerce').fillna(0)

    df['created_dt'] = df['created_at'].apply(parse_tanggal)
    df['account_age_days'] = df['created_dt'].apply(
        lambda x: (TANGGAL_REFERENSI - x).days if x is not None else None
    )

    median_age = df['account_age_days'].median()
    df['account_age_days'] = df['account_age_days'].fillna(median_age)

    df['ff_ratio']       = df['friends_count'] / (df['followers_count'] + 1)
    df['tweets_per_day'] = df['statuses_count'] / (df['account_age_days'] + 1)

    df['has_description'] = df['description'].apply(
        lambda x: 0 if str(x).strip() in ['', 'nan'] else 1
    ) if 'description' in df.columns else pd.Series(0, index=df.index)

    df['verified_int'] = df['verified'].apply(
        lambda x: 1 if str(x).strip() in ['True', '1', '1.0'] else 0
    ) if 'verified' in df.columns else pd.Series(0, index=df.index)

    df['is_new_account'] = (df['account_age_days'] < 730).astype(int)
    df['low_followers']  = (df['followers_count'] < 500).astype(int)
    df['high_ff_ratio']  = (df['ff_ratio'] > 3).astype(int)
    df['no_listed']      = (df['listed_count'] == 0).astype(int)

    if 'screen_name' in df.columns:
        df['username_digit_ratio'] = df['screen_name'].apply(
            lambda x: sum(c.isdigit() for c in str(x)) / (len(str(x)) + 1)
        )
    else:
        df['username_digit_ratio'] = pd.Series(0, index=df.index)

    df['favourites_count'] = pd.to_numeric(
        df['favourites_count'], errors='coerce'
    ).fillna(0) if 'favourites_count' in df.columns else pd.Series(0, index=df.index)

    df['default_profile'] = pd.to_numeric(
        df['default_profile'], errors='coerce'
    ).fillna(0) if 'default_profile' in df.columns else pd.Series(0, index=df.index)

    df['default_profile_image'] = pd.to_numeric(
        df['default_profile_image'], errors='coerce'
    ).fillna(0) if 'default_profile_image' in df.columns else pd.Series(0, index=df.index)

    return df

df1 = build_features(df1)
df2 = build_features(df2)

FITUR = [
    'followers_count',
    'friends_count',
    'statuses_count',
    'listed_count',
    'account_age_days',
    'ff_ratio',
    'tweets_per_day',
    'has_description',
    'verified_int',
    'is_new_account',
    'low_followers',
    'high_ff_ratio',
    'no_listed',
    'favourites_count',
    'default_profile',
    'default_profile_image',
    'username_digit_ratio',
]

combined = pd.concat(
    [df1[FITUR + ['label']], df2[FITUR + ['label']]],
    ignore_index=True
)

X = combined[FITUR]
y = combined['label']

print(f"  -> {len(FITUR)} fitur berhasil dibuat")
print(f"  -> Total gabungan: {len(combined)} akun")
print(f"  -> Normal: {(y==0).sum()} | Buzzer: {(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n  -> Train: {len(X_train)} akun | Test: {len(X_test)} akun")

print("\n-> Melatih model...")

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=4,
    min_samples_leaf=200,
    random_state=42,
    class_weight='balanced'
)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skor_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

model.fit(X_train, y_train)

print(f"\n  CV F1 Score (train): {skor_cv.mean():.2f} (+/- {skor_cv.std():.2f})")
if skor_cv.mean() >= 0.85:
    print(f"  Hasil CV: BAGUS")
elif skor_cv.mean() >= 0.70:
    print(f"  Hasil CV: CUKUP")
else:
    print(f"  Hasil CV: PERLU DIPERBAIKI")

print("\n-> Evaluasi di TEST SET (data baru, bukan training)...")

prediksi_test = model.predict(X_test)
NAMA_KELAS    = ['Normal', 'Buzzer']

print()
print(classification_report(y_test, prediksi_test, target_names=NAMA_KELAS))

print("  Fitur paling berpengaruh:")
print("  " + "-" * 45)
importances = sorted(zip(FITUR, model.feature_importances_), key=lambda x: -x[1])
for fitur, imp in importances[:8]:
    bar = "x" * int(imp * 50)
    print(f"  {fitur:<25} {bar} {imp:.3f}")

print("\n-> Retraining di semua data sebelum disimpan...")
model.fit(X, y)

print("\n-> Menyimpan model...")

MODEL_FILE = "buzzer_model.pkl"

with open(MODEL_FILE, "wb") as f:
    pickle.dump({
        'model':     model,
        'features':  FITUR,
        'classes':   NAMA_KELAS,
        'label_map': {'normal': 0, 'buzzer': 1},
        'f1_score':  float(skor_cv.mean()),
    }, f)

print(f" -> Tersimpan: {MODEL_FILE}")

print("\n" + "=" * 60)
print("  TRAINING SELESAI!")
print("=" * 60)
print(f"""
  Ringkasan:
  - Dataset 1 : account_profile_labeled.csv ({len(df1)} akun)
  - Dataset 2 : bot_detection_dataset.xlsx ({len(df2)} akun)
  - Total     : {len(combined)} akun gabungan
  - Train/Test: {len(X_train)} / {len(X_test)} akun
  - Fitur     : {len(FITUR)} fitur per akun
  - Kelas     : Normal (0) vs Buzzer (1)
  - Algoritma : Random Forest (80 pohon, max depth 4)
  - CV F1     : {skor_cv.mean():.2f} / 1.00  (diukur di data training)
""")
