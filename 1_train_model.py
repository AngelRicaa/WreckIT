"""
STEP 1: Training AI Model Buzzer Detector
==========================================
Jalankan file ini PERTAMA untuk melatih model AI.
Hasil: file 'buzzer_model.pkl'

Cara jalankan:
    pip install pandas scikit-learn openpyxl
    python 1_train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("  BUZZER DETECTOR - TRAINING MODEL")
print("=" * 50)

# ── 1. Load data ──────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_excel("account_profile_labeled_network.xlsx")
print(f"      Loaded {len(df)} baris")

# ── 2. Convert label ──────────────────────────────────
# buzzer_label: 1=Normal, 2=Suspicious, 3=Buzzer
# Kita ubah jadi:  0=Normal, 1=Suspicious, 2=Buzzer
print("[2/5] Preparing labels...")
df['label'] = df['buzzer_label'] - 1
counts = df['label'].value_counts().sort_index()
print(f"      Normal: {counts.get(0,0)} | Suspicious: {counts.get(1,0)} | Buzzer: {counts.get(2,0)}")

# ── 3. Feature engineering ────────────────────────────
print("[3/5] Engineering features...")

def parse_date(s):
    try:
        return datetime.strptime(str(s).strip(), '%a %b %d %H:%M:%S +0000 %Y')
    except:
        return None

df['created_dt'] = df['created_at'].apply(parse_date)
ref_date = datetime(2020, 12, 31)
df['account_age_days'] = df['created_dt'].apply(
    lambda x: (ref_date - x).days if x else None
)
df['account_age_days'] = df['account_age_days'].fillna(df['account_age_days'].median())

for col in ['followers_count','friends_count','statuses_count','listed_count']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['ff_ratio']          = df['friends_count'] / (df['followers_count'] + 1)
df['tweets_per_day']    = df['statuses_count'] / (df['account_age_days'] + 1)
df['has_description']   = df['description'].apply(lambda x: 0 if str(x).strip() in ['','nan'] else 1)
df['verified_int']      = df['verified'].apply(lambda x: 1 if str(x).strip() == 'True' else 0)
df['is_new_account']    = (df['account_age_days'] < 365).astype(int)
df['low_followers']     = (df['followers_count'] < 100).astype(int)
df['high_ff_ratio']     = (df['ff_ratio'] > 5).astype(int)
df['buzzer_score_feat'] = pd.to_numeric(df['buzzer_score'], errors='coerce').fillna(0)

FEATURES = [
    'followers_count','friends_count','statuses_count','listed_count',
    'account_age_days','ff_ratio','tweets_per_day','has_description',
    'verified_int','is_new_account','low_followers','high_ff_ratio',
    'buzzer_score_feat'
]

X = df[FEATURES]
y = df['label']

# ── 4. Balance & train ────────────────────────────────
print("[4/5] Balancing & training model...")
max_n = y.value_counts().max()
frames = []
for cls in y.unique():
    subset = df[y == cls]
    frames.append(resample(subset, replace=True, n_samples=max_n, random_state=42))
df_bal = pd.concat(frames)
X_bal, y_bal = df_bal[FEATURES], df_bal['label']

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_bal, y_bal, cv=cv, scoring='f1_weighted')
model.fit(X_bal, y_bal)

print(f"\n      ✅ F1 Score: {scores.mean():.2f} (+/- {scores.std():.2f})")
print("\n      Feature importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:6]:
    bar = "█" * int(imp * 40)
    print(f"      {feat:<22} {bar} {imp:.3f}")

# ── 5. Simpan ─────────────────────────────────────────
print("\n[5/5] Saving model...")
with open("buzzer_model.pkl", "wb") as f:
    pickle.dump({'model': model, 'features': FEATURES}, f)

print("\n" + "=" * 50)
print("  ✅ SELESAI! 'buzzer_model.pkl' tersimpan.")
print("  Sekarang jalankan: python 2_api_server.py")
print("=" * 50)
