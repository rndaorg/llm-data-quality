import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import hashlib

# -----------------------------
# Load dataset
# -----------------------------
# Example CSV columns:
# text, label, domain

df = pd.read_csv("dataset.csv")

# -----------------------------
# Train/Test Split
# -----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# 1. Contamination Detection
# -----------------------------
def hash_row(row):
    return hashlib.md5(
        str(row.values).encode()
    ).hexdigest()

train_hashes = set(train_df.apply(hash_row, axis=1))
test_hashes = set(test_df.apply(hash_row, axis=1))

overlap = train_hashes.intersection(test_hashes)

print("=" * 50)
print("CONTAMINATION CHECK")
print("=" * 50)
print(f"Duplicate records across train/test: {len(overlap)}")

# -----------------------------
# 2. Class Imbalance
# -----------------------------
print("\nCLASS DISTRIBUTION")

label_counts = Counter(df["label"])

total = len(df)

for label, count in label_counts.items():
    pct = 100 * count / total
    print(f"{label}: {count} ({pct:.2f}%)")

imbalance_ratio = max(label_counts.values()) / min(label_counts.values())

print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

# -----------------------------
# 3. Domain Imbalance
# -----------------------------
print("\nDOMAIN DISTRIBUTION")

domain_counts = Counter(df["domain"])

for domain, count in domain_counts.items():
    pct = 100 * count / total
    print(f"{domain}: {count} ({pct:.2f}%)")

domain_ratio = max(domain_counts.values()) / min(domain_counts.values())

print(f"\nDomain imbalance ratio: {domain_ratio:.2f}")

# -----------------------------
# 4. Noisy Label Detection
# -----------------------------
# Simple heuristic:
# Train a model and find samples
# where prediction disagrees with label.

feature_cols = [c for c in df.columns
                if c not in ["label", "domain"]]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_train)

suspected_noise = train_df[preds != y_train]

print("\nPOTENTIAL LABEL NOISE")
print(f"Suspicious samples: {len(suspected_noise)}")
print(f"Noise rate: {100*len(suspected_noise)/len(train_df):.2f}%")

# -----------------------------
# 5. Overall Model Performance
# -----------------------------
test_preds = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print(classification_report(y_test, test_preds))
