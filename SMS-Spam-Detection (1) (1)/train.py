import io, zipfile, json, requests
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score

def load_sms_dataset():
    mirrors = [
        "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    ]
    for url in mirrors:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="\t")
            if {"label","message"}.issubset(df.columns):
                print(f"Loaded dataset from: {url}")
                return df
        except Exception as e:
            print(f"Mirror failed: {url} -> {e}")

    # Fallback: UCI zip
    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    print("Trying UCI zip...")
    r = requests.get(zip_url, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["label","message"], encoding="latin-1")
    print("Loaded dataset from UCI zip.")
    return df

def main():
    df = load_sms_dataset()
    df = df.dropna(subset=["message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    assert set(df["label"].unique()) <= {0, 1}, "Labels must be ham/spam only."

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"].values, df["label"].values,
        test_size=0.2, stratify=df["label"].values, random_state=42
    )

    base_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)),
    ])
    model = base_model.fit(X_train, y_train)

    # Evaluate @0.5
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("\nTest metrics @0.5")
    print(classification_report(y_test, y_pred, digits=4))
    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1_05 = f1_score(y_test, y_pred)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC (AP): {ap:.4f} | F1@0.5: {f1_05:.4f}")

    # Threshold tuning via CV on training set (optimize F1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_proba_cv = cross_val_predict(
        base_model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    thresholds = np.linspace(0, 1, 101)
    f1s = [f1_score(y_train, (y_train_proba_cv >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    print(f"Best threshold from 5-fold CV for F1: {best_t:.2f}")

    # Save artifacts
    joblib.dump(model, "sms_spam_pipeline.joblib")
    with open("spam_config.json", "w") as f:
        json.dump({"best_threshold": best_t}, f, indent=2)
    with open("metrics.json", "w") as f:
        json.dump(
            {"roc_auc": float(roc), "pr_auc": float(ap), "f1_at_0_5": float(f1_05), "best_threshold": best_t},
            f, indent=2
        )
    print("\nSaved: sms_spam_pipeline.joblib, spam_config.json, metrics.json")

if __name__ == "__main__":
    main()
