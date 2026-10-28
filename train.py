"""Phishing Email Detection Model - training, evaluation and comparison.

Usage:
    python train.py --data emails.csv
"""
import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from features import extra_features

TEXT_COLS = ["text", "Email Text", "email_text", "body", "message"]
LABEL_COLS = ["label", "Email Type", "type", "class", "target"]


def pick(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"Could not find {name} column. Columns present: {list(df.columns)}")


def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"Dataset '{path}' not found. Download from Kaggle and pass --data <file>.")
    df = pd.read_csv(path)
    tcol, lcol = pick(df, TEXT_COLS, "text"), pick(df, LABEL_COLS, "label")
    df = df[[tcol, lcol]].dropna().drop_duplicates()
    df.columns = ["text", "label"]
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.lower().str.contains("phish").astype(int)
    return df.reset_index(drop=True)


def make_pipeline(classifier):
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("extra", Pipeline([("f", FunctionTransformer(extra_features)),
                            ("scale", MaxAbsScaler())])),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def plot_cm(cm, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="emails.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data(args.data)
    print(f"Loaded {len(df)} emails | Phishing: {df.label.sum()} | Safe: {(df.label == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size,
        stratify=df["label"], random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
    }

    report_lines, best_name, best_acc, best_pipe, best_pred = [], None, -1, None, None
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        report_lines += [
            f"=== {name} ===",
            f"5-fold CV F1 (train): {cv.mean():.4f} +/- {cv.std():.4f}",
            f"Test accuracy: {acc:.4f}",
            classification_report(y_test, pred, target_names=["Safe", "Phishing"]),
        ]
        plot_cm(confusion_matrix(y_test, pred), f"{name} (acc {acc:.2%})",
                f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png")
        if acc > best_acc:
            best_name, best_acc, best_pipe, best_pred = name, acc, pipe, pred

    report_lines.append(f"Best model: {best_name} (accuracy {best_acc:.2%})")
    report = "\n".join(report_lines)
    print(report)

    plot_cm(confusion_matrix(y_test, best_pred), f"{best_name} (acc {best_acc:.2%})",
            "images/confusion_matrix.png")
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(report)
    joblib.dump(best_pipe, "models/phishing_model.joblib")
    print("Saved: models/phishing_model.joblib, results.txt, images/confusion_matrix*.png")


if __name__ == "__main__":
    main()
