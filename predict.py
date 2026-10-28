"""Classify a single email.

Usage:
    python predict.py "Urgent! Verify your account at http://192.168.1.5/login"
    python predict.py --file email.txt
"""
import argparse
import sys

import joblib

import features  # noqa: F401  (needed so the saved pipeline can be loaded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Email text")
    ap.add_argument("--file", help="Read email text from a file")
    args = ap.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        sys.exit("Provide email text or --file")

    model = joblib.load("models/phishing_model.joblib")
    label = model.predict([text])[0]
    msg = "PHISHING" if label == 1 else "SAFE"
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba([text])[0][label]
        print(f"Result: {msg} (confidence {conf:.1%})")
    else:
        print(f"Result: {msg}")


if __name__ == "__main__":
    main()
