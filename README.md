# Phishing Email Detection Model

A machine learning model built with **Scikit-learn** that classifies emails as **Phishing** or **Safe** using textual content and URL features.

## Features
- Trains on a dataset of phishing and legitimate emails
- **TF-IDF** text features (unigrams + bigrams)
- **URL features**: URL count, IP-based URLs, long URLs, shortened links, `@` symbol, non-HTTPS links
- **Keyword features**: urgent, verify, password, account, click, suspended, etc.
- Compares **Logistic Regression** vs **Random Forest** with 5-fold cross-validation
- Shows **accuracy, precision, recall, F1** and **confusion matrix**
- `predict.py` to classify any single email

## Project Structure
```
phishing-email-detection/
├── features.py          # URL + keyword feature extraction
├── train.py             # training, evaluation, model comparison
├── predict.py           # classify a single email
├── requirements.txt
├── images/              # confusion matrix plots
└── models/              # saved model (generated)
```

## Dataset
Download the **Phishing Email Dataset** from Kaggle and save it as `emails.csv`.
Column names `Email Text` / `Email Type` (or `text` / `label`) are detected automatically.

## Setup & Run
```bash
pip install -r requirements.txt
python train.py --data emails.csv
python predict.py "Urgent! Verify your account at http://192.168.1.5/login"
```

## Results
| Model | CV F1 | Test Accuracy |
|-------|-------|---------------|
| Logistic Regression | XX | XX% |
| Random Forest | XX | XX% |

(Fill in from `results.txt` after running `train.py`.)

![Confusion Matrix](images/confusion_matrix.png)

## Tech Stack
Python, Scikit-learn, Pandas, SciPy, Matplotlib, Seaborn

## Future Work
- XGBoost / BERT-based comparison
- Email header analysis (SPF/DKIM)
- Flask web app for live email checking

## Author
**Obito (Durairaj V)** - Cybersecurity student | [GitHub](https://github.com/durairajv172-byte)
