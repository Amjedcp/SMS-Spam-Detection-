# 📨 SMS Spam Detector (scikit-learn + Streamlit)

Classifies SMS messages as Spam or Ham using TF-IDF + Logistic Regression.
Includes a tuned decision threshold and a Streamlit demo app.

## Dataset
- UCI SMS Spam Collection (~5.5k messages, ~13% spam)

## Modeling
- Pipeline: TF-IDF (1–2 grams, min_df=2) → LogisticRegression(class_weight='balanced', solver='liblinear')
- Threshold tuning via 5-fold CV to maximize F1 (saved as best_threshold)

## Results (test set)
- ROC-AUC: 0.9862471688412746
- PR-AUC: 0.9655545324114828
- F1@0.5: 0.9108910891089109
- Tuned threshold: 0.55

## Run locally
pip install -r requirements.txt
# Train (optional if model files are included)
python train.py
# Launch app
streamlit run streamlit_app.py

## Files
- train.py — trains model, evaluates, saves artifacts (sms_spam_pipeline.joblib, spam_config.json, metrics.json)
- streamlit_app.py — Streamlit UI that loads the trained pipeline and threshold
- requirements.txt — dependencies
- metrics.json — reported metrics
- sms_spam_pipeline.joblib, spam_config.json — model + threshold