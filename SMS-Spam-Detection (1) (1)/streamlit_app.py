import json, joblib, streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📨")
st.title("📨 SMS Spam Detector")

@st.cache_resource
def load_artifacts():
    model = joblib.load("sms_spam_pipeline.joblib")
    try:
        with open("spam_config.json") as f:
            th = float(json.load(f).get("best_threshold", 0.5))
    except FileNotFoundError:
        th = 0.5
    return model, th

model, default_thresh = load_artifacts()

st.sidebar.header("Settings")
thresh = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(default_thresh), 0.01)

st.write("Paste an SMS message and click Predict.")
msg = st.text_area("Message", height=120, placeholder="e.g., Congratulations! You won a prize. Call now to claim.")

if st.button("Predict") and msg.strip():
    proba = float(model.predict_proba([msg])[:, 1])
    label = "Spam" if proba >= thresh else "Ham"
    st.metric("Prediction", label)
    st.caption(f"Spam probability: {proba:.3f} | Threshold: {thresh:.2f}")

with st.expander("Try examples"):
    examples = [
        "Congratulations! You won a $1000 gift card. Call now to claim.",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT! Your account is suspended. Verify your details immediately."
    ]
    st.code("\\n\\n".join(examples))
