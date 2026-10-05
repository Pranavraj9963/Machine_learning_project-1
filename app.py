from pathlib import Path
import re
import joblib
import nltk
import streamlit as st
import torch
import torch.nn as nn

from utils.preprocessing import preprocess_text

# Safe database import: works when MySQL is present, doesn't crash on cloud environments
try:
    from utils.database import save_prediction
except Exception:
    def save_prediction(*args, **kwargs):
        pass

# Ensure NLTK resources
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Resolve model paths robustly
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class SpamLSTM(nn.Module):
    """Bidirectional LSTM Architecture matching existing project model."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden).squeeze(1)


@st.cache_resource
def load_rf_model():
    model = joblib.load(MODELS_DIR / "random forest.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, tfidf


@st.cache_resource
def load_bilstm_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = joblib.load(MODELS_DIR / "vocab.pkl")
    model = SpamLSTM(len(vocab))
    model.load_state_dict(
        torch.load(MODELS_DIR / "spam_lstm_best.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model, vocab, device


def encode_bilstm(sentence: str, vocab: dict, max_len: int = 100):
    tokens = sentence.split()
    ids = [vocab.get(token, vocab.get("<UNK>", 1)) for token in tokens][:max_len]
    pad_id = vocab.get("<PAD>", 0)
    while len(ids) < max_len:
        ids.append(pad_id)
    return ids


st.set_page_config(
    page_title="Smart Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

st.title("Smart email spam Detection")
st.write("spam Detection using Machine Learning + Deep Learning + NLP")

st.sidebar.title("Model")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Bi-LSTM"])

email = st.text_area("Enter Email Content", height=250)
predict = st.button("Predict")

if predict:
    if email.strip() == "":
        st.warning("Please Enter Email Content")
    else:
        text = preprocess_text(email)

        if model_choice == "Random Forest":
            model, tfidf = load_rf_model()
            vector = tfidf.transform([text])
            prediction = int(model.predict(vector)[0])
            probability = float(model.predict_proba(vector)[0][1])

            if prediction == 1:
                st.error("SPAM EMAIL")
            else:
                st.success("HAM EMAIL")

            st.write(f"Spam Probability : {probability * 100:.2f}%")
            st.progress(max(0.0, min(1.0, float(probability))))

            try:
                save_prediction(
                    email,
                    "Random Forest",
                    "SPAM" if prediction == 1 else "HAM",
                    probability
                )
            except Exception:
                pass

        elif model_choice == "Bi-LSTM":
            dl_model, vocab, device = load_bilstm_model()
            MAX_LEN = 100
            encoded = encode_bilstm(text, vocab, max_len=MAX_LEN)
            x = torch.tensor([encoded], dtype=torch.long).to(device)

            with torch.no_grad():
                output = dl_model(x)
                probability = float(torch.sigmoid(output).item())

            prediction = 1 if probability >= 0.5 else 0

            if prediction == 1:
                st.error("SPAM Email")
            else:
                st.success("HAM Email")

            st.write(f"Spam Probability : {probability * 100:.2f}%")
            st.progress(max(0.0, min(1.0, float(probability))))

            try:
                save_prediction(
                    email,
                    "Bi-LSTM",
                    "SPAM" if prediction == 1 else "HAM",
                    probability
                )
            except Exception:
                pass
