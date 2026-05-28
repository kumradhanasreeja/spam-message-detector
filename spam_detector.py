"""
Spam Message Detector
Uses TF-IDF + Multinomial Naive Bayes for classification.
"""

import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------- Text Preprocessing ----------

def preprocess(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)                   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


# ---------- Sample Training Data ----------

SAMPLE_DATA = [
    # spam
    ("FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121", "spam"),
    ("WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!", "spam"),
    ("Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles", "spam"),
    ("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575", "spam"),
    ("URGENT! You have won a 1 week FREE membership in our dollar prize", "spam"),
    ("Congratulations! You've been selected for a cash reward. Call now 08001560100.", "spam"),
    ("Free entry to our prize draw! Reply WIN to claim your reward worth £500.", "spam"),
    ("You have 1 new voicemail. Call 09090900040 to listen. Cost 35p/min", "spam"),
    ("PRIVATE! Your 2003 Account Statement shows 800 un-redeemed S-I-M points", "spam"),
    ("Are you interested in enlarging certain parts of your body? Call us today!", "spam"),
    ("Claim your FREE gift voucher worth £200. Text GIFT to 80880 now!", "spam"),
    ("Alert: Your account is suspended. Verify immediately at http://fakbank.com", "spam"),
    ("Win a brand new iPhone! You are our lucky visitor. Click here to claim.", "spam"),
    ("CASH PRIZE: You have won £1000 Tesco gift card. Reply CLAIM to 80811.", "spam"),
    ("Loan approved! Get £5000 cash today. No credit check required. Call now.", "spam"),
    # ham
    ("I'll be there in 10 minutes, just leaving the office now.", "ham"),
    ("Are you coming to the meeting tomorrow?", "ham"),
    ("Hey, can you pick up some milk on your way home?", "ham"),
    ("The project deadline has been moved to next Friday.", "ham"),
    ("Happy birthday! Hope you have a wonderful day.", "ham"),
    ("Don't forget we have dinner with mum tonight at 7.", "ham"),
    ("I finished the report, sending it over now.", "ham"),
    ("Can we reschedule our call to 3pm instead?", "ham"),
    ("Thanks for your help yesterday, really appreciated it.", "ham"),
    ("The kids' school play is on Thursday evening at 6:30.", "ham"),
    ("Just landed, waiting for my bag at baggage claim.", "ham"),
    ("Good morning! Did you sleep well?", "ham"),
    ("The WiFi password is written on the back of the router.", "ham"),
    ("Let me know when you're free to chat about the proposal.", "ham"),
    ("I'll grab lunch from the canteen, want anything?", "ham"),
]


# ---------- Model ----------

class SpamDetector:
    MODEL_PATH = "spam_model.pkl"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=preprocess,
                ngram_range=(1, 2),
                max_features=10_000,
                sublinear_tf=True,
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        self.trained = False

    # ---- training ----

    def train(self, texts: list[str], labels: list[str], verbose: bool = True) -> dict:
        """Train on a list of (text, label) pairs where label is 'spam' or 'ham'."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.pipeline.fit(X_train, y_train)
        self.trained = True

        y_pred = self.pipeline.predict(X_test)
        report = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
        }

        if verbose:
            print(f"\n✅ Model trained on {len(X_train)} samples.")
            print(f"📊 Test accuracy: {report['accuracy']:.2%}")
            print(report["classification_report"])

        return report

    def train_on_sample_data(self):
        """Quick-start: train on the built-in sample dataset."""
        texts  = [t for t, _ in SAMPLE_DATA]
        labels = [l for _, l in SAMPLE_DATA]
        # For demo purposes train on all (tiny dataset)
        self.pipeline.fit(texts, labels)
        self.trained = True
        print("✅ Model trained on built-in sample data.")

    # ---- prediction ----

    def predict(self, text: str) -> dict:
        """Return {'label': 'spam'|'ham', 'confidence': float}."""
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx = proba.argmax()
        return {
            "label": classes[idx],
            "confidence": round(float(proba[idx]), 4),
            "scores": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ---- persistence ----

    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"💾 Model saved to {path}")

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file at {path}")
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.trained = True
        print(f"📂 Model loaded from {path}")