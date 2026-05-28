"""
Tests for the Spam Message Detector.
Run with: pytest tests.py -v
"""

import pytest
from spam_detector import SpamDetector, preprocess


# ---- preprocess ----

def test_preprocess_lowercase():
    assert preprocess("HELLO WORLD") == "hello world"

def test_preprocess_removes_urls():
    assert "http" not in preprocess("visit http://spam.com today")

def test_preprocess_removes_numbers():
    assert "123" not in preprocess("call 123456789 now")

def test_preprocess_removes_punctuation():
    result = preprocess("Hello, world!!!")
    assert "," not in result and "!" not in result

def test_preprocess_empty_string():
    assert preprocess("") == ""


# ---- SpamDetector ----

@pytest.fixture
def trained_detector():
    d = SpamDetector()
    d.train_on_sample_data()
    return d


def test_train_on_sample_data(trained_detector):
    assert trained_detector.trained is True


def test_predict_returns_expected_keys(trained_detector):
    result = trained_detector.predict("Hello, how are you?")
    assert "label" in result
    assert "confidence" in result
    assert "scores" in result


def test_predict_label_is_valid(trained_detector):
    for msg in ["Free prize!", "Let's meet tomorrow"]:
        result = trained_detector.predict(msg)
        assert result["label"] in ("spam", "ham")


def test_confidence_is_probability(trained_detector):
    result = trained_detector.predict("Win a free gift now!")
    assert 0.0 <= result["confidence"] <= 1.0


def test_obvious_spam(trained_detector):
    result = trained_detector.predict(
        "CONGRATULATIONS! You won £1000. Claim now by texting WIN to 80800."
    )
    assert result["label"] == "spam"


def test_obvious_ham(trained_detector):
    result = trained_detector.predict(
        "Hi mum, I'll be home by 7 for dinner tonight."
    )
    assert result["label"] == "ham"


def test_predict_batch(trained_detector):
    msgs = ["Free prize!", "See you tomorrow", "Win cash now!"]
    results = trained_detector.predict_batch(msgs)
    assert len(results) == 3
    for r in results:
        assert r["label"] in ("spam", "ham")


def test_predict_without_training_raises():
    d = SpamDetector()
    with pytest.raises(RuntimeError):
        d.predict("Hello")


def test_save_and_load(tmp_path, trained_detector):
    model_path = str(tmp_path / "model.pkl")
    trained_detector.save(model_path)

    new_detector = SpamDetector()
    new_detector.load(model_path)

    result = new_detector.predict("Win a free prize now!")
    assert result["label"] in ("spam", "ham")


def test_train_with_custom_data():
    texts = [
        "Win a free prize!",
        "Claim your cash reward now!",
        "Hey, are you free tonight?",
        "Let's grab coffee tomorrow.",
    ]
    labels = ["spam", "spam", "ham", "ham"]
    d = SpamDetector()
    d.train(texts, labels, verbose=False)
    assert d.trained is True
    result = d.predict("Win cash!")
    assert result["label"] in ("spam", "ham")