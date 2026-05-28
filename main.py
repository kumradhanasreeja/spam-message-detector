#!/usr/bin/env python3
"""
CLI for the Spam Message Detector.

Usage examples:
  python main.py                          # interactive mode
  python main.py --message "Win a prize!" # single message
  python main.py --file messages.txt      # classify file (one message per line)
  python main.py --train data.csv         # train on a CSV (columns: text, label)
  python main.py --demo                   # run built-in demo
"""

import argparse
import csv
import sys
from spam_detector import SpamDetector


def load_csv(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"].strip().lower())
    return texts, labels


def print_result(message: str, result: dict):
    icon = "🚫 SPAM" if result["label"] == "spam" else "✅ HAM "
    bar_len = int(result["confidence"] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"\n{icon}  [{bar}] {result['confidence']:.0%} confidence")
    print(f"       Message : {message[:80]}{'…' if len(message) > 80 else ''}")
    print(f"       Scores  : spam={result['scores'].get('spam', 0):.4f}  "
          f"ham={result['scores'].get('ham', 0):.4f}")


def demo(detector: SpamDetector):
    samples = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now.",
        "Hey, are you joining us for lunch today?",
        "URGENT: Your bank account has been suspended. Verify at http://scam.com",
        "Don't forget to bring your laptop to the meeting.",
        "You have been selected for a £1000 cash reward. Reply YES to claim!",
        "The report is ready, I'll send it over this afternoon.",
    ]
    print("\n" + "=" * 60)
    print("  SPAM DETECTOR — DEMO")
    print("=" * 60)
    for msg in samples:
        result = detector.predict(msg)
        print_result(msg, result)
    print("\n" + "=" * 60)


def interactive(detector: SpamDetector):
    print("\n🔍 Spam Detector — Interactive Mode")
    print("Type a message and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("Message> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        result = detector.predict(msg)
        print_result(msg, result)
        print()


def main():
    parser = argparse.ArgumentParser(description="Spam Message Detector")
    parser.add_argument("--message", "-m", help="Single message to classify")
    parser.add_argument("--file", "-f", help="Text file with one message per line")
    parser.add_argument("--train", "-t", help="CSV file with columns: text, label")
    parser.add_argument("--model", default="spam_model.pkl", help="Model file path")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo")
    args = parser.parse_args()

    detector = SpamDetector()

    # ---- train or load ----
    if args.train:
        print(f"📖 Loading training data from {args.train}…")
        texts, labels = load_csv(args.train)
        detector.train(texts, labels)
        detector.save(args.model)
        return

    try:
        detector.load(args.model)
    except FileNotFoundError:
        print("ℹ️  No saved model found — training on built-in sample data.")
        detector.train_on_sample_data()
        detector.save(args.model)

    # ---- classify ----
    if args.demo:
        demo(detector)
    elif args.message:
        result = detector.predict(args.message)
        print_result(args.message, result)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n📂 Classifying {len(lines)} messages from {args.file}:\n")
        for line in lines:
            result = detector.predict(line)
            print_result(line, result)
        print()
    else:
        interactive(detector)


if __name__ == "__main__":
    main()