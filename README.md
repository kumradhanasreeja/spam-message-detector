\# 🚫 Spam Message Detector



A lightweight, accurate spam classifier built with Python and scikit-learn.  

Uses \*\*TF-IDF + Multinomial Naive Bayes\*\* — fast, interpretable, and easy to extend.



\---



\## ✨ Features



\- Classifies messages as \*\*spam\*\* or \*\*ham\*\* (not spam)

\- Confidence score with probability breakdown

\- Train on your own CSV data or use the built-in sample dataset

\- Batch classification from a text file

\- Interactive CLI mode

\- Saves / loads trained models

\- Full test suite with `pytest`



\---



\## 🚀 Quick Start



\### 1. Clone \& install dependencies



```bash

git clone https://github.com/YOUR\_USERNAME/spam-detector.git

cd spam-detector

pip install -r requirements.txt

```



\### 2. Run the demo



```bash

python main.py --demo

```



\### 3. Try a single message



```bash

python main.py --message "Congratulations! You've won a FREE iPhone!"

```



\### 4. Interactive mode



```bash

python main.py

```



\---



\## 🎓 Training on Your Own Data



Prepare a CSV with two columns: `text` and `label` (values: `spam` or `ham`).



```bash

python main.py --train data/sample\_data.csv

```



A great free dataset to use: \[UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)



\---



\## 📂 Project Structure



```

spam-detector/

├── spam\_detector.py     # Core model (SpamDetector class + preprocessing)

├── main.py              # CLI entry point

├── tests.py             # Pytest test suite

├── requirements.txt     # Dependencies

└── data/

&#x20;   └── sample\_data.csv  # Example training data

```



\---



\## 🛠️ Usage as a Library



```python

from spam\_detector import SpamDetector



detector = SpamDetector()

detector.train\_on\_sample\_data()   # or detector.load("spam\_model.pkl")



result = detector.predict("You've won a cash prize! Call now.")

print(result)

\# {'label': 'spam', 'confidence': 0.9987, 'scores': {'ham': 0.0013, 'spam': 0.9987}}

```



\---



\## 🧪 Running Tests



```bash

pytest tests.py -v

```



\---



\## 📈 Improving Accuracy



\- Use the full \[UCI SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (\~5,500 messages)

\- Add stop-word filtering: `TfidfVectorizer(stop\_words='english')`

\- Try `LinearSVC` instead of `MultinomialNB` for larger datasets

\- Use `GridSearchCV` to tune hyperparameters



\---



\## 📄 License



MIT License — free to use and modify.

