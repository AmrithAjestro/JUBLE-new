# JUBLIE — The Judicial Assistant

JUBLIE is a Streamlit app that helps users explore Indian law. It combines LLM chatbots (via [Groq](https://groq.com)) with a machine-learning model trained on past Supreme Court judgments to give a predicted outcome for a legal scenario.

## Features

- **Login portal**: local username/password accounts, stored as salted SHA-256 hashes in `user_db.json`.
- **Law-Jublie ⚖️**: a law-focused chatbot
  - answers questions grounded in the Constitution of India dataset (`dataset/coi.csv`)
  - summarises uploaded PDF documents
  - looks up fine amounts for Indian offences
  - exports the chat to PDF / DOCX
- **Judgment prediction**: describe a case (or upload a PDF/DOCX). A trained classifier predicts the likely decision, and the LLM explains it with reference to similar past cases and relevant law provisions.
- **Com-Jublie 💬**: a general-purpose chatbot with adjustable tone and response length, translation into other languages, and export to Markdown / text.

## Project structure

```
app.py                     # Streamlit entry point (UI, navigation, prediction page)
module/
  login.py                 # Login / registration
  law_chatbot.py           # Law-Jublie chatbot
  general_chatbot.py       # Com-Jublie chatbot
  llm.py                   # Shared Groq client, model fallback, JSON parsing
prediction_model.py        # ML + LLM judgment prediction
dataset_loader.py          # Dataset loading, similar-case / law retrieval, shared classifier class
legal_classifier_train.py  # Trains the SVC / RF / XGB ensemble → models/legal_classifier.pkl
dataset/
  fj.csv                   # Past judgments (title, issues, decision, cited cases)
  coi.csv                  # Constitution of India articles
models/                    # Trained models (created by legal_classifier_train.py)
```

## Setup

Requires Python 3.10+.

```bash
git lfs install
git clone https://github.com/AmrithAjestro/JUBLE-new.git
cd JUBLE-new
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored, so never commit it):

```env
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-120b
GROQ_MODEL=openai/gpt-oss-120b
```

## Train the prediction model

The prediction page loads `models/legal_classifier.pkl`. Create it by running:

```bash
python legal_classifier_train.py
```

## Run

```bash
python -m streamlit run app.py
```

Then open the URL that Streamlit prints (usually http://localhost:8501), register an account, and log in.

## Disclaimer

JUBLIE is for educational and research use only. Its predictions and answers are not legal advice.
