"""
Dataset access and retrieval for fj.csv (past judgments) and coi.csv (Constitution of India).

Also holds `build_case_text` and `LegalClassifier`, which training and prediction share so
the saved model unpickles from an importable module and sees identical features both times.
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dataset"
LABELS = {0: "Against Petitioner", 1: "In Favor of Petitioner"}

_WS = re.compile(r"\s+")
_ARTICLE_REF = re.compile(r"\barticles?\s+(\d+[a-z]{0,3})\b", re.I)


def clean_text(text) -> str:
    """Collapse whitespace; non-strings (e.g. NaN) become ''."""
    return _WS.sub(" ", text).strip() if isinstance(text, str) else ""


@lru_cache(maxsize=None)
def load_dataset(filename: str) -> pd.DataFrame:
    """Load and clean a CSV from the dataset folder (cached — do not mutate the result)."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"[⚠️ WARN] File not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[❌ ERROR] Failed to read {file_path}: {e}")
        return pd.DataFrame()

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(clean_text)
    return df


@lru_cache(maxsize=1)
def coi_lookup() -> dict:
    """Article number (lower-case, e.g. '21a') -> 'Title. Description'."""
    coi = load_dataset("coi.csv")
    if coi.empty:
        return {}
    return {
        str(a).strip().lower(): f"{t}. {d}"
        for a, t, d in zip(coi["article"], coi["title"], coi["description"])
    }


def article_refs(text: str) -> list[str]:
    """Unique Article numbers mentioned in text, in order (e.g. ['21', '14'])."""
    return list(dict.fromkeys(m.lower() for m in _ARTICLE_REF.findall(text or "")))


def build_case_text(text: str, max_desc: int = 400) -> str:
    """Enrich text with the Constitution text of every Article it cites."""
    lookup = coi_lookup()
    extra = [lookup[ref][:max_desc] for ref in article_refs(text) if ref in lookup]
    return " ".join([text, *extra])


class LegalClassifier:
    """TF-IDF + weighted soft-voting ensemble over raw scenario text (trained by legal_classifier_train.py)."""

    def __init__(self, vec, models: dict, weights: dict, classes_):
        self.vec = vec
        self.models = models
        self.weights = weights
        self.classes_ = np.array(classes_)

    def predict_proba(self, texts):
        X = self.vec.transform([build_case_text(t) for t in texts])
        total = sum(self.weights.get(n, 1.0) for n in self.models)
        return sum(m.predict_proba(X) * self.weights.get(n, 1.0) for n, m in self.models.items()) / total

    def predict(self, texts):
        return self.classes_[np.argmax(self.predict_proba(texts), axis=1)]


# -----------------------
# Relevance retrieval
# -----------------------
def _tfidf_index(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    return vec, vec.fit_transform(docs)


@lru_cache(maxsize=1)
def _case_index():
    fj = load_dataset("fj.csv")
    return _tfidf_index((fj["case title"] + " " + fj["issues"]).map(build_case_text))


@lru_cache(maxsize=1)
def _law_index():
    coi = load_dataset("coi.csv")
    return _tfidf_index(coi["title"] + " " + coi["description"])


def _top_k(index, query: str, k: int) -> list[int]:
    vec, matrix = index
    scores = (matrix @ vec.transform([query]).T).toarray().ravel()
    return [i for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]


def get_past_cases(limit: int = 10, query: str | None = None) -> list[dict]:
    """Past judgments most similar to `query` (or the first `limit` rows if no query)."""
    fj = load_dataset("fj.csv")
    if fj.empty:
        return []
    rows = _top_k(_case_index(), build_case_text(query), limit) if query else range(min(limit, len(fj)))
    return [
        {
            "case_title": fj.iloc[i]["case title"],
            "issues": fj.iloc[i]["issues"],
            "decision": LABELS.get(fj.iloc[i]["decision"], str(fj.iloc[i]["decision"])),
        }
        for i in rows
    ]


def get_law_provisions(limit: int = 10, query: str | None = None, max_desc: int = 500) -> list[dict]:
    """Constitution articles explicitly cited in `query` first, then the most textually relevant."""
    coi = load_dataset("coi.csv")
    if coi.empty:
        return []
    if query:
        articles = coi["article"].astype(str).str.lower()
        cited = [i for ref in article_refs(query) for i in np.flatnonzero(articles == ref)]
        rows = list(dict.fromkeys(cited + _top_k(_law_index(), query, limit)))[:limit]
    else:
        rows = range(min(limit, len(coi)))
    return [
        {
            "article": coi.iloc[i]["article"],
            "title": coi.iloc[i]["title"],
            "description": coi.iloc[i]["description"][:max_desc],
        }
        for i in rows
    ]


if __name__ == "__main__":
    q = "A citizen was detained without trial for 6 months under preventive detention laws."
    print("\n[INFO] Similar past cases:")
    for case in get_past_cases(3, q):
        print(case)
    print("\n[INFO] Relevant law provisions:")
    for law in get_law_provisions(3, q):
        print(law["article"], law["title"])
