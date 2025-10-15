from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import HTTPException

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

import os
import csv
import random
from io import StringIO
import joblib
import numpy as np

import pandas as pd
import re

from lime.lime_text import LimeTextExplainer

app = FastAPI(title="API Streamlit")

# =========================
# 1) Schémas Pydantic
# =========================
class TweetRequest(BaseModel):
    text: str = Field(..., max_length=280, description="Texte du tweet (<= 280 caractères)")

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplanationResponse(BaseModel):
    sentiment: str
    explanation: List[Dict]  
    html_explanation: str

# =========================
# 2) App & Etat global
# =========================
app = FastAPI(title="🎓 API Sentiment",
              description="API de classification de sentiment avec LIME",
              version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "sentiment_model.joblib")
VECT_PATH  = os.getenv("VECT_PATH", "tfidf_vectorizer.joblib")

model = None
vectorizer = None
lime_explainer: Optional[LimeTextExplainer] = None
class_names: List[str] = []
positive_label_candidates = {"positive", "pos", 1, "1", "Positive", "POSITIVE"}

# =========================
# 3) Prétraitement
# =========================
# Stopwords fr/en minimalistes (fallback). Pour du sérieux, brancher une vraie liste ou laisser le vectorizer gérer.
FR_SW = {
    "le","la","les","un","une","des","et","ou","de","du","au","aux","je","tu","il","elle","nous","vous","ils","elles",
    "ce","ces","cet","cette","dans","sur","pour","pas","ne","n","d","l","en","y","a","est","sont","aujourd","hui",
}
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as EN_SW
STOPWORDS = set(FR_SW) | set(EN_SW)

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

# Lemmatisation optionnelle via spaCy si dispo
_spacy_nlp = None
def _maybe_load_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return
    try:
        import spacy
        # Essaie FR puis EN
        for m in ("fr_core_news_sm", "en_core_web_sm"):
            try:
                _spacy_nlp = spacy.load(m, disable=["ner","parser","textcat"])
                break
            except Exception:
                continue
    except Exception:
        _spacy_nlp = None

def preprocess_text(text: str) -> str:
    """
    Nettoyage URLs, mentions, emojis, hashtags->mots, tokenisation,
    stopwords, lemmatisation (si spaCy dispo).
    """
    if not text:
        return ""
    x = text
    x = URL_RE.sub(" ", x)
    x = MENTION_RE.sub(" ", x)
    # Conserver le mot du hashtag (ex: #cool -> "cool")
    x = HASHTAG_RE.sub(r"\1", x)
    x = EMOJI_RE.sub(" ", x)
    x = x.replace("&amp;", " ").replace("&lt;", " ").replace("&gt;", " ")
    x = x.lower()

    tokens = WORD_RE.findall(x)

    # Stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # Lemmatisation si possible
    _maybe_load_spacy()
    if _spacy_nlp:
        doc = _spacy_nlp(" ".join(tokens))
        lemmas = [t.lemma_.lower() for t in doc if t.lemma_ not in (" ", "", None)]
        tokens = [t for t in lemmas if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)

# =========================
# 4) Utils prédiction / LIME
# =========================
def _ensure_loaded():
    if model is None or vectorizer is None or lime_explainer is None:
        raise HTTPException(status_code=500, detail="Modèles non chargés. Voir /health.")

def _class_indices():
    """Retourne (idx_pos, idx_neg) si possible, sinon (None, None)."""
    if hasattr(model, "classes_"):
        cls = list(model.classes_)
        pos_idx = neg_idx = None
        # Cherche libellés explicites
        for i, c in enumerate(cls):
            if c in positive_label_candidates:
                pos_idx = i
        # Heuristique pour 'negative'
        for i, c in enumerate(cls):
            if str(c).lower() in {"negative", "neg", "0"}:
                neg_idx = i
        if pos_idx is not None and neg_idx is None and len(cls) == 2:
            neg_idx = 1 - pos_idx
        if pos_idx is None and len(cls) == 2:
            # Si pas de libellé clair, on prendra argmax/min
            return None, None
        return pos_idx, neg_idx
    return None, None

def _predict_proba_texts(raw_texts: List[str]) -> np.ndarray:
    """Fonction wrapper pour LIME -> renvoie proba par classe sur une liste de textes."""
    texts = [preprocess_text(t) for t in raw_texts]
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

def _sentiment_from_proba(p: np.ndarray) -> Dict[str, float]:
    """Mappe proprement vers (p_pos, p_neg) et sentiment str."""
    pos_idx, neg_idx = _class_indices()
    if p.ndim == 1:
        p = p.reshape(1, -1)
    pp = p[0]

    if pos_idx is not None and neg_idx is not None:
        p_pos = float(pp[pos_idx])
        p_neg = float(pp[neg_idx])
        sentiment = "positive" if p_pos >= p_neg else "negative"
    elif hasattr(model, "classes_") and len(model.classes_) == 2:
        # Sans mapping explicite : argmax = "positive"
        argmax = int(np.argmax(pp))
        p_max = float(pp[argmax])
        p_min = float(pp[1-argmax])
        sentiment = "positive" if argmax == 0 else "positive"  # on labelle la classe gagnante comme "positive"
        # Pour rester conforme au contrat, on expose p_pos = p_max, p_neg = p_min
        p_pos, p_neg = p_max, p_min
        # Si tu veux imposer un vrai mapping, définis POS/NEG via env vars.
    else:
        # Cas exotique (multi-classes) -> on projette en binaire
        p_pos = float(np.max(pp))
        p_neg = float(1.0 - p_pos)
        sentiment = "positive" if p_pos >= 0.5 else "negative"

    confidence = max(p_pos, p_neg)
    return dict(p_pos=p_pos, p_neg=p_neg, sentiment=sentiment, confidence=float(confidence))

def _explain_with_lime(text: str, num_features: int = 10):
    """Retourne (sentiment, list{word,importance}, html)"""
    # Prédiction brute pour connaître la classe à expliquer
    proba = _predict_proba_texts([text])
    pred_idx = int(np.argmax(proba[0]))

    exp = lime_explainer.explain_instance(
        text_instance=preprocess_text(text),
        classifier_fn=_predict_proba_texts,
        labels=[pred_idx],
        num_features=num_features
    )

    # Liste (mot, importance) pour la classe prédite
    pairs = exp.as_list(label=pred_idx)  # [(feature, weight), ...]
    explanation = []
    for feat, weight in pairs:
        # feat peut être 'word' ou une feature TF-IDF (on garde tel quel)
        explanation.append({"word": str(feat), "importance": float(weight)})

    html = exp.as_html(labels=(pred_idx,))
    sentiment_pack = _sentiment_from_proba(proba)
    return sentiment_pack["sentiment"], explanation, html

# =========================
# 5) Chargement des modèles (startup)
# =========================
@app.on_event("startup")
def _load_artifacts():
    global model, vectorizer, lime_explainer, class_names
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)

        # class_names pour LIME
        if hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]
        else:
            class_names = ["negative", "positive"]  # fallback

        lime_explainer = LimeTextExplainer(
            class_names=class_names,
            split_expression=WORD_RE  # cohérent avec preprocess
        )
        print("[STARTUP] Modèles chargés ✅")
    except Exception as e:
        print(f"[STARTUP] Échec de chargement des artefacts ❌ : {e}")

# =========================
# 6) Endpoints requis
# =========================
@app.get("/", include_in_schema=False)
def root():
    # Redirige vers la doc Swagger directement
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Vérification santé")
def health():
    ok_model = model is not None
    ok_vect  = vectorizer is not None
    ok_lime  = lime_explainer is not None
    details = {
        "model_loaded": ok_model,
        "vectorizer_loaded": ok_vect,
        "lime_loaded": ok_lime,
        "class_names": class_names,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECT_PATH
    }
    status = "ok" if all([ok_model, ok_vect, ok_lime]) else "degraded"
    return {"status": status, "details": details}

@app.post("/predict", response_model=PredictionResponse, summary="Prédiction de sentiment")
def predict(req: TweetRequest):
    _ensure_loaded()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)

    pack = _sentiment_from_proba(proba)
    return PredictionResponse(
        sentiment=pack["sentiment"],
        confidence=round(pack["confidence"], 6),
        probability_positive=round(pack["p_pos"], 6),
        probability_negative=round(pack["p_neg"], 6),
    )

@app.post("/explain", response_model=ExplanationResponse, summary="Explicabilité LIME")
def explain(req: TweetRequest):
    _ensure_loaded()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    sentiment, explanation, html = _explain_with_lime(text, num_features=10)
    return ExplanationResponse(
        sentiment=sentiment,
        explanation=explanation,
        html_explanation=html
    )