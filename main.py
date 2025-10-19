from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi import HTTPException

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as EN_SW


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

MODEL_PATH = os.getenv("MODEL_PATH", "api_artifacts/sentiment_model.joblib")
VECT_PATH  = os.getenv("VECT_PATH", "api_artifacts/tfidf_vectorizer.joblib")

model = None
vectorizer = None
lime_explainer = None
class_names = []

# diagnostics de chargement
model_raw_type = None
model_resolved_type = None
load_warning = None

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

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def _softmax(scores):
    s = np.asarray(scores, dtype=float)
    s = s - np.max(s, axis=1, keepdims=True)  # stabilité num.
    e = np.exp(s)
    return e / np.sum(e, axis=1, keepdims=True)

def _predict_proba_safely(X):
    """
    Retourne des probabilités même si le modèle n'expose pas predict_proba,
    en se rabattant sur decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        # binaire → 1 colonne ou 2 colonnes selon l’estimateur
        if np.ndim(scores) == 1:
            # marge binaire, transforme en proba via sigmoid
            p_pos = _sigmoid(scores)
            return np.c_[1.0 - p_pos, p_pos]

        # multi-classes → softmax
        return _softmax(scores)

    raise HTTPException(
        status_code=500,
        detail="Le modèle n’expose ni predict_proba ni decision_function."
    )


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


# ========= Helpers proba =========

import numpy as np

def getCoreEstimator(modelObj):
    """
    Remonte jusqu'à l'estimateur final (grid.best_estimator_, pipeline[-1], etc.)
    """
    seen = set()
    m = modelObj

    while True:
        key = id(m)
        if key in seen:
            break
        seen.add(key)

        if hasattr(m, "best_estimator_") and m.best_estimator_ is not None:
            m = m.best_estimator_
            continue

        if hasattr(m, "named_steps"):
            try:
                # Pipeline sklearn
                last = list(m.named_steps.values())[-1]
                m = last
                continue
            except Exception:
                pass

        if hasattr(m, "steps"):
            try:
                m = m.steps[-1][1]
                continue
            except Exception:
                pass

        if hasattr(m, "estimator"):
            m = m.estimator
            continue

        if hasattr(m, "base_estimator"):
            m = m.base_estimator
            continue

        break

    return m


def scoresToProba(scores):
    s = np.asarray(scores, dtype=float)

    # Binaire : score de marge → sigmoid
    if s.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.c_[1.0 - p1, p1]

    # Multi-classes : softmax
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=1, keepdims=True)


def predictProbaSafely(modelObj, X):
    """
    Renvoie des probabilités pour tout type de wrapper sklearn :
    - utilise predict_proba si dispo
    - sinon decision_function + (sigmoid/softmax)
    - sinon fallback 0/1 à partir de predict()
    """
    core = getCoreEstimator(modelObj)

    if hasattr(modelObj, "predict_proba"):
        return modelObj.predict_proba(X)

    if hasattr(modelObj, "decision_function"):
        return scoresToProba(modelObj.decision_function(X))

    if hasattr(core, "predict_proba"):
        return core.predict_proba(X)

    if hasattr(core, "decision_function"):
        return scoresToProba(core.decision_function(X))

    # Dernier recours : probas déterministes (0/1) via predict()
    if hasattr(modelObj, "predict"):
        y = modelObj.predict(X)
    else:
        y = core.predict(X)

    classes = getattr(modelObj, "classes_", getattr(core, "classes_", np.array([0, 1])))
    classes = list(classes)

    out = np.zeros((len(y), len(classes)))
    for i, yi in enumerate(y):
        j = classes.index(yi)
        out[i, j] = 1.0

    return out

# --- Helpers d'empaquetage & proba ---

import numpy as np
from typing import Any

def unwrap_model(obj: Any) -> Any:
    """
    Tente d'extraire l'estimateur "coeur" depuis différents empaquetages :
    - dict: clés usuelles ('model', 'pipeline', 'estimator', 'clf', 'classifier') ou 1er objet ayant predict/_proba
    - tuple/list: dernier élément ou celui qui a predict/_proba
    - Pipeline/GridSearchCV/etc: on laisse tel quel, géré plus bas par get_core_estimator()
    """
    m = obj

    # dict
    if isinstance(m, dict):
        for key in ("model", "pipeline", "estimator", "clf", "classifier"):
            if key in m:
                return m[key]
        # sinon, 1ère valeur ressemblant à un estimateur
        for v in m.values():
            if hasattr(v, "predict") or hasattr(v, "predict_proba") or hasattr(v, "decision_function"):
                return v
        return m

    # tuple / list
    if isinstance(m, (list, tuple)):
        # on préfère un objet ayant predict/_proba; sinon le dernier
        pick = None
        for v in m:
            if hasattr(v, "predict") or hasattr(v, "predict_proba") or hasattr(v, "decision_function"):
                pick = v
        return pick if pick is not None else m[-1]

    return m


def get_core_estimator(model_obj: Any) -> Any:
    """
    Remonte jusqu'à l'estimateur final à l'intérieur d'un wrapper (Pipeline, GridSearchCV, OneVsRest, etc.).
    """
    seen = set()
    m = model_obj

    while True:
        if id(m) in seen:
            break
        seen.add(id(m))

        # GridSearchCV / RandomizedSearchCV
        if hasattr(m, "best_estimator_") and m.best_estimator_ is not None:
            m = m.best_estimator_
            continue

        # Pipeline sklearn
        if hasattr(m, "named_steps"):
            try:
                m = list(m.named_steps.values())[-1]
                continue
            except Exception:
                pass
        if hasattr(m, "steps"):
            try:
                m = m.steps[-1][1]
                continue
            except Exception:
                pass

        # méta-estimateurs
        if hasattr(m, "estimator"):
            m = m.estimator
            continue
        if hasattr(m, "base_estimator"):
            m = m.base_estimator
            continue

        break

    return m


def scores_to_proba(scores: Any) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim == 1:  # binaire
        p1 = 1.0 / (1.0 + np.exp(-s))  # sigmoid
        return np.c_[1.0 - p1, p1]
    # multi-classes : softmax
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=1, keepdims=True)


def predict_proba_safely(model_obj: Any, X) -> np.ndarray:
    """
    Renvoie des probabilités, quel que soit le wrapper :
    - predict_proba si dispo
    - sinon decision_function → proba (sigmoid/softmax)
    - sinon fallback 0/1 via predict()
    """
    core = get_core_estimator(model_obj)

    # direct sur l'objet haut-niveau
    if hasattr(model_obj, "predict_proba"):
        return model_obj.predict_proba(X)
    if hasattr(model_obj, "decision_function"):
        return scores_to_proba(model_obj.decision_function(X))

    # sinon sur l'estimateur coeur
    if hasattr(core, "predict_proba"):
        return core.predict_proba(X)
    if hasattr(core, "decision_function"):
        return scores_to_proba(core.decision_function(X))

    # dernier recours : proba déterministes depuis predict()
    target = model_obj if hasattr(model_obj, "predict") else core
    if not hasattr(target, "predict"):
        raise RuntimeError("Aucune méthode predict/predict_proba/decision_function trouvée sur le modèle chargé.")

    y = target.predict(X)
    classes = getattr(model_obj, "classes_", getattr(core, "classes_", np.array([0, 1])))
    classes = list(classes)

    out = np.zeros((len(y), len(classes)))
    for i, yi in enumerate(y):
        j = classes.index(yi)
        out[i, j] = 1.0
    return out

def safe_hasattr(obj, attr: str) -> bool:
    return (obj is not None) and hasattr(obj, attr)

def safe_core_estimator(obj):
    return get_core_estimator(obj) if obj is not None else None


# =========================
# 5) Chargement des modèles (startup)
# =========================
@app.on_event("startup")
def _load_artifacts():
    global model, vectorizer, lime_explainer, class_names
    global model_raw_type, model_resolved_type, load_warning
    load_warning = None
    model_raw_type = None
    model_resolved_type = None

    try:
        import joblib, os
        from lime.lime_text import LimeTextExplainer

        # 1) charge brut
        raw = joblib.load(MODEL_PATH)
        model_raw_type = type(raw).__name__

        # 2) unwrap si dict/tuple/etc.
        unwrapped = unwrap_model(raw)
        model = unwrapped
        model_resolved_type = type(model).__name__

        # 3) vectorizer (si fourni via VECT_PATH)
        if VECT_PATH and os.path.exists(VECT_PATH):
            vectorizer = joblib.load(VECT_PATH)
        else:
            vectorizer = None  # pipeline possible

        # 4) classes
        class_names_local = getattr(model, "classes_", None)
        if class_names_local is None and hasattr(get_core_estimator(model), "classes_"):
            class_names_local = getattr(get_core_estimator(model), "classes_", None)
        class_names[:] = [str(c) for c in (class_names_local if class_names_local is not None else ["negative", "positive"])]

        # 5) LIME
        lime_explainer = LimeTextExplainer(class_names=class_names, split_expression=WORD_RE)

        # 6) avertissement si le "modèle" ressemble à un objet non-estimateur
        suspicious = {"Series", "DataFrame", "ndarray"}
        if model_resolved_type in suspicious:
            load_warning = (f"Le fichier MODEL_PATH contient un objet {model_resolved_type} — ce n'est pas un estimateur. "
                            f"Vérifie l'artefact sauvegardé (attendu: Pipeline/Classifier).")

        print(f"[STARTUP] raw={model_raw_type}, resolved={model_resolved_type}, warning={load_warning}")

    except Exception as e:
        print(f"[STARTUP] Échec de chargement: {e}")
        raise

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

    core = safe_core_estimator(model)

    details = {
        "model_loaded": ok_model,
        "vectorizer_loaded": ok_vect,
        "lime_loaded": ok_lime,

        "class_names": class_names,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECT_PATH,

        # diagnostics robustes
        "model_raw_type": model_raw_type,
        "model_resolved_type": model_resolved_type,
        "model_has_predict_proba": safe_hasattr(model, "predict_proba") or safe_hasattr(core, "predict_proba"),
        "model_has_decision_function": safe_hasattr(model, "decision_function") or safe_hasattr(core, "decision_function"),
        "model_has_predict": safe_hasattr(model, "predict") or safe_hasattr(core, "predict"),
        "model_n_features_in": getattr(core, "n_features_in_", None) if core is not None else None,
        "vectorizer_type": (type(vectorizer).__name__ if vectorizer is not None else None),
        "vectorizer_vocab_size": (len(getattr(vectorizer, "vocabulary_", {})) if vectorizer is not None else None),
        "load_warning": load_warning,
    }

    return {"status": "ok", "details": details}

@app.post("/predict", response_model=PredictionResponse, summary="Prédiction de sentiment")
def predict(req: TweetRequest):
    _ensure_loaded()

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    cleaned = preprocess_text(text)

    # pipeline ou vectorizer séparé
    if vectorizer is None and (hasattr(model, "transform") or hasattr(model, "predict_proba")):
        # pipeline (le modèle gère la vectorisation)
        try:
            proba = predict_proba_safely(model, [cleaned])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Échec predict (pipeline): {type(e).__name__}: {e}")
    else:
        # vectorizer + modèle séparés
        X = vectorizer.transform([cleaned])

        core = safe_core_estimator(model)
        if not any([safe_hasattr(model, "predict_proba"), safe_hasattr(model, "decision_function"), safe_hasattr(model, "predict"),
                    safe_hasattr(core, "predict_proba"), safe_hasattr(core, "decision_function"), safe_hasattr(core, "predict")]):
            raise HTTPException(
                status_code=500,
                detail=f"Artefact invalide: objet chargé de type '{model_resolved_type}' (attendu: Pipeline/Classifier)."
            )


        n_vec = X.shape[1]
        n_mod = getattr(get_core_estimator(model), "n_features_in_", None)
        if n_mod is not None and n_vec != n_mod:
            raise HTTPException(
                status_code=500,
                detail=(f"Incompatibilité vectorizer/modèle: X a {n_vec} features, "
                        f"le modèle en attend {n_mod}. Utilise le vectorizer "
                        f"créé avec CE modèle.")
            )
        try:
            proba = predict_proba_safely(model, X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Échec predict: {type(e).__name__}: {e}")

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
