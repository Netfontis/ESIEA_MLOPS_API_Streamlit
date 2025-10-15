import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI (mocke le modèle si absent)."""
    # ✅ Si ton app a un attribut global model = None, on simule un modèle factice
    if hasattr(app.state, "model") and app.state.model is None:
        class FakeModel:
            def predict(self, X): return ["Positif"]
            def predict_proba(self, X): return [[0.2, 0.8]]
        app.state.model = FakeModel()
        print("⚙️ Modèle simulé chargé pour les tests.")
    return TestClient(app)

'''
import pytest
from fastapi.testclient import TestClient
from main import app  # Assure-toi que "app" vient bien de ton main.py

@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI partagé entre tous les tests"""
    return TestClient(app)
'''