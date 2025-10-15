import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI (mocke le modèle si absent)."""
    if not hasattr(app.state, "model") or app.state.model is None:
        class FakeModel:
            def predict(self, X): return ["Positif"]
            def predict_proba(self, X): return [[0.3, 0.7]]
        app.state.model = FakeModel()
        print("⚙️ Modèle simulé chargé pour les tests.")
    return TestClient(app)
