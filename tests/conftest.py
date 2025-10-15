import pytest
from fastapi.testclient import TestClient
from main import app  # Assure-toi que "app" vient bien de ton main.py

@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI partagé entre tous les tests"""
    return TestClient(app)
