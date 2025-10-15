import time
import pytest

def test_explain_endpoint_valid(client):
    """Test l'endpoint /explain avec un texte valide"""
    
    test_data = {"text": "Ce film est absolument terrible, je le déteste !"}
    response = client.post("/explain", json=test_data)
    
    # 👉 Gestion du cas où le modèle n'est pas encore chargé
    if response.status_code == 500 and "Modèles non chargés" in response.text:
        pytest.skip("Modèle non chargé - test ignoré dans CI.")
    
    assert response.status_code == 200

    data = response.json()
    assert "sentiment" in data
    assert "explanation" in data
    assert "html_explanation" in data

    assert isinstance(data["html_explanation"], str)
    assert "<div" in data["html_explanation"]
    assert len(data["html_explanation"]) > 50

    print("✅ Test /explain OK")
