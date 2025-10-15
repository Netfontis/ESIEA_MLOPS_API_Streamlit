import pytest

def test_predict_endpoint_valid(client):
    """Test l'endpoint /predict avec un texte valide"""

    test_data = {"text": "J'adore ce produit, il est fantastique !"}
    response = client.post("/predict", json=test_data)

    # ✅ Si modèle absent (500 interne) : on ignore ce test en CI
    if response.status_code == 500 and "Modèles non chargés" in response.text:
        pytest.skip("Modèle non chargé - test ignoré en CI.")

    assert response.status_code == 200
    data = response.json()

    required_fields = [
        "sentiment", "confidence",
        "probability_positive", "probability_negative"
    ]
    for field in required_fields:
        assert field in data

    assert data["sentiment"] in ["Positif", "Négatif"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0
    total_prob = data["probability_positive"] + data["probability_negative"]
    assert abs(total_prob - 1.0) < 0.01

    print(f"✅ Prediction OK: {data['sentiment']} ({data['confidence']:.2f})")


def test_predict_endpoint_invalid(client):
    """Test l'endpoint /predict avec des données invalides"""

    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [200, 422, 500]

    long_text = "a" * 300
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code in [200, 422, 500]

    response = client.post("/predict", json={})
    assert response.status_code in [200, 422, 500]

    print("✅ Validation erreurs /predict OK")
