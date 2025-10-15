import time
import pytest

def test_explain_endpoint_valid(client):
    """Test l'endpoint /explain avec un texte valide"""
    
    test_data = {"text": "Ce film est absolument terrible, je le déteste !"}
    
    start_time = time.time()
    response = client.post("/explain", json=test_data)
    duration = time.time() - start_time
    
    assert response.status_code == 200
    
    data = response.json()
    required_fields = ["sentiment", "explanation", "html_explanation"]
    for field in required_fields:
        assert field in data
    
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    
    html = data["html_explanation"]
    assert isinstance(html, str)
    assert len(html) > 100
    assert "<div" in html
    
    assert duration < 120  # 2 min max
    
    print(f"✅ LIME OK ({len(data['explanation'])} mots) — {duration:.1f}s")


@pytest.mark.timeout(90)
def test_explain_endpoint_robustness(client):
    """Test la robustesse de /explain avec divers textes"""
    
    test_cases = [
        "Super !",
        "😊" * 10,
        "http://example.com test"
    ]
    
    for text in test_cases:
        response = client.post("/explain", json={"text": text})
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            assert "html_explanation" in data
    
    print("✅ Robustesse LIME OK")
