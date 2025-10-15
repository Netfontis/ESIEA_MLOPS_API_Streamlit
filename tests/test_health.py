def test_health_endpoint(client):
    """Test l'endpoint /health (API opérationnelle)"""
    
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    
    # Si ton endpoint renvoie d'autres infos
    if "model_loaded" in data:
        assert data["model_loaded"] is True
    
    print("✅ Health check OK - API vivante")
