def test_health_endpoint(client):
    """Test l'endpoint /health (API opérationnelle)"""
    
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data

    # Accepte 'ok' (modèle chargé) ou 'degraded' (modèle manquant)
    assert data["status"] in ["ok", "degraded"]

    print(f"✅ /health OK - status = {data['status']}")
