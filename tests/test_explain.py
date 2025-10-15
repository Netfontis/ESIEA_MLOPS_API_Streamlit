def test_explain_endpoint_valid(client):
    """Test l'endpoint /explain avec un texte valide"""
    
    test_data = {"text": "Ce film est absolument terrible, je le déteste !"}
    response = client.post("/explain", json=test_data)
    
    # 👉 Ajout temporaire pour debug :
    if response.status_code != 200:
        print("❌ Réponse serveur :", response.text)
    
    assert response.status_code == 200
