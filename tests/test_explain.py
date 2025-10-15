def test_explain_endpoint_valid(client):
    test_data = {"text": "Ce film est absolument terrible, je le déteste !"}
    response = client.post("/explain", json=test_data)
    if response.status_code == 500 and "Modèles non chargés" in response.text:
        pytest.skip("Modèle non chargé - test ignoré dans CI.")
    assert response.status_code == 200
