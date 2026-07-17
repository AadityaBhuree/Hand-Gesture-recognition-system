import pytest
import io
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Handwritten Digit Recognition API' in response.data

def test_predict_get(client):
    response = client.get('/predict')
    # Can be 200 or 503 depending on if model is available locally during test
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'predicted' in data
        assert 'ground_truth' in data

def test_predict_post_no_image(client):
    response = client.post('/predict', data={})
    assert response.status_code in [400, 503]

def test_predict_post_invalid_base64(client):
    payload = json.dumps({'image': 'data:image/png;base64'})  # missing comma
    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code in [400, 503]
    if response.status_code == 400:
        assert b'Invalid base64 string format' in response.data

def test_large_payload(client):
    # Simulate a file larger than 5MB
    large_data = b'0' * (6 * 1024 * 1024)
    response = client.post('/predict', 
                           data={'file': (io.BytesIO(large_data), 'large.png')},
                           content_type='multipart/form-data')
    assert response.status_code == 413
    assert b'File too large' in response.data
