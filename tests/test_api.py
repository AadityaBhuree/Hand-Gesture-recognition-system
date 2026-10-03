import pytest
import io
import json
import base64
from PIL import Image
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

def test_index_route_accept_json(client):
    response = client.get('/', headers={'Accept': 'application/json'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'Handwritten Digit Recognition API' in data['message']
    assert 'GET /predict' in data['endpoints']

def test_api_spec_route(client):
    response = client.get('/api')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'endpoints' in data

def test_predict_get(client):
    response = client.get('/predict')
    # Can be 200 or 503 depending on if model is available locally during test
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'predicted' in data
        assert 'ground_truth' in data
        assert isinstance(data['probabilities'], list)
        assert len(data['probabilities']) == 10

def test_predict_post_no_image(client):
    response = client.post('/predict', data={})
    assert response.status_code in [400, 503]

def test_predict_post_invalid_base64(client):
    payload = json.dumps({'image': 'data:image/png;base64'})  # missing comma
    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code in [400, 503]
    if response.status_code == 400:
        assert b'Invalid base64 string format' in response.data

def test_predict_post_corrupted_file(client):
    corrupted_data = b'NOT_A_VALID_IMAGE_FILE_HEADER_GARBAGE'
    response = client.post(
        '/predict',
        data={'file': (io.BytesIO(corrupted_data), 'corrupted.png')},
        content_type='multipart/form-data'
    )
    assert response.status_code in [400, 503]
    if response.status_code == 400:
        data = json.loads(response.data)
        assert 'Cannot process the uploaded file' in data['error']

def test_predict_post_corrupted_base64_payload(client):
    # Valid base64 encoding of non-image binary data
    garbage_b64 = base64.b64encode(b'RAW_NON_IMAGE_CORRUPTED_BYTES').decode('utf-8')
    payload = json.dumps({'image': f'data:image/png;base64,{garbage_b64}'})
    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code in [400, 503]
    if response.status_code == 400:
        data = json.loads(response.data)
        assert 'Failed to decode base64 image' in data['error']

def test_predict_post_valid_canvas_base64(client):
    # Generate a genuine 28x28 black image with white stroke (like user drawing canvas)
    img = Image.new('L', (28, 28), color=0)
    for i in range(10, 20):
        img.putpixel((14, i), 255)  # draw vertical stroke for '1'
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    payload = json.dumps({'image': f'data:image/png;base64,{b64_str}'})

    response = client.post('/predict', data=payload, content_type='application/json')
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'predicted' in data
        assert isinstance(data['predicted'], int)
        assert 0 <= data['predicted'] <= 9
        assert len(data['probabilities']) == 10
        assert 0.99 <= sum(data['probabilities']) <= 1.01

def test_large_payload(client):
    # Simulate a file larger than 5MB
    large_data = b'0' * (6 * 1024 * 1024)
    response = client.post(
        '/predict',
        data={'file': (io.BytesIO(large_data), 'large.png')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 413
    assert b'File too large' in response.data
