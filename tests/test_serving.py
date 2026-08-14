import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    """Test health check endpoints return status 200 and expected keys."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "engine_ready" in data
    assert "num_indexed_vectors" in data

    response_alias = client.get("/health")
    assert response_alias.status_code == 200


def test_search_by_image_endpoint(client):
    """Test POST /api/search/image endpoint with a dummy image file upload."""
    # Create temporary PNG image bytes
    img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    response = client.post("/api/search/image?engine_type=resnet&top_k=3", files=files)

    if response.status_code == 53:  # Service Unavailable if indices missing
        pytest.skip("Search engine indices not initialized")

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "image"
    assert data["engine_type"] == "resnet"
    assert data["top_k"] == 3
    assert len(data["results"]) == 3
    assert "score" in data["results"][0]


def test_search_by_image_invalid_file(client):
    """Test POST /api/search/image rejects non-image files."""
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/api/search/image", files=files)
    assert response.status_code in (400, 503)


def test_search_by_text_endpoint(client):
    """Test POST /api/search/text endpoint."""
    payload = {"text_query": "a bird with red feathers", "top_k": 4}
    response = client.post("/api/search/text", json=payload)

    if response.status_code == 503:
        pytest.skip("Search engine indices not initialized")

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "text"
    assert data["engine_type"] == "clip"
    assert data["top_k"] == 4
    assert len(data["results"]) == 4


def test_get_metrics_endpoint(client):
    """Test GET /api/metrics endpoint."""
    response = client.get("/api/metrics")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
