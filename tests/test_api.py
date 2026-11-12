import pytest
from fastapi.testclient import TestClient
from src.api import app
import src.auth as auth


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """Test /health endpoint returns healthy status and metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "sop_count" in data
    assert data["sop_count"] >= 18
    assert "llm_provider" in data
    assert "embedding_provider" in data


def test_api_list_sops(client):
    """Test /api/v1/sops lists active SOP definitions."""
    resp = client.get("/api/v1/sops")
    assert resp.status_code == 200
    sops = resp.json()
    assert isinstance(sops, list)
    assert len(sops) >= 18
    assert any(s["id"] == "SOP-OVERRIDE-001" for s in sops)


def test_api_reload_sops(client):
    """Test /api/v1/sops/reload forces dynamic hot-reload."""
    resp = client.post("/api/v1/sops/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["sop_count"] >= 18


def test_api_auth_gate(client, monkeypatch):
    """Test that setting ACCESS_KEY protects endpoints with 401."""
    monkeypatch.setattr(auth, "ACCESS_KEY", "prod-secret-999")
    try:
        # Without key -> 401
        resp_unauth = client.get("/api/v1/sops")
        assert resp_unauth.status_code == 401

        # With wrong key -> 401
        resp_wrong = client.get("/api/v1/sops", headers={"X-API-Key": "wrong-secret"})
        assert resp_wrong.status_code == 401

        # With valid key -> 200
        resp_ok = client.get("/api/v1/sops", headers={"X-API-Key": "prod-secret-999"})
        assert resp_ok.status_code == 200
    finally:
        monkeypatch.setattr(auth, "ACCESS_KEY", "")
