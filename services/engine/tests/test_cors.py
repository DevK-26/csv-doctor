"""Phase C2 tests: the browser app (a different origin) may call the engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)
ALLOWED = main.WEB_ORIGINS[0]


def preflight(path: str, origin: str, method: str = "POST", headers: str = "content-type"):
    return client.options(path, headers={
        "Origin": origin, "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": headers})


@pytest.mark.parametrize("path", ["/analyse", "/ask", "/jobs"])
def test_preflight_allows_the_web_origin(path):
    resp = preflight(path, ALLOWED)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == ALLOWED
    assert "POST" in resp.headers["access-control-allow-methods"]
    assert "content-type" in resp.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in resp.headers


def test_preflight_rejects_other_origins():
    resp = preflight("/analyse", "https://evil.example")
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_actual_response_carries_the_allow_origin_header():
    resp = client.get("/health", headers={"Origin": ALLOWED})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == ALLOWED


def test_default_origin_is_the_web_dev_server():
    assert main.parse_web_origins(None) == ["http://localhost:3000"]
    assert main.parse_web_origins("  ") == ["http://localhost:3000"]


def test_origins_come_from_a_comma_separated_value():
    got = main.parse_web_origins("https://csvdoctor.example/, http://127.0.0.1:3000")
    assert got == ["https://csvdoctor.example", "http://127.0.0.1:3000"]


def test_wildcard_origin_is_refused():
    with pytest.raises(ValueError, match="exact origins"):
        main.parse_web_origins("*")
