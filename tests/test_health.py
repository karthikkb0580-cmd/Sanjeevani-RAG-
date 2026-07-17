"""tests/test_health.py – Health endpoint tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_structure(client: AsyncClient):
    response = await client.get("/health")
    data = response.json()

    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "dependencies" in data
    assert "qdrant" in data["dependencies"]


@pytest.mark.asyncio
async def test_health_service_name(client: AsyncClient):
    response = await client.get("/health")
    data = response.json()
    assert "Sanjeevani" in data["service"]


@pytest.mark.asyncio
async def test_health_config_fields(client: AsyncClient):
    response = await client.get("/health")
    data = response.json()
    config = data.get("config", {})
    assert "embedding_model" in config
    assert "chat_model" in config
    assert "collection" in config
