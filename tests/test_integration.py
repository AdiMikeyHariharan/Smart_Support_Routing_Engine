import asyncio
import pytest
from httpx import AsyncClient
from smart_support.main import app


@pytest.mark.asyncio
async def test_ticket_m3_full_flow():
    """
    Integration test for Milestone 3 endpoint:
    1. Submit ticket to /ticket
    2. Poll /job/{job_id}
    3. Validate routing output
    """

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/ticket",
            json={
                "subject": "Critical Outage",
                "description": "Entire production system is down"
            }
        )

        assert response.status_code == 202

        data = response.json()
        job_id = data["job_id"]

        # Poll job status
        for _ in range(15):
            job_response = await ac.get(f"/job/{job_id}")
            job_data = job_response.json()

            if job_data["status"] == "finished":
                break

            await asyncio.sleep(1)

        assert job_data["status"] == "finished"
        assert "assigned_agent" in job_data["result"]
        assert "category" in job_data["result"]
        assert "urgency" in job_data["result"]