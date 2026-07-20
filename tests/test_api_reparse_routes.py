import pytest

from financial_dashboard.schemas.emails import ReparseEmailResponse
from financial_dashboard.schemas.sms import ReparseSmsResponse

pytestmark = pytest.mark.anyio


async def test_api_sms_reparse_forwards_to_canonical_operation(client, monkeypatch):
    calls = []

    async def fake_reparse(sms_id, force_new, session):
        calls.append((sms_id, force_new, session is not None))
        return ReparseSmsResponse(
            message="Synthetic SMS reparse",
            new_status="enriched",
            txn_id=42,
            diff=["counterparty"],
        )

    monkeypatch.setattr("financial_dashboard.api.sms.reparse_sms_service", fake_reparse)

    response = await client.post("/api/sms/7/reparse?force_new=true")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Synthetic SMS reparse",
        "new_status": "enriched",
        "txn_id": 42,
        "diff": ["counterparty"],
    }
    assert calls == [(7, True, True)]


async def test_api_email_reparse_forwards_to_canonical_operation(client, monkeypatch):
    calls = []

    async def fake_reparse(email_id, force_new, session):
        calls.append((email_id, force_new, session is not None))
        return ReparseEmailResponse(
            message="Synthetic email reparse",
            new_status="parsed",
            txn_id=84,
        )

    monkeypatch.setattr(
        "financial_dashboard.api.emails.reparse_email_service", fake_reparse
    )

    response = await client.post("/api/emails/9/reparse")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Synthetic email reparse",
        "new_status": "parsed",
        "txn_id": 84,
    }
    assert calls == [(9, False, True)]


async def test_api_reparse_openapi_routes_are_typed(client):
    document = (await client.get("/openapi.json")).json()
    expected = {
        "/api/sms/{sms_id}/reparse": "ReparseSmsResponse",
        "/api/emails/{email_id}/reparse": "ReparseEmailResponse",
    }
    for path, model in expected.items():
        schema = document["paths"][path]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{model}"}
