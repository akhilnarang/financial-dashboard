import pytest

from financial_dashboard.db import AuditInteraction
from financial_dashboard.services.assistant.evals import iter_audit_eval_rows


@pytest.mark.anyio
async def test_eval_export_has_stable_ids_and_no_attachment_bytes(session):
    session.add(
        AuditInteraction(
            telegram_update_id="eval-1",
            inbound_chat_id=2,
            trigger="attachment",
            user_text="receipt",
            inbound_payload_json='{"file_id":"telegram-id","bytes":"omitted"}',
            model_input_json='{"transaction":{"id":9}}',
            status="delivered",
            outcome="attachment",
            output_mode="json_schema",
        )
    )
    await session.commit()

    rows = [row async for row in iter_audit_eval_rows(session)]

    assert rows[0]["interaction_id"] == 1
    assert rows[0]["model_input"] == {"transaction": {"id": 9}}
    assert rows[0]["output_mode"] == "json_schema"
    assert "inbound_payload" not in rows[0]
