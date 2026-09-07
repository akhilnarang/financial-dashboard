from pathlib import Path

import httpx
import pytest

from financial_dashboard.config import settings
from financial_dashboard.db import AuditInteraction, Transaction
from financial_dashboard.services.transaction_attachments import (
    AttachmentError,
    MAX_ATTACHMENT_BYTES,
    StoredAttachment,
    attach_downloaded_attachment,
    detect_attachment_type,
    download_attachment,
    resolve_attachment_path,
)


def test_detect_attachment_type_uses_file_signature():
    assert detect_attachment_type(b"%PDF-1.7\n") == ("application/pdf", ".pdf")
    assert detect_attachment_type(b"\x89PNG\r\n\x1a\nrest") == (
        "image/png",
        ".png",
    )
    with pytest.raises(AttachmentError):
        detect_attachment_type(b"not really a pdf")


def test_resolve_attachment_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))
    with pytest.raises(AttachmentError):
        resolve_attachment_path("../outside.pdf")


@pytest.mark.anyio
async def test_download_attachment_streams_and_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7\nreceipt")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        stored = await download_attachment(
            "https://telegram.invalid/file",
            transaction_id=42,
            declared_size=16,
            client=client,
        )
    finally:
        await client.aclose()

    assert stored.media_type == "application/pdf"
    assert stored.relative_path.startswith("txn-42-")
    assert stored.absolute_path.read_bytes() == b"%PDF-1.7\nreceipt"
    assert not list(Path(tmp_path).glob("*.part"))


@pytest.mark.anyio
async def test_download_attachment_rejects_declared_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))
    with pytest.raises(AttachmentError, match="20 MB"):
        await download_attachment(
            "https://telegram.invalid/file",
            transaction_id=1,
            declared_size=MAX_ATTACHMENT_BYTES + 1,
        )


@pytest.mark.anyio
async def test_download_attachment_enforces_streamed_size_without_length_header(
    tmp_path, monkeypatch
):
    import financial_dashboard.services.transaction_attachments as attachments

    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))
    monkeypatch.setattr(attachments, "MAX_ATTACHMENT_BYTES", 16)

    class OversizeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"%PDF-1.7\n"
            yield b"x" * 20

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=OversizeStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AttachmentError, match="20 MB"):
            await download_attachment(
                "https://telegram.invalid/file",
                transaction_id=42,
                declared_size=None,
                client=client,
            )

    assert not list(Path(tmp_path).iterdir())


@pytest.mark.anyio
async def test_attachment_route_serves_only_saved_path(
    session, client, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))
    (tmp_path / "receipt.pdf").write_bytes(b"%PDF-1.7\nreceipt")
    transaction = Transaction(
        bank="hdfc",
        email_type="purchase",
        direction="debit",
        amount="12.00",
        attachment_path="receipt.pdf",
    )
    session.add(transaction)
    await session.commit()

    response = await client.get(f"/api/transactions/{transaction.id}/attachment")

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("attachment;")


@pytest.mark.anyio
async def test_attachment_fence_and_caption_are_exact(session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transaction_attachment_root", str(tmp_path))
    transaction = Transaction(
        bank="hdfc", email_type="purchase", direction="debit", amount="12.00"
    )
    interaction = AuditInteraction(
        inbound_chat_id=7,
        trigger="attachment",
        status="processing",
        worker_token="current-worker",
    )
    session.add_all([transaction, interaction])
    await session.flush()
    receipt = tmp_path / "receipt.pdf"
    receipt.write_bytes(b"%PDF-1.7\nreceipt")
    stored = StoredAttachment("receipt.pdf", receipt, "application/pdf")

    with pytest.raises(AttachmentError, match="lease"):
        await attach_downloaded_attachment(
            session,
            transaction.id,
            stored,
            caption="  exact caption  ",
            interaction_id=interaction.id,
            worker_token="stale-worker",
        )

    mutation = await attach_downloaded_attachment(
        session,
        transaction.id,
        stored,
        caption="  exact caption  ",
        interaction_id=interaction.id,
        worker_token="current-worker",
    )
    assert mutation.new_path == "receipt.pdf"
    assert transaction.note == "  exact caption  "
    assert transaction.attachment_path == "receipt.pdf"
