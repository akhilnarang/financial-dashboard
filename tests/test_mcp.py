import json

import pytest

from financial_dashboard.config import settings
from financial_dashboard.db import Receivable, Transaction
from financial_dashboard.services import settings as settings_service

TOKEN = "test-mcp-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def mcp_token(monkeypatch):
    monkeypatch.setattr(settings, "mcp_token", type(settings.mcp_token)(TOKEN))


async def rpc(client, method, params=None, request_id=1):
    """Send one JSON-RPC request to the MCP endpoint and return the reply body."""
    response = await client.post(
        "/api/mcp",
        headers=AUTH,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    return response.json()


@pytest.mark.anyio
async def test_mcp_client_reads_rows_and_reports(
    client, session, mcp_token, monkeypatch
):
    monkeypatch.setitem(
        settings_service._cache, "categorization.hidden_identifiers", "Holder"
    )
    food, rent = (
        Transaction(
            bank="testbank",
            email_type="test",
            direction="debit",
            amount=amount,
            counterparty=counterparty,
        )
        for amount, counterparty in ((250, "FOOD APP"), (9000, "PAT HOLDER RENT"))
    )
    session.add_all([food, rent, Receivable(person="Pat Holder", amount=500)])
    await session.flush()

    started = (await rpc(client, "initialize", {"protocolVersion": "2025-06-18"}))[
        "result"
    ]
    assert started["protocolVersion"] == "2025-06-18"
    assert started["capabilities"] == {"tools": {"listChanged": False}}
    assert started["serverInfo"]["name"] == "financial-dashboard"
    # A notification gets no reply body.
    notified = await client.post(
        "/api/mcp",
        headers=AUTH,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert notified.status_code == 202

    tools = {
        tool["name"]: tool
        for tool in (await rpc(client, "tools/list"))["result"]["tools"]
    }
    assert len(tools) == 9
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools.values())
    # The tool name is the MCP tool, never one of its arguments.
    assert "name" not in tools["list_transactions"]["inputSchema"]["properties"]
    assert "search" in tools["list_transactions"]["inputSchema"]["properties"]

    read = (
        await rpc(
            client,
            "tools/call",
            {"name": "list_transactions", "arguments": {"search": "food"}},
        )
    )["result"]
    assert read["isError"] is False
    assert [row["id"] for row in read["structuredContent"]["items"]] == [food.id]
    # The row has null fields. A client checks the result against the output
    # schema, so every required field must be there, null or not.
    output_defs = tools["list_transactions"]["outputSchema"]["$defs"]
    required = set(output_defs["AssistantTransaction"]["required"])
    assert required <= read["structuredContent"]["items"][0].keys()

    owed = (
        await rpc(client, "tools/call", {"name": "list_receivables", "arguments": None})
    )["result"]
    assert [item["person"] for item in owed["structuredContent"]["items"]] == [
        "[redacted-name]"
    ]

    # A report is text for the user; through MCP it reaches a model, so a
    # configured name is hidden there too.
    report = (
        await rpc(
            client,
            "tools/call",
            {"name": "show_transactions", "arguments": {"search": "rent"}},
        )
    )["result"]
    assert "[redacted-name]" in report["content"][0]["text"]
    assert "HOLDER" not in report["content"][0]["text"].upper()
    # A link query could carry a name past the redaction, so there are none.
    assert "<" not in report["content"][0]["text"]

    # Bad arguments are a tool error that the model can fix, not a protocol error.
    bad = (
        await rpc(
            client,
            "tools/call",
            {"name": "list_transactions", "arguments": {"limit": 99}},
        )
    )["result"]
    assert bad["isError"] is True
    assert (await rpc(client, "resources/list"))["error"]["code"] == -32601
    # A request id is never null, and a body must be one JSON-RPC object.
    for body in ({"jsonrpc": "2.0", "id": None, "method": "ping"}, [], "{"):
        refused = await client.post(
            "/api/mcp",
            headers={**AUTH, "Content-Type": "application/json"},
            content=body if isinstance(body, str) else json.dumps(body),
        )
        assert refused.status_code == 400
    stream = await client.get("/api/mcp", headers=AUTH)
    assert (stream.status_code, stream.headers["allow"]) == (405, "POST")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("headers", "token", "status"),
    [
        ({}, TOKEN, 401),
        ({"Authorization": "Bearer wrong"}, TOKEN, 401),
        ({**AUTH, "Origin": "http://evil.example"}, TOKEN, 403),
        (AUTH, "", 404),
    ],
)
async def test_mcp_refuses_requests_without_its_token(
    client, monkeypatch, headers, token, status
):
    monkeypatch.setattr(settings, "mcp_token", type(settings.mcp_token)(token))

    response = await client.post(
        "/api/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )

    assert response.status_code == status
