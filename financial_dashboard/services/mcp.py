"""A minimal, read-only MCP server over stateless Streamable HTTP.

It speaks the subset of the Model Context Protocol that a tool server needs:
``initialize``, ``ping``, ``tools/list`` and ``tools/call``. Each POST holds one
JSON-RPC message and gets one JSON reply. There is no session and no event
stream. The tools are the assistant's reads and reports, built from the same
contracts, so an MCP client and the Telegram assistant see the same data.
"""

import html
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from pydantic.alias_generators import to_camel
from pydantic.types import StrictInt, StrictStr
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.schemas.receivables import ReceivableRead
from financial_dashboard.services.assistant.context import ToolResult
from financial_dashboard.services.assistant.contracts import (
    Answer,
    GetCardDues,
    GetCashflowSummary,
    GetHoldings,
    GetInvestmentFlows,
    GetNetworth,
    ListAccounts,
    ListReceivables,
    ListTransactions,
    ShowTransactions,
    StrictModel,
)
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.assistant.packs import tool_name
from financial_dashboard.services.assistant.reads import (
    dispatch_read,
    read_transactions,
)
from financial_dashboard.services.assistant.reports import render_report
from financial_dashboard.services.categorization.normalize import (
    redact_names,
    redact_pii,
)
from financial_dashboard.services.categorization.vocabulary import get_active_slugs
from financial_dashboard.services.settings import get_redact_name_tokens

# The newest version first. A client that asks for another gets the newest.
# 2025-03-26 is left out: it allows batches, which this server does not take.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")
SERVER_NAME = "financial-dashboard"
SERVER_VERSION = "0.1.0"
INSTRUCTIONS = """\
Read-only access to one person's bank, card, investment and cashflow records.
Amounts are in INR unless a row names another currency. Dates are YYYY-MM-DD.
A financial year (FY) runs from 1 April to 31 March.
For totals, use a report tool: get_cashflow_summary, get_investment_flows or
show_transactions. Their totals follow the dashboard's rules for excluded rows
and account scope, so never add up rows from list_transactions yourself.
list_transactions returns at most 20 rows. Narrow it with filters, such as
search for a merchant, a category slug, an account_id from list_accounts, or a
date range. Names that the owner hides read as [redacted-name]."""
EMPTY_LIST_HINT = (
    "No row matched every filter. list_accounts gives the account and card ids "
    "for account_id and card_id, and the server instructions list the category "
    "slugs."
)
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
# An Indian PAN, which a CAS snapshot label can hold.
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")
READ_TOOLS = (ListTransactions, ListAccounts, ListReceivables)
REPORT_TOOLS = (
    ShowTransactions,
    GetCashflowSummary,
    GetInvestmentFlows,
    GetHoldings,
    GetNetworth,
    GetCardDues,
)
McpCall: TypeAlias = Annotated[
    ListTransactions
    | ListAccounts
    | ListReceivables
    | ShowTransactions
    | GetCashflowSummary
    | GetInvestmentFlows
    | GetHoldings
    | GetNetworth
    | GetCardDues,
    Field(discriminator="name"),
]
CALL_ADAPTER: TypeAdapter[McpCall] = TypeAdapter(McpCall)
TOOL_MODELS: dict[str, type[StrictModel]] = {
    tool_name(model): model for model in (*READ_TOOLS, *REPORT_TOOLS)
}
TOOL_DESCRIPTIONS = {
    "list_transactions": (
        "Read up to 20 transaction rows that match the filters, newest first, "
        "to find or explain them. Give transaction_ids to read rows by id."
    ),
    "list_accounts": "List the accounts and their cards, with ids and names.",
    "list_receivables": (
        "List money that a person owes the owner (receivable) or that the owner "
        "owes a person (payable), with links to transactions."
    ),
    "show_transactions": (
        "Report the rows that match one set of filters, with their count, "
        "debit and credit totals."
    ),
    "get_cashflow_summary": (
        "Report income, spending, investment and cashflow totals of the bank "
        "accounts over an inclusive date range."
    ),
    "get_investment_flows": (
        "Report money put into and taken out of investments over a date range, "
        "by counterparty."
    ),
    "get_holdings": "Report current funds and stocks with their value and units.",
    "get_networth": (
        "Report net worth, assets, liabilities and the latest balance of every account."
    ),
    "get_card_dues": (
        "Report the latest bill, due date and payment status of each credit card."
    ),
}


class McpModel(BaseModel):
    """A protocol object, written with the camelCase names of the MCP schema."""

    model_config = ConfigDict(
        alias_generator=to_camel, validate_by_name=True, serialize_by_alias=True
    )


class ToolAnnotations(McpModel):
    """The behaviour hints of a tool. Every tool here only reads.

    Attributes:
        read_only_hint: The tool changes nothing.
        destructive_hint: The tool can delete or overwrite data.
        idempotent_hint: A repeated call has no further effect.
        open_world_hint: The tool reaches outside systems.
    """

    read_only_hint: bool = True
    destructive_hint: bool = False
    idempotent_hint: bool = True
    open_world_hint: bool = False


class McpTool(McpModel):
    """One entry of ``tools/list``.

    Attributes:
        name: The tool name.
        description: What the tool does.
        input_schema: The JSON schema of the arguments.
        output_schema: The JSON schema of ``structuredContent``, or None for a
            tool that returns text only.
        annotations: The behaviour hints.
    """

    name: str
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    annotations: ToolAnnotations = ToolAnnotations()


class ToolList(McpModel):
    """The result of ``tools/list``. There is one page, so no cursor."""

    tools: list[McpTool]


class TextContent(McpModel):
    """A text block of a tool result."""

    type: Literal["text"] = "text"
    text: str


class CallToolResult(McpModel):
    """The result of ``tools/call``.

    Attributes:
        content: The result as text, for clients that read only text.
        structured_content: The read result, for a tool with an output schema.
        is_error: The tool could not run with these arguments.
    """

    content: list[TextContent]
    structured_content: ToolResult | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    is_error: bool = False


class ServerInfo(McpModel):
    """The name and version of this server."""

    name: str = SERVER_NAME
    version: str = SERVER_VERSION


class ToolsCapability(McpModel):
    """The tools capability. The tool list never changes while running."""

    list_changed: bool = False


class ServerCapabilities(McpModel):
    """The capabilities of this server: tools only."""

    tools: ToolsCapability = ToolsCapability()


class InitializeResult(McpModel):
    """The result of ``initialize``.

    Attributes:
        protocol_version: The agreed protocol version.
        capabilities: What this server offers.
        server_info: The server name and version.
        instructions: How to use the tools, with the category slugs.
    """

    protocol_version: str
    capabilities: ServerCapabilities = ServerCapabilities()
    server_info: ServerInfo = ServerInfo()
    instructions: str


class EmptyResult(McpModel):
    """The result of ``ping``."""


class JsonRpcRequest(BaseModel):
    """One JSON-RPC request or notification from the client.

    Attributes:
        jsonrpc: The JSON-RPC version.
        id: The request id, or None for a notification.
        method: The method name.
        params: The method parameters.
    """

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"]
    id: StrictInt | StrictStr | None = None
    method: str
    params: dict[str, JsonValue] | None = None


class CallToolParams(BaseModel):
    """The parameters of ``tools/call``.

    Attributes:
        name: The tool to call.
        arguments: The tool arguments.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: dict[str, JsonValue] | None = None


class JsonRpcError(BaseModel):
    """A JSON-RPC error object."""

    code: int
    message: str


class JsonRpcResponse(BaseModel):
    """One JSON-RPC reply: a result or an error, never both.

    Attributes:
        jsonrpc: The JSON-RPC version.
        id: The id of the request, or None when the request had no valid id.
        result: The method result.
        error: The error.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    result: InitializeResult | ToolList | CallToolResult | EmptyResult | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    error: JsonRpcError | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


async def handle_message(
    session: AsyncSession, message: JsonValue
) -> JsonRpcResponse | None:
    """Answer one JSON-RPC message.

    Args:
        session: The request session. The tools only read.
        message: The decoded request body.

    Returns:
        The reply, or None for a notification or a client response, which
        get no reply.
    """
    if not isinstance(message, dict):
        return error_response(None, INVALID_REQUEST, "send one JSON-RPC object")
    request_id = message.get("id")
    valid_id = isinstance(request_id, str) or (
        isinstance(request_id, int) and not isinstance(request_id, bool)
    )
    reply_id = request_id if valid_id else None
    if "method" not in message:
        # A client response. This server sends no requests, so it drops it.
        is_response = message.get("jsonrpc") == "2.0" and (
            ("result" in message) != ("error" in message)
        )
        if is_response and valid_id:
            return None
        return error_response(reply_id, INVALID_REQUEST, "invalid JSON-RPC message")
    if "id" in message and not valid_id:
        return error_response(
            None, INVALID_REQUEST, "the id must be a string or number"
        )
    try:
        request = JsonRpcRequest.model_validate(message)
    except ValidationError:
        return error_response(reply_id, INVALID_REQUEST, "invalid JSON-RPC request")
    if request.id is None:
        return None
    params = request.params or {}
    match request.method:
        case "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            slugs = ", ".join(await get_active_slugs(session))
            result = InitializeResult(
                protocol_version=str(version),
                instructions=f"{INSTRUCTIONS}\nCategory slugs: {slugs}.",
            )
        case "ping":
            result = EmptyResult()
        case "tools/list":
            result = ToolList(tools=[_tool(name) for name in TOOL_MODELS])
        case "tools/call":
            try:
                call_params = CallToolParams.model_validate(params)
            except ValidationError:
                return error_response(request.id, INVALID_PARAMS, "invalid params")
            if call_params.name not in TOOL_MODELS:
                return error_response(
                    request.id, INVALID_PARAMS, f"unknown tool: {call_params.name}"
                )
            result = await call_tool(
                session, call_params.name, call_params.arguments or {}
            )
        case _:
            return error_response(
                request.id, METHOD_NOT_FOUND, f"method not found: {request.method}"
            )
    return JsonRpcResponse(id=request.id, result=result)


def error_response(
    request_id: int | str | None, code: int, message: str
) -> JsonRpcResponse:
    """Build a JSON-RPC error reply.

    Args:
        request_id: The id of the request, or None when it had no valid id.
        code: The JSON-RPC error code.
        message: What went wrong.

    Returns:
        The error reply.
    """
    return JsonRpcResponse(
        id=request_id, error=JsonRpcError(code=code, message=message)
    )


async def call_tool(
    session: AsyncSession, name: str, arguments: dict[str, JsonValue]
) -> CallToolResult:
    """Run one tool. Bad arguments are a tool error that the model can fix.

    Args:
        session: The request session.
        name: A tool name from ``TOOL_MODELS``.
        arguments: The tool arguments, without the tool name.

    Returns:
        The tool result: the redacted rows of a read, or the text of a report.
    """
    if "name" in arguments:
        return _tool_error("invalid arguments: name is the tool, not an argument")
    try:
        call = CALL_ADAPTER.validate_python({**arguments, "name": name})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        return _tool_error(f"invalid arguments: {problems}")
    tokens = get_redact_name_tokens()
    match call:
        case ListTransactions():
            try:
                rows = await read_transactions(session, call)
            except MutationRejected as exc:
                return _tool_error(str(exc))
            read = ToolResult(
                tool=name, items=rows, hint=None if rows else EMPTY_LIST_HINT
            )
        case ListAccounts():
            read = ToolResult(tool=name, items=await dispatch_read(session, call))
        case ListReceivables():
            receivables = await dispatch_read(session, call)
            read = ToolResult(
                tool=name,
                items=[
                    item.model_copy(
                        update={
                            "person": _redact(item.person, tokens),
                            "note": item.note and _redact(item.note, tokens),
                        }
                    )
                    for item in receivables
                    if isinstance(item, ReceivableRead)
                ],
            )
        case _:
            # No links: a link query can hold a name that redaction cannot see.
            match await render_report(session, call, base_url=""):
                case Answer(text=text):
                    plain = "\n".join(
                        _redact(html.unescape(HTML_TAG.sub("", line)), tokens)
                        for line in text.splitlines()
                    )
                    return CallToolResult(content=[TextContent(text=plain)])
                case error:
                    return _tool_error(error.message)
    return CallToolResult(
        content=[TextContent(text=read.model_dump_json())],
        structured_content=read,
    )


def _redact(text: str, tokens: tuple[str, ...]) -> str:
    """Hide configured names, long numbers, phone numbers and PANs in one line.

    Redact one line at a time: the name rule joins nearby markers, and across
    lines it would merge two report rows into one.
    """
    return redact_names(redact_pii(PAN.sub("[redacted-pan]", text)), tokens)


def _tool(name: str) -> McpTool:
    """Describe one tool for ``tools/list``.

    The input schema is the contract model's schema without the tool name,
    which MCP sends apart from the arguments.
    """
    schema = TOOL_MODELS[name].model_json_schema()
    properties = schema.get("properties", {})
    properties.pop("name", None)
    required = [field for field in schema.get("required", []) if field != "name"]
    input_schema = {
        **{
            key: value
            for key, value in schema.items()
            if key not in ("title", "description", "required")
        },
        "type": "object",
        "properties": properties,
        **({"required": required} if required else {}),
    }
    return McpTool(
        name=name,
        description=TOOL_DESCRIPTIONS[name],
        input_schema=input_schema,
        output_schema=(
            ToolResult.model_json_schema(mode="serialization")
            if TOOL_MODELS[name] in READ_TOOLS
            else None
        ),
    )


def _tool_error(message: str) -> CallToolResult:
    """Return a tool error that the model reads and can act on."""
    return CallToolResult(content=[TextContent(text=message)], is_error=True)
