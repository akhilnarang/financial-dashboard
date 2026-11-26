"""Fastmail JMAP fetch, driven through the real client against a scripted server."""

import io
import json

import pytest
from urllib.parse import unquote, urlsplit
from urllib.request import Request

from financial_dashboard.db import FetchRule
from financial_dashboard.integrations.email import jmap_fastmail

ACCOUNT = "acct1"
API_URL = "https://api.example/jmap/api/"
DOWNLOAD_URL = (
    "https://dl.example/jmap/download/{accountId}/{blobId}/{name}?type={type}"
)
REFUSED_SENDER = "alerts@refused.example"
FAILING_SENDER = "alerts@failing.example"
SEEN_IDS = {f"e-seen-{n}" for n in range(5)}
EMAILS = {
    "e-new": "alerts@new.example",
    "e-old": "alerts@new.example",
} | dict.fromkeys(sorted(SEEN_IDS), "alerts@seen.example")


class _FakeFastmail:
    """A JMAP server that answers the session, method and download requests.

    It resolves result references, so a request the real server could not
    answer fails here too.
    """

    def __call__(self, req: Request, timeout: float = 0) -> io.BytesIO:
        url = req.full_url
        if url == jmap_fastmail.JMAP_SESSION_URL:
            return self._json(
                {
                    "apiUrl": API_URL,
                    "downloadUrl": DOWNLOAD_URL,
                    "primaryAccounts": {jmap_fastmail.JMAP_MAIL: ACCOUNT},
                    "capabilities": {jmap_fastmail.JMAP_CORE: {"maxObjectsInGet": 4}},
                }
            )
        if url == API_URL:
            assert isinstance(req.data, bytes)
            calls = json.loads(req.data)["methodCalls"]
            return self._json({"methodResponses": self._answer(calls)})
        parts = [unquote(p) for p in urlsplit(url).path.split("/")]
        assert parts[3] == ACCOUNT
        return io.BytesIO(f"raw {parts[4]}".encode())

    def _json(self, payload: dict) -> io.BytesIO:
        return io.BytesIO(json.dumps(payload).encode())

    def _answer(self, calls: list[list]) -> list[list]:
        answered: dict[str, tuple[str, dict]] = {}
        responses: list[list] = []
        for name, args, call_id in calls:
            if ref := args.get("#ids"):
                ref_name, ref_args = answered[ref["resultOf"]]
                if ref_name == "error":
                    answered[call_id] = ("error", {"type": "invalidResultReference"})
                    responses.append(["error", answered[call_id][1], call_id])
                    continue
                args["ids"] = ref_args["ids"]
            answered[call_id] = self._method(name, args)
            responses.append([*answered[call_id], call_id])
        return responses

    def _method(self, name: str, args: dict) -> tuple[str, dict]:
        match name:
            case "Email/query" if args["filter"].get("from") == REFUSED_SENDER:
                return "error", {"type": "unsupportedFilter"}
            case "Email/query" if args["filter"].get("from") == FAILING_SENDER:
                return name, {"ids": [FAILING_SENDER]}
            case "Email/query":
                ids = [i for i, s in EMAILS.items() if s == args["filter"].get("from")]
                return name, {"ids": ids[args["position"] :][: args["limit"]]}
            case "Email/get" if FAILING_SENDER in args["ids"]:
                return "error", {"type": "serverFail", "description": None}
            case "Email/get":
                assert len(args["ids"]) <= 4
                return name, {
                    "list": [
                        {
                            "id": i,
                            "blobId": f"blob-{i}",
                            "from": [{"email": EMAILS[i]}],
                            "subject": "Alert",
                            "mailboxIds": {"inbox": True},
                        }
                        for i in reversed(args["ids"])
                    ]
                }
            case "Mailbox/query":
                return name, {"ids": []}
            case "Mailbox/get":
                return name, {"list": []}
            case _:
                raise AssertionError(f"unexpected JMAP method {name}")


def test_fetch_source_keeps_backfill_pending_for_rules_that_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a rule whose search finished completes its backfill.

    A JMAP method error or a missing folder must not read as "no mail". A
    rule whose matches are all stored already still completes its backfill,
    and its pages stay within the server's maxObjectsInGet. The fetch
    limit keeps the newest email, whatever order Email/get answers in.
    Other rules keep running.
    """
    monkeypatch.setattr(jmap_fastmail, "urlopen", _FakeFastmail())
    rules = [
        FetchRule(id=1, bank="new", sender="alerts@new.example"),
        FetchRule(id=2, bank="seen", sender="alerts@seen.example"),
        FetchRule(id=3, bank="refused", sender=REFUSED_SENDER),
        FetchRule(id=4, bank="folder", sender="alerts@new.example", folder="Missing"),
    ]

    result = jmap_fastmail._fetch_fastmail_source_sync(
        rules,
        token="tok",
        fetch_limit=1,
        source_id=7,
        existing_remote_ids={"e-old"} | SEEN_IDS,
    )

    assert result.fetch_ok is True
    assert result.results_by_rule == {
        1: [("src7:fastmail:e-new", "e-new", b"raw blob-e-new")],
        2: [],
        3: [],
        4: [],
    }
    assert result.backfill_ready_rule_ids == {1, 2}


def test_fetch_source_fails_on_a_server_level_method_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server-level JMAP error fails the whole source.

    The error comes inside an HTTP 200 answer. It must not read as an empty
    page, and the other rules must not mark the sync as good.
    """
    monkeypatch.setattr(jmap_fastmail, "urlopen", _FakeFastmail())
    rules = [
        FetchRule(id=1, bank="new", sender="alerts@new.example"),
        FetchRule(id=2, bank="failing", sender=FAILING_SENDER),
    ]

    result = jmap_fastmail._fetch_fastmail_source_sync(
        rules,
        token="tok",
        fetch_limit=0,
        source_id=7,
        existing_remote_ids=set(),
    )

    assert result.fetch_ok is False
    assert result.backfill_ready_rule_ids == set()
