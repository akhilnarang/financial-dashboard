"""Gmail IMAP fetch, driven through the real imaplib against a scripted server."""

import imaplib
from email.header import Header

import pytest

from financial_dashboard.db import FetchRule
from financial_dashboard.integrations.email import imap_gmail

SUBJECT_UTF8 = "Payment received ✅"
SENDERS = {1: "alerts@tick.example", 2: "alerts@plain.example"}


def _message(uid: int) -> bytes:
    """Return the raw message the fake server stores under one UID."""
    subject = Header(SUBJECT_UTF8, "utf-8").encode() if uid == 1 else "Account alert"
    return f"From: {SENDERS[uid]}\r\nSubject: {subject}\r\n\r\nbody\r\n".encode()


class _FakeGmail(imaplib.IMAP4):
    """An IMAP4 client whose transport is a scripted Gmail server.

    imaplib still encodes every command, so an argument that the real
    client cannot send fails here too.
    """

    def __init__(self, host: str, refusal: str) -> None:
        self.refusal = refusal
        self.searches: list[tuple[bytes, bytes]] = []
        self._replies = bytearray()
        self._awaiting_literal: bytes | None = None
        super().__init__(host)

    def open(
        self,
        host: str = "",
        port: int = imaplib.IMAP4_PORT,
        timeout: float | None = None,
    ) -> None:
        self.host, self.port = host, port
        self._replies += b"* OK Gmail ready\r\n"

    def read(self, size: int) -> bytes:
        data = bytes(self._replies[:size])
        del self._replies[:size]
        return data

    def readline(self) -> bytes:
        return self.read(self._replies.index(b"\n") + 1)

    def shutdown(self) -> None:
        pass

    def send(self, data: bytes) -> None:
        if data == b"\r\n":
            return
        if (line := self._awaiting_literal) is not None:
            self._awaiting_literal = None
            self._answer(line, data)
        elif data.endswith(b"}\r\n"):
            self._awaiting_literal = data
            self._replies += b"+ go ahead\r\n"
        else:
            self._answer(data, b"")

    def _answer(self, line: bytes, literal: bytes) -> None:
        """Queue the server reply to one command."""
        tag, *words = line.split()
        if literal and words[:2] != [b"UID", b"SEARCH"]:
            self._replies += tag + b" BAD Unexpected literal\r\n"
            return
        if words[:2] == [b"UID", b"SEARCH"]:
            self.searches.append((line, literal))
            if b'FROM "alerts@refused.example"' in line:
                self._refuse(tag)
                return
            uid = 2 if b'FROM "alerts@plain.example"' in line else 0
            if b"CHARSET UTF-8" in line and literal == SUBJECT_UTF8.encode():
                uid = 1
            self._replies += b"* SEARCH %d\r\n" % uid if uid else b"* SEARCH\r\n"
        elif line.endswith(b" (X-GM-MSGID RFC822)\r\n"):
            raw = _message(int(words[2]))
            self._replies += b"* %s FETCH (UID %s X-GM-MSGID 9%s RFC822 {%d}\r\n" % (
                words[2],
                words[2],
                words[2],
                len(raw),
            )
            self._replies += raw + b")\r\n"
        elif words[:1] == [b"CAPABILITY"]:
            self._replies += b"* CAPABILITY IMAP4rev1\r\n"
        self._replies += tag + b" OK done\r\n"

    def _refuse(self, tag: bytes) -> None:
        """Reject a SEARCH with BAD, or drop the connection."""
        if self.refusal == "drop":
            raise OSError("connection reset")
        self._replies += tag + b" BAD Could not parse command\r\n"


@pytest.mark.parametrize(
    ("refusal", "fetch_ok", "fetched", "backfilled", "literals"),
    [
        (
            "BAD",
            True,
            {1: ["91"], 2: [], 3: [], 4: ["92"]},
            {1, 4},
            [SUBJECT_UTF8.encode(), b"", b""],
        ),
        (
            "drop",
            False,
            {1: [], 2: [], 3: [], 4: []},
            set(),
            [SUBJECT_UTF8.encode(), b""],
        ),
    ],
)
def test_non_ascii_rule_searches_as_utf8_and_a_failed_search_skips_one_rule(
    monkeypatch: pytest.MonkeyPatch,
    refusal: str,
    fetch_ok: bool,
    fetched: dict[int, list[str]],
    backfilled: set[int],
    literals: list[bytes],
) -> None:
    """A "✅" subject searches with a UTF-8 literal and fetches its email.

    A SEARCH that imaplib or the server rejects skips only its rule, and the
    rejected rule does not finish its backfill. A dropped connection still
    fails the whole source.
    """
    servers: list[_FakeGmail] = []

    def connect(host: str) -> _FakeGmail:
        servers.append(server := _FakeGmail(host, refusal))
        return server

    monkeypatch.setattr(imap_gmail.imaplib, "IMAP4_SSL", connect)
    rules = [
        FetchRule(id=1, bank="tick", sender=SENDERS[1], subject=SUBJECT_UTF8),
        FetchRule(
            id=2, bank="crlf", sender="alerts@crlf.example\r\n", subject="Zahlung ✅"
        ),
        FetchRule(
            id=3, bank="refused", sender="alerts@refused.example", folder="INBOX"
        ),
        FetchRule(id=4, bank="plain", sender=SENDERS[2], folder="INBOX"),
    ]

    result = imap_gmail._fetch_gmail_source_sync(
        rules,
        user="u@example.com",
        password="pw",
        fetch_limit=10,
        source_id=1,
        existing_remote_ids=set(),
    )

    assert result.fetch_ok is fetch_ok
    assert {
        rid: [e.remote_id for e in got] for rid, got in result.results_by_rule.items()
    } == fetched
    assert result.backfill_ready_rule_ids == backfilled
    assert [literal for _, literal in servers[0].searches] == literals
