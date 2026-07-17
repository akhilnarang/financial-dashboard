"""Atomic publisher: tempfile-in-same-dir replace, skip-on-unchanged, path
safety, and failure isolation (an interrupted write must not corrupt the
existing file).
"""

import os
import stat

import pytest

from financial_dashboard.services.paisa.publisher import (
    GENERATED_HEADER_VERSION,
    HEADER_LINES,
    PublishError,
    PublishResult,
    publish_journal,
)

pytestmark = pytest.mark.anyio


def test_publish_writes_header_with_version_and_hash(tmp_path):
    target = tmp_path / "out.journal"
    body = "2026-01-01 * Test\n    Assets:Bank      10.00 INR\n\n"
    result = publish_journal(str(target), body)
    assert isinstance(result, PublishResult)
    assert result.published is True
    assert result.version == GENERATED_HEADER_VERSION
    on_disk = target.read_text()
    assert on_disk.startswith("\n".join(HEADER_LINES) + "\n")
    assert f"; hash: {result.body_hash}" in on_disk
    assert on_disk.endswith(body)
    # bytes_written counts bytes (UTF-8), not characters — the header em-dash
    # is multibyte, so compare against the encoded length.
    assert result.bytes_written == len(on_disk.encode("utf-8"))


def test_publish_skips_rewrite_when_bytes_unchanged(tmp_path):
    target = tmp_path / "out.journal"
    body = "2026-01-01 * Test\n    Assets:Bank      10.00 INR\n\n"
    first = publish_journal(str(target), body)
    mtime_after_first = target.stat().st_mtime_ns
    second = publish_journal(str(target), body)
    assert second.published is False
    assert second.body_hash == first.body_hash
    # The file was not rewritten: mtime unchanged.
    assert target.stat().st_mtime_ns == mtime_after_first


def test_publish_overwrites_when_body_changes(tmp_path):
    target = tmp_path / "out.journal"
    publish_journal(str(target), "old body\n")
    result = publish_journal(str(target), "new body\n")
    assert result.published is True
    assert "new body" in target.read_text()
    assert "old body" not in target.read_text()


def test_publish_creates_file_atomically_no_partial(tmp_path):
    target = tmp_path / "deep" / "out.journal"
    body = "body\n"
    with pytest.raises(PublishError) as exc:
        publish_journal(str(target), body)
    assert "parent directory does not exist" in str(exc.value)
    # No file and no leftover tempfile.
    assert not target.exists()
    assert not any(p.suffix == ".tmp" for p in tmp_path.rglob("*") if p.is_file())


def test_publish_rejects_relative_path(tmp_path):
    target = tmp_path / "out.journal"
    with pytest.raises(PublishError):
        publish_journal("out.journal", "body\n")
    assert not target.exists()


def test_publish_rejects_path_traversal(tmp_path):
    with pytest.raises(PublishError):
        publish_journal(str(tmp_path / ".." / "evil.journal"), "body\n")


def test_publish_rejects_empty_path(tmp_path):
    with pytest.raises(PublishError):
        publish_journal("", "body\n")
    with pytest.raises(PublishError):
        publish_journal("   ", "body\n")


def test_publish_failure_leaves_existing_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "out.journal"
    publish_journal(str(target), "original body\n")
    original_bytes = target.read_bytes()

    # Sabotage os.replace so the atomic step fails after the tempfile is written.
    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("financial_dashboard.services.paisa.publisher.os.replace", boom)
    with pytest.raises(OSError):
        publish_journal(str(target), "new body\n")
    # The original file is untouched...
    assert target.read_bytes() == original_bytes
    # ...and no tempfile was orphaned in the directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.journal")]
    assert leftovers == []
    monkeypatch.setattr(
        "financial_dashboard.services.paisa.publisher.os.replace", real_replace
    )


def test_publish_rewrites_same_file_repeatedly(tmp_path):
    target = tmp_path / "out.journal"
    bodies = [f"body {i}\n" for i in range(5)]
    for body in bodies:
        result = publish_journal(str(target), body)
        assert result.published is True
        assert body in target.read_text()


def test_header_hash_matches_body_sha256(tmp_path):
    import hashlib

    target = tmp_path / "out.journal"
    body = "content line\n"
    result = publish_journal(str(target), body)
    assert result.body_hash == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_published_file_permissions_are_normal(tmp_path):
    target = tmp_path / "out.journal"
    publish_journal(str(target), "x\n")
    mode = stat.S_IMODE(target.stat().st_mode)
    # tempfile.mksttemp creates 0o600; os.replace preserves it. We only assert
    # the file is readable/writable by the owner — not a specific mode, since
    # umask and platform differ.
    assert mode & stat.S_IRUSR
    assert mode & stat.S_IWUSR
