# AGENTS.md — AI assistant instructions for financial-dashboard

## Project layout

```text
financial_dashboard/
  main.py                 FastAPI app factory + lifespan wiring
  api/                    JSON routes
  web/
    __init__.py           HTML router aggregation
    dashboard.py
    accounts.py
    sources.py
    rules.py
    transactions.py
    emails.py
    statements.py
    bank_statements.py    Included before statements.py
    settings.py
    polling.py
    forms.py
  services/
    accounts.py
    emails.py
    fetch.py
    linker.py
    reminders.py
    rules.py
    settings.py
    sources.py
    telegram.py
    transactions.py
    statements/
      __init__.py
      cc.py
      bank.py
      shared.py
      dates.py
  integrations/
    parsers.py
    email/
      base.py
      body.py
      parsing.py
      imap_gmail.py
      jmap_fastmail.py
      orchestrator.py
  core/                   Shared templating, date, auth, crypto, deps helpers
  db/                     Engine/session setup, models, enums, init_db glue
  templates/              Jinja templates
  static/                 CSS and JS assets
  data/
    failed/               Failed parse spools (.eml files, max 7 days old)
    statements/           Saved statement PDFs

scripts/
  main.py                 Raw-email dev CLI
  seed.py
  populate.py
```

## How to run

```bash
uv run fastapi dev
uv run python scripts/seed.py
uv run pytest -q
```

## Service conventions

### Session handling

- Route handlers get `session: AsyncSession = Depends(get_session)` from `core/deps.py`.
- Services take `session: AsyncSession` as a required first parameter when the caller already owns the request session.
- Background tasks (fetch polling, Telegram handlers, reminders) open their own session with `async with async_session() as session:` and pass it onward where applicable.
- Do not add `async_session_factory` fallback parameters.

## Compatibility rules

- Preserve current HTTP routes, JSON response shapes, template behavior, parser-derived `email_type` values, and script entrypoints unless a task explicitly allows a breaking change.
- Shared poll state belongs on `app.state.fetch_service`; avoid new module-level poll loops or duplicate status dicts.
- Keep `bank_statements.router` registered before `statements.router`.

## Database schema docs

- The Mermaid ER diagram in `README.md` is documentation for `financial_dashboard/db/models.py`; when models change, update the diagram, the model summary table, and the key-constraints notes together.
- Keep every ORM table in the diagram, including standalone tables such as `settings`, and include every real foreign-key edge from `cards`, `fetch_rules`, `emails`, `statement_uploads`, `bank_statement_uploads`, and `transactions`.
- If a field is only a parser/linker hint (`card_mask`, `account_mask`, similar denormalized values), describe it in nearby prose instead of drawing a fake foreign-key relationship.

## Local cross-repo development

**The committed state always uses git/PyPI-tagged sibling versions.**
`pyproject.toml`'s `[tool.uv.sources]` pins each sibling to its
`git = "https://github.com/..."` URL, and `uv.lock` records the exact SHA
that CI, deploys, and reviewers see.

When you're actively developing a change that spans this repo and a
sibling (`bank-email-parser`, `bank-statement-parser`, `cc-parser`), you can
**temporarily** swap those git sources for local path sources so edits in
the sibling are picked up immediately — no reinstall, no version bump, no
lockfile churn. The siblings live at `../<name>` relative to this repo.

Temporary dev-only block (do NOT commit this):

```toml
[tool.uv.sources]
bank-email-parser = { path = "../bank-email-parser", editable = true }
bank-statement-parser = { path = "../bank-statement-parser", editable = true }
cc-parser = { path = "../cc-parser", editable = true }
```

Rules:

- **Before pushing or opening a PR**: revert `[tool.uv.sources]` to the
  `git = "..."` form and run `uv lock` so the lockfile pins the new SHA
  (tag the sibling repo first if the change isn't already on `main`).
  Deploys run off these git sources; path sources would break CI.
- **Never commit path sources.** If `git diff pyproject.toml uv.lock` shows
  path entries or a stale lockfile, fix that before pushing.
- **During dev with path sources**, direct attribute access works and `ty`
  sees the new sibling shape — no `getattr` scaffolding needed.
- Sibling repo SHAs pinned by the committed lockfile can be inspected in
  `uv.lock` under the relevant
  `[[package]]` block.

## Quality gates

Run all of these before finishing a refactor:

- `uv run ruff check financial_dashboard tests scripts`
- `uv run ruff format --check financial_dashboard tests scripts`
- `uv run ty check financial_dashboard`
- `uv run pytest -q`

Use `uv run` for every command. PEP 758 parenthesis-free `except X, Y:` syntax is valid in this repo. Prefer `python-dateutil` helpers from `core/dates.py`.
