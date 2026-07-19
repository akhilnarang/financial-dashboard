import logging
import pytest
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, Base, Card
from financial_dashboard.services import database as database_service

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _clear_database(session):
    for table in reversed(list(Base.metadata.tables.values())):
        await session.execute(table.delete())
    await session.commit()


async def _insert_dangling_cards(session, *row_ids: int) -> None:
    await session.execute(
        Card.__table__.insert(),
        [
            {
                "id": row_id,
                "account_id": 9999,
                "card_mask": f"synthetic-{row_id}",
            }
            for row_id in row_ids
        ],
    )
    await session.commit()


async def test_foreign_key_check_reports_real_sqlite_violation(client, session):
    foreign_keys_enabled = await session.scalar(text("PRAGMA foreign_keys"))
    assert foreign_keys_enabled == 0
    await _insert_dangling_cards(session, 7)

    response = await client.get("/api/system/foreign-key-check")

    assert response.status_code == 200
    assert response.json() == {
        "status": "violations",
        "backend": "sqlite",
        "returned_count": 1,
        "limit": 100,
        "truncated": False,
        "violations": [
            {
                "child_table": "cards",
                "child_row_id": 7,
                "parent_table": "accounts",
                "fk_constraint_index": 0,
            }
        ],
    }


async def test_foreign_key_check_supports_without_rowid_violations(client, session):
    await session.execute(
        text("CREATE TABLE wr_parent (id TEXT PRIMARY KEY) WITHOUT ROWID")
    )
    await session.execute(
        text(
            "CREATE TABLE wr_child ("
            "id TEXT PRIMARY KEY, parent_id TEXT, "
            "FOREIGN KEY(parent_id) REFERENCES wr_parent(id)) WITHOUT ROWID"
        )
    )
    await session.execute(
        text("INSERT INTO wr_child (id, parent_id) VALUES ('synthetic', 'missing')")
    )
    await session.commit()

    response = await client.get("/api/system/foreign-key-check")

    assert response.status_code == 200
    assert response.json()["violations"] == [
        {
            "child_table": "wr_child",
            "child_row_id": None,
            "parent_table": "wr_parent",
            "fk_constraint_index": 0,
        }
    ]


async def test_foreign_key_check_has_stable_ordering_and_truncation(client, session):
    await _insert_dangling_cards(session, 30, 10, 20)

    first = await client.get("/api/system/foreign-key-check?limit=2")
    second = await client.get("/api/system/foreign-key-check?limit=2")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["status"] == "violations"
    assert body["returned_count"] == 2
    assert body["limit"] == 2
    assert body["truncated"] is True
    assert [item["child_row_id"] for item in body["violations"]] == [10, 20]


@pytest.mark.parametrize("limit", ["0", "501", "not-an-integer"])
async def test_foreign_key_check_validates_limit(client, limit):
    response = await client.get(
        "/api/system/foreign-key-check", params={"limit": limit}
    )

    assert response.status_code == 422


async def test_foreign_key_check_does_not_autoflush_and_executes_one_bounded_query(
    client, session
):
    pending_account = Account(
        bank="synthetic",
        label="Must remain pending",
        type="bank_account",
    )
    session.add(pending_account)

    executions: list[tuple[str, object, bool | None]] = []
    bind = session.get_bind()

    def record_statement(_conn, _cursor, statement, parameters, context, _many):
        executions.append(
            (statement, parameters, context.execution_options.get("autoflush"))
        )

    event.listen(bind, "before_cursor_execute", record_statement)
    try:
        response = await client.get("/api/system/foreign-key-check?limit=17")
    finally:
        event.remove(bind, "before_cursor_execute", record_statement)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert len(executions) == 1
    statement, parameters, autoflush = executions[0]
    assert " ".join(statement.split()).lower() == (
        'select "table" as child_table, rowid as child_row_id, '
        "parent as parent_table, fkid as fk_constraint_index "
        'from pragma_foreign_key_check order by "table" collate binary asc, '
        "rowid asc, parent collate binary asc, fkid asc limit ?"
    )
    assert parameters == (18,)
    assert autoflush is False
    assert inspect(pending_account).pending
    assert pending_account.id is None


async def test_foreign_key_check_normalizes_and_bounds_schema_names(client, session):
    parent_name = "parent_" + "p" * 300
    child_name = "child\n" + "c" * 300

    def quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    await session.execute(
        text(f"CREATE TABLE {quote_identifier(parent_name)} (id INTEGER PRIMARY KEY)")
    )
    await session.execute(
        text(
            f"CREATE TABLE {quote_identifier(child_name)} ("
            "id INTEGER PRIMARY KEY, parent_id INTEGER, "
            f"FOREIGN KEY(parent_id) REFERENCES {quote_identifier(parent_name)}(id))"
        )
    )
    await session.execute(
        text(
            f"INSERT INTO {quote_identifier(child_name)} (id, parent_id) "
            "VALUES (1, 404)"
        )
    )
    await session.commit()

    response = await client.get("/api/system/foreign-key-check")

    assert response.status_code == 200
    violation = response.json()["violations"][0]
    assert len(violation["child_table"]) == 256
    assert len(violation["parent_table"]) == 256
    assert "\n" not in violation["child_table"]
    assert violation["child_table"].startswith("child�")
    assert violation["parent_table"].startswith("parent_")


async def test_foreign_key_check_failure_is_sanitized(client, monkeypatch, caplog):
    secret_detail = "sqlite+aiosqlite:////private/operator.db?token=secret"

    async def fail_check(self, statement, *args, **kwargs):
        raise SQLAlchemyError(secret_detail)

    monkeypatch.setattr(AsyncSession, "execute", fail_check)
    with caplog.at_level(logging.WARNING, logger=database_service.__name__):
        response = await client.get("/api/system/foreign-key-check?limit=9")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unavailable",
        "backend": "sqlite",
        "returned_count": 0,
        "limit": 9,
        "truncated": False,
        "violations": [],
    }
    assert secret_detail not in response.text
    assert any(
        secret_detail in record.getMessage()
        for record in caplog.records
        if record.name == database_service.__name__
    )


async def test_foreign_key_check_openapi_uses_inferred_typed_response(client):
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/api/system/foreign-key-check"]["get"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert response_schema == {"$ref": "#/components/schemas/ForeignKeyCheckResponse"}

    response_model = document["components"]["schemas"]["ForeignKeyCheckResponse"]
    assert set(response_model["required"]) == {
        "status",
        "backend",
        "returned_count",
        "limit",
        "truncated",
        "violations",
    }
    assert set(response_model["properties"]["status"]["enum"]) == {
        "ok",
        "violations",
        "unavailable",
    }
    assert response_model["properties"]["backend"]["const"] == "sqlite"
    violation_model = document["components"]["schemas"]["ForeignKeyViolation"]
    assert {
        item.get("type")
        for item in violation_model["properties"]["child_row_id"]["anyOf"]
    } == {
        "integer",
        "null",
    }
    assert violation_model["properties"]["fk_constraint_index"]["minimum"] == 0

    limit_parameter = next(
        parameter
        for parameter in operation["parameters"]
        if parameter["name"] == "limit"
    )
    assert limit_parameter["schema"] == {
        "type": "integer",
        "maximum": 500,
        "minimum": 1,
        "default": 100,
        "title": "Limit",
    }
