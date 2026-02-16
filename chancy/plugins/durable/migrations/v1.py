from psycopg import AsyncCursor, sql
from psycopg.rows import DictRow

from chancy.migrate import Migration, Migrator


class V1Migration(Migration):
    async def up(self, migrator: Migrator, cursor: AsyncCursor[DictRow]):
        # Saved step results for replay
        await cursor.execute(
            sql.SQL("""
                CREATE TABLE {table} (
                    job_id UUID NOT NULL REFERENCES {jobs}(id) ON DELETE CASCADE,
                    step_name TEXT NOT NULL,
                    result JSONB,
                    seq BIGSERIAL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (job_id, step_name)
                )
            """).format(
                table=sql.Identifier(f"{migrator.prefix}durable_checkpoints"),
                jobs=sql.Identifier(f"{migrator.prefix}jobs"),
            )
        )

        # Emitted events (persist so late subscribers still get them)
        # Events are scoped to a specific job via job_id.
        await cursor.execute(
            sql.SQL("""
                CREATE TABLE {table} (
                    job_id UUID NOT NULL REFERENCES {jobs}(id) ON DELETE CASCADE,
                    event_name TEXT NOT NULL,
                    payload JSONB,
                    emitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (job_id, event_name)
                )
            """).format(
                table=sql.Identifier(f"{migrator.prefix}durable_events"),
                jobs=sql.Identifier(f"{migrator.prefix}jobs"),
            )
        )

        # Which jobs are waiting for which events
        await cursor.execute(
            sql.SQL("""
                CREATE TABLE {table} (
                    job_id UUID NOT NULL REFERENCES {jobs}(id) ON DELETE CASCADE,
                    step_name TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    timeout_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (job_id, step_name)
                )
            """).format(
                table=sql.Identifier(f"{migrator.prefix}durable_waits"),
                jobs=sql.Identifier(f"{migrator.prefix}jobs"),
            )
        )

        await cursor.execute(
            sql.SQL("""
                CREATE INDEX {index_name}
                ON {table} (event_name)
            """).format(
                index_name=sql.Identifier(
                    f"{migrator.prefix}durable_waits_event_idx"
                ),
                table=sql.Identifier(f"{migrator.prefix}durable_waits"),
            )
        )

    async def down(self, migrator: Migrator, cursor: AsyncCursor[DictRow]):
        await cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(f"{migrator.prefix}durable_waits")
            )
        )
        await cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(f"{migrator.prefix}durable_events")
            )
        )
        await cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {table}").format(
                table=sql.Identifier(f"{migrator.prefix}durable_checkpoints")
            )
        )
