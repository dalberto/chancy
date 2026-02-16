"""
Durable execution plugin for Chancy.

Provides a complementary primitive to DAG-based workflows: a single
long-running function that can checkpoint its progress, suspend to wait
for events or timers, and resume from where it left off even after
crashes.

The design uses **checkpoint memoization** — step execution order doesn't
matter for correctness, only step names do. This sidesteps the
determinism problem that plagues replay-based systems (Temporal, Restate).

Usage
-----

.. code-block:: python

    from chancy import Chancy, Queue
    from chancy.plugins.durable import DurablePlugin, DurableContext, durable

    @durable()
    async def order_workflow(ctx: DurableContext):
        order = await ctx.step(fetch_order, order_id=ctx.kwargs["order_id"])
        approval = await ctx.wait_for_event("payment")
        await ctx.step(process_order, order=order, payment=approval)
        await ctx.sleep(86400)
        await ctx.step(send_follow_up, order_id=ctx.kwargs["order_id"])

    async with Chancy(
        "postgresql://localhost/postgres",
        plugins=[DurablePlugin()]
    ) as chancy:
        await chancy.migrate()
        await chancy.declare(Queue("default", executor=Chancy.Executor.Async))
        ref = await chancy.push(order_workflow.job.with_kwargs(order_id="12345"))

    # From anywhere (caller must know the job ID):
    await DurablePlugin.emit_event(chancy, ref.id, "payment", {"amount": 99.99})

Versioning
----------

Step names are your version identifiers:

- **Add a step**: executes normally (no checkpoint exists).
- **Remove a step**: old checkpoint becomes orphaned (harmless, warned).
- **Rename a step**: old checkpoint orphaned, new step executes fresh.
- **Change step logic**: old checkpoint returns stale result.
- **Reorder steps**: no effect on correctness.

Time-Travel
-----------

Rewind to any checkpoint and re-execute from that point:

.. code-block:: python

    await DurablePlugin.rewind_to_step(chancy, job_id, "call_llm#1")
"""

import contextvars
import dataclasses
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from psycopg import sql
from psycopg.rows import dict_row

from chancy.app import Chancy
from chancy.job import QueuedJob, job
from chancy.plugin import Plugin
from chancy.utils import json_dumps
from chancy.worker import Worker

logger = logging.getLogger(__name__)

_durable_pool: contextvars.ContextVar = contextvars.ContextVar(
    "_durable_pool", default=None
)
_durable_prefix: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_durable_prefix", default="chancy_"
)


class SuspendExecution(BaseException):
    """Raised to suspend a durable function.

    Inherits from BaseException so user code with
    ``except Exception:`` cannot accidentally catch it.
    """

    def __init__(
        self,
        *,
        wake_at: datetime | None = None,
        event_name: str | None = None,
    ):
        self.wake_at = wake_at
        self.event_name = event_name
        super().__init__()


class _DurableSuspend(Exception):
    """Internal marker flowing through the executor/plugin pipeline.

    Created by the ``@durable()`` wrapper when it catches
    :class:`SuspendExecution`. Carries the same fields but inherits from
    ``Exception`` so the executor processes it normally.
    """

    def __init__(self, suspend: SuspendExecution):
        self.wake_at = suspend.wake_at
        self.event_name = suspend.event_name
        super().__init__()


class DurableContext:
    """Context object passed to durable functions.

    Provides methods for checkpointed steps, durable sleep, and
    event-based suspension.
    """

    def __init__(self, job: QueuedJob):
        self._job = job
        self._checkpoints: dict[str, Any] = {}
        self._accessed_steps: set[str] = set()
        self._step_counts: dict[str, int] = {}
        # Auto-incrementing counter for sleep checkpoints. Produces
        # __sleep__0, __sleep__1, etc. Resets each replay (new instance).
        self._sleep_count: int = 0

    @property
    def kwargs(self) -> dict[str, Any]:
        """The job's keyword arguments."""
        return self._job.kwargs or {}

    @property
    def job(self) -> QueuedJob:
        """The underlying QueuedJob instance."""
        return self._job

    async def step(
        self,
        fn: Callable,
        *args: Any,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a checkpointed step.

        On first execution the function runs and its result is saved.
        On replay the saved result is returned immediately.

        When the same base name is used more than once (common in loops),
        an auto-incrementing suffix is appended:
        ``name``, ``name#1``, ``name#2``, ...

        :param fn: The function to execute.
        :param name: Explicit step name. Defaults to ``fn.__qualname__``.
        :returns: The step result.
        """
        base_name = name or fn.__qualname__
        count = self._step_counts.get(base_name, 0)
        self._step_counts[base_name] = count + 1
        step_name = f"{base_name}#{count}" if count > 0 else base_name

        self._accessed_steps.add(step_name)

        if step_name in self._checkpoints:
            return self._checkpoints[step_name]

        if inspect.iscoroutinefunction(fn):
            result = await fn(*args, **kwargs)
        else:
            result = fn(*args, **kwargs)

        await self._save_checkpoint(step_name, result)
        self._checkpoints[step_name] = result
        return result

    async def sleep(self, seconds: int | float) -> None:
        """Suspend the durable function for *seconds*.

        The job is re-queued with ``scheduled_at`` set to
        ``now + seconds``.  On resume, all prior steps replay from
        checkpoints.
        """
        from datetime import timedelta

        wake_at = datetime.now(tz=timezone.utc) + timedelta(seconds=seconds)

        # Uses __sleep__N naming to avoid collisions with user step names.
        # Counter resets each replay, so sleep ordering must be stable.
        checkpoint_name = f"__sleep__{self._sleep_count}"
        self._sleep_count += 1
        if checkpoint_name in self._checkpoints:
            self._accessed_steps.add(checkpoint_name)
            return

        await self._save_checkpoint(checkpoint_name, True)
        self._checkpoints[checkpoint_name] = True
        self._accessed_steps.add(checkpoint_name)
        raise SuspendExecution(wake_at=wake_at)

    async def wait_for_event(
        self,
        event_name: str,
        *,
        timeout: float | None = None,
    ) -> Any | None:
        """Suspend until an external event is emitted.

        If the event has already been emitted, returns its payload
        immediately.  Otherwise registers a wait and suspends.

        Events are scoped to the current job — only events emitted
        targeting this job's ID will match.

        :param event_name: The event name to wait for.
        :param timeout: Optional timeout in seconds. If the event is not
            emitted within this time, returns ``None``.
        :returns: The event payload, or ``None`` on timeout.
        """
        # __event__ prefix reserves this namespace for internal
        # checkpoints, avoiding collisions with user step names.
        checkpoint_name = f"__event__{event_name}"

        # Already have a checkpoint for this event
        if checkpoint_name in self._checkpoints:
            self._accessed_steps.add(checkpoint_name)
            return self._checkpoints[checkpoint_name]

        pool = _durable_pool.get()
        prefix = _durable_prefix.get()

        timeout_at = None
        if timeout is not None:
            from datetime import timedelta

            timeout_at = datetime.now(tz=timezone.utc) + timedelta(
                seconds=timeout
            )

        async with pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cursor:
                    # Check if event already exists for this job
                    await cursor.execute(
                        sql.SQL("""
                            SELECT payload
                            FROM {events}
                            WHERE job_id = %(job_id)s
                              AND event_name = %(event_name)s
                        """).format(
                            events=sql.Identifier(f"{prefix}durable_events"),
                        ),
                        {
                            "job_id": self._job.id,
                            "event_name": event_name,
                        },
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        payload = row["payload"]
                        await self._save_checkpoint(checkpoint_name, payload)
                        self._checkpoints[checkpoint_name] = payload
                        self._accessed_steps.add(checkpoint_name)
                        return payload

                    # No event yet — register wait
                    await cursor.execute(
                        sql.SQL("""
                            INSERT INTO {waits}
                                (job_id, step_name, event_name, timeout_at)
                            VALUES
                                (%(job_id)s, %(step_name)s, %(event_name)s,
                                 %(timeout_at)s)
                            ON CONFLICT (job_id, step_name) DO NOTHING
                        """).format(
                            waits=sql.Identifier(f"{prefix}durable_waits"),
                        ),
                        {
                            "job_id": self._job.id,
                            "step_name": checkpoint_name,
                            "event_name": event_name,
                            "timeout_at": timeout_at,
                        },
                    )

        # Suspend AFTER the transaction commits so the wait row is
        # visible to concurrent emit_event calls.
        raise SuspendExecution(
            event_name=event_name,
            wake_at=timeout_at,
        )

    async def invalidate(self, name: str) -> None:
        """Delete a checkpoint, forcing re-execution on the next run.

        :param name: The step name whose checkpoint should be deleted.
        """
        pool = _durable_pool.get()
        prefix = _durable_prefix.get()

        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql.SQL("""
                        DELETE FROM {checkpoints}
                        WHERE job_id = %(job_id)s
                          AND step_name = %(step_name)s
                    """).format(
                        checkpoints=sql.Identifier(
                            f"{prefix}durable_checkpoints"
                        ),
                    ),
                    {"job_id": self._job.id, "step_name": name},
                )
        self._checkpoints.pop(name, None)

    async def _load_checkpoints(self) -> None:
        pool = _durable_pool.get()
        prefix = _durable_prefix.get()

        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    sql.SQL("""
                        SELECT step_name, result
                        FROM {checkpoints}
                        WHERE job_id = %(job_id)s
                    """).format(
                        checkpoints=sql.Identifier(
                            f"{prefix}durable_checkpoints"
                        ),
                    ),
                    {"job_id": self._job.id},
                )
                rows = await cursor.fetchall()
                self._checkpoints = {
                    row["step_name"]: row["result"] for row in rows
                }

    async def _save_checkpoint(self, step_name: str, result: Any) -> None:
        pool = _durable_pool.get()
        prefix = _durable_prefix.get()

        try:
            dumped = json_dumps(result)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Step '{step_name}' returned a non-JSON-serializable "
                f"result: {e}"
            ) from e

        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql.SQL("""
                        INSERT INTO {checkpoints}
                            (job_id, step_name, result)
                        VALUES
                            (%(job_id)s, %(step_name)s, %(result)s)
                        -- DO NOTHING, not DO UPDATE: first successful
                        -- write wins. A crash-retry must not overwrite
                        -- the original result (at-least-once idempotency).
                        ON CONFLICT (job_id, step_name) DO NOTHING
                    """).format(
                        checkpoints=sql.Identifier(
                            f"{prefix}durable_checkpoints"
                        ),
                    ),
                    {
                        "job_id": self._job.id,
                        "step_name": step_name,
                        "result": dumped,
                    },
                )

    async def _on_complete(self) -> None:
        """Called on successful completion. Warns about orphaned
        checkpoints."""
        orphaned = set(self._checkpoints.keys()) - self._accessed_steps
        if orphaned:
            logger.warning(
                "Durable job %s has orphaned checkpoints that were "
                "loaded but never accessed: %s. These may be from "
                "renamed or removed steps.",
                self._job.id,
                orphaned,
            )


def durable(**job_kwargs):
    """Decorator that turns an async function into a durable function.

    The decorated function receives a :class:`DurableContext` as its
    sole argument.  It can still be called normally (the ``ctx`` must
    be provided manually in that case) and exposes a ``.job`` attribute
    for pushing.

    .. code-block:: python

        @durable()
        async def my_workflow(ctx: DurableContext):
            result = await ctx.step(some_function, arg=1)
            await ctx.sleep(60)
            await ctx.step(another_function, data=result)

    :param job_kwargs: Additional keyword arguments forwarded to
        :func:`chancy.job.job`.
    """

    def decorator(fn):
        # The executor calls get_function_and_kwargs() which merges
        # job.kwargs with _chancy_job and passes them all as keyword
        # args. **_kwargs captures the user kwargs here, but they're
        # not forwarded — the durable function receives a DurableContext
        # as its sole argument and accesses kwargs via ctx.kwargs
        # (which reads job.kwargs).
        async def wrapper(*, _chancy_job: QueuedJob, **_kwargs):
            ctx = DurableContext(_chancy_job)
            await ctx._load_checkpoints()
            try:
                result = await fn(ctx)
                await ctx._on_complete()
                return result
            except SuspendExecution as suspend:
                raise _DurableSuspend(suspend) from None

        wrapper.__module__ = fn.__module__
        wrapper.__qualname__ = fn.__qualname__

        job_kwargs.setdefault("meta", {})
        job_kwargs["meta"]["__durable__"] = True
        job_kwargs.setdefault("max_attempts", 2**31 - 1)

        return job(**job_kwargs)(wrapper)

    return decorator


class DurablePlugin(Plugin):
    """Plugin providing durable execution support.

    :param poll_interval: How often (in seconds) to poll for timed-out
        event waits.  Default is 30 seconds.
    """

    def __init__(
        self,
        *,
        poll_interval: int = 30,
    ):
        super().__init__()
        self.poll_interval = poll_interval

    async def run(self, worker: Worker, chancy: Chancy):
        """Poll for expired event wait timeouts."""
        while await self.sleep(self.poll_interval):
            await self._expire_waits(chancy)

    async def _expire_waits(self, chancy: Chancy) -> None:
        """Wake jobs whose wait_for_event timeout has expired."""
        async with chancy.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cursor:
                    # Find and delete expired waits
                    await cursor.execute(
                        sql.SQL("""
                            DELETE FROM {waits} w
                            WHERE w.timeout_at IS NOT NULL
                              AND w.timeout_at <= NOW()
                            RETURNING w.job_id, w.step_name
                        """).format(
                            waits=sql.Identifier(
                                f"{chancy.prefix}durable_waits"
                            ),
                        ),
                    )
                    expired = await cursor.fetchall()
                    if not expired:
                        return

                    # Save a None checkpoint for each expired wait so
                    # on replay wait_for_event returns None immediately
                    for row in expired:
                        await cursor.execute(
                            sql.SQL("""
                                INSERT INTO {checkpoints}
                                    (job_id, step_name, result)
                                VALUES
                                    (%(job_id)s, %(step_name)s, 'null')
                                ON CONFLICT (job_id, step_name) DO NOTHING
                            """).format(
                                checkpoints=sql.Identifier(
                                    f"{chancy.prefix}durable_checkpoints"
                                ),
                            ),
                            {
                                "job_id": row["job_id"],
                                "step_name": row["step_name"],
                            },
                        )

                    # Wake the expired jobs
                    job_ids = [row["job_id"] for row in expired]
                    await cursor.execute(
                        sql.SQL("""
                            UPDATE {jobs} j
                            SET state = 'retrying',
                                scheduled_at = NOW()
                            WHERE j.id = ANY(%(job_ids)s)
                              AND j.state IN (
                                  'retrying', 'pending', 'failed'
                              )
                            RETURNING j.queue
                        """).format(
                            jobs=sql.Identifier(f"{chancy.prefix}jobs"),
                        ),
                        {"job_ids": job_ids},
                    )
                    rows = await cursor.fetchall()
                    for row in rows:
                        await chancy.notify(
                            cursor,
                            "queue.pushed",
                            {"q": row["queue"]},
                        )

    async def on_job_starting(
        self, *, job: QueuedJob, worker: Worker
    ) -> QueuedJob:
        if job.meta.get("__durable__"):
            _durable_pool.set(worker.chancy.pool)
            _durable_prefix.set(worker.chancy.prefix)
        return job

    async def on_job_completed(
        self,
        *,
        worker: Worker,
        job: QueuedJob,
        exc: Exception | None = None,
        result: Any | None = None,
    ) -> QueuedJob:
        if not job.meta.get("__durable__"):
            return job

        if isinstance(exc, _DurableSuspend):
            return dataclasses.replace(
                job,
                state=QueuedJob.State.RETRYING,
                scheduled_at=exc.wake_at or datetime.now(tz=timezone.utc),
                errors=job.errors[:-1],  # Remove _DurableSuspend tb
                completed_at=None,
            )

        return job

    async def cleanup(self, chancy: Chancy) -> int | None:
        """No-op.  All durable tables use ``ON DELETE CASCADE`` from the
        jobs table, so the Pruner plugin handles cleanup automatically.
        """
        return 0

    # ------------------------------------------------------------------
    # Class methods for external use
    # ------------------------------------------------------------------

    @classmethod
    async def emit_event(
        cls,
        chancy: Chancy,
        job_id: str,
        event_name: str,
        payload: Any = None,
    ) -> None:
        """Emit an event, waking a job waiting for it.

        Events are scoped to a specific job.  Uses "latest wins"
        semantics: re-emitting overwrites the previous payload.

        :param chancy: The Chancy application instance.
        :param job_id: The target job ID.
        :param event_name: The event name.
        :param payload: JSON-serializable payload.
        """
        async with chancy.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cursor:
                    await cls.emit_event_ex(
                        cursor, chancy, job_id, event_name, payload
                    )

    @classmethod
    async def emit_event_ex(
        cls,
        cursor,
        chancy: Chancy,
        job_id: str,
        event_name: str,
        payload: Any = None,
    ) -> None:
        """Transaction-aware version of :meth:`emit_event`.

        :param cursor: An open database cursor.
        :param chancy: The Chancy application instance.
        :param job_id: The target job ID.
        :param event_name: The event name.
        :param payload: JSON-serializable payload.
        """
        dumped = json_dumps(payload) if payload is not None else "null"

        # Upsert event (latest wins)
        await cursor.execute(
            sql.SQL("""
                INSERT INTO {events} (job_id, event_name, payload)
                VALUES (%(job_id)s, %(name)s, %(payload)s)
                ON CONFLICT (job_id, event_name) DO UPDATE
                SET payload = EXCLUDED.payload,
                    emitted_at = NOW()
            """).format(
                events=sql.Identifier(f"{chancy.prefix}durable_events"),
            ),
            {"job_id": job_id, "name": event_name, "payload": dumped},
        )

        # Wake the waiting job, discover its queue
        await cursor.execute(
            sql.SQL("""
                WITH waiting AS (
                    DELETE FROM {waits} w
                    WHERE w.job_id = %(job_id)s
                      AND w.event_name = %(name)s
                    RETURNING w.job_id
                ),
                updated AS (
                    UPDATE {jobs} j
                    SET state = 'retrying',
                        scheduled_at = NOW()
                    FROM waiting w
                    WHERE j.id = w.job_id
                      AND j.state IN ('retrying', 'pending', 'failed')
                    RETURNING j.queue
                )
                SELECT DISTINCT queue FROM updated
            """).format(
                waits=sql.Identifier(f"{chancy.prefix}durable_waits"),
                jobs=sql.Identifier(f"{chancy.prefix}jobs"),
            ),
            {"job_id": job_id, "name": event_name},
        )

        rows = await cursor.fetchall()
        for row in rows:
            await chancy.notify(cursor, "queue.pushed", {"q": row["queue"]})

    @classmethod
    async def cancel_event(
        cls, chancy: Chancy, job_id: str, event_name: str
    ) -> None:
        """Remove a pending event.

        :param chancy: The Chancy application instance.
        :param job_id: The target job ID.
        :param event_name: The event name to remove.
        """
        async with chancy.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    sql.SQL("""
                        DELETE FROM {events}
                        WHERE job_id = %(job_id)s
                          AND event_name = %(name)s
                    """).format(
                        events=sql.Identifier(f"{chancy.prefix}durable_events"),
                    ),
                    {"job_id": job_id, "name": event_name},
                )

    @classmethod
    async def cancel_durable_job(cls, chancy: Chancy, job_id: str) -> None:
        """Cancel a suspended durable job.

        Deletes any active waits and sets the job state to failed.

        :param chancy: The Chancy application instance.
        :param job_id: The job ID to cancel.
        """
        async with chancy.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        sql.SQL("""
                            DELETE FROM {waits}
                            WHERE job_id = %(job_id)s
                        """).format(
                            waits=sql.Identifier(
                                f"{chancy.prefix}durable_waits"
                            ),
                        ),
                        {"job_id": job_id},
                    )
                    await cursor.execute(
                        sql.SQL("""
                            UPDATE {jobs}
                            SET state = 'failed',
                                completed_at = NOW()
                            WHERE id = %(job_id)s
                        """).format(
                            jobs=sql.Identifier(f"{chancy.prefix}jobs"),
                        ),
                        {"job_id": job_id},
                    )

    @classmethod
    async def rewind_to_step(
        cls, chancy: Chancy, job_id: str, step_name: str
    ) -> None:
        """Rewind a durable job to a specific step.

        Deletes the named checkpoint and all subsequent ones (by
        sequence number), clears any active waits, and re-queues the job.
        On replay, earlier steps return their cached results while the
        rewound step and everything after it execute fresh.

        :param chancy: The Chancy application instance.
        :param job_id: The job ID to rewind.
        :param step_name: The checkpoint to rewind to.
        :raises ValueError: If the checkpoint does not exist.
        """
        async with chancy.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor(row_factory=dict_row) as cursor:
                    # Find the target checkpoint
                    await cursor.execute(
                        sql.SQL("""
                            SELECT seq
                            FROM {checkpoints}
                            WHERE job_id = %(job_id)s
                              AND step_name = %(step_name)s
                        """).format(
                            checkpoints=sql.Identifier(
                                f"{chancy.prefix}durable_checkpoints"
                            ),
                        ),
                        {
                            "job_id": job_id,
                            "step_name": step_name,
                        },
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise ValueError(
                            f"No checkpoint '{step_name}' found "
                            f"for job {job_id}"
                        )

                    # Delete this checkpoint and all after it
                    await cursor.execute(
                        sql.SQL("""
                            DELETE FROM {checkpoints}
                            WHERE job_id = %(job_id)s
                              AND seq >= %(seq)s
                        """).format(
                            checkpoints=sql.Identifier(
                                f"{chancy.prefix}durable_checkpoints"
                            ),
                        ),
                        {
                            "job_id": job_id,
                            "seq": row["seq"],
                        },
                    )

                    # Clear any active waits
                    await cursor.execute(
                        sql.SQL("""
                            DELETE FROM {waits}
                            WHERE job_id = %(job_id)s
                        """).format(
                            waits=sql.Identifier(
                                f"{chancy.prefix}durable_waits"
                            ),
                        ),
                        {"job_id": job_id},
                    )

                    # Re-queue the job
                    await cursor.execute(
                        sql.SQL("""
                            UPDATE {jobs}
                            SET state = 'retrying',
                                scheduled_at = NOW(),
                                completed_at = NULL
                            WHERE id = %(job_id)s
                            RETURNING queue
                        """).format(
                            jobs=sql.Identifier(f"{chancy.prefix}jobs"),
                        ),
                        {"job_id": job_id},
                    )
                    result = await cursor.fetchone()
                    if result:
                        await chancy.notify(
                            cursor,
                            "queue.pushed",
                            {"q": result["queue"]},
                        )

    @classmethod
    async def get_job_checkpoints(
        cls, chancy: Chancy, job_id: str
    ) -> list[dict[str, Any]]:
        """Return all checkpoints for a durable job.

        :param chancy: The Chancy application instance.
        :param job_id: The job ID.
        :returns: A list of checkpoint dicts with ``step_name``,
            ``result``, and ``created_at`` keys, ordered by sequence
            number.
        """
        async with chancy.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    sql.SQL("""
                        SELECT step_name, result, created_at
                        FROM {checkpoints}
                        WHERE job_id = %(job_id)s
                        ORDER BY seq ASC
                    """).format(
                        checkpoints=sql.Identifier(
                            f"{chancy.prefix}durable_checkpoints"
                        ),
                    ),
                    {"job_id": job_id},
                )
                return await cursor.fetchall()

    # ------------------------------------------------------------------
    # Plugin metadata
    # ------------------------------------------------------------------

    def migrate_package(self) -> str:
        return "chancy.plugins.durable.migrations"

    def migrate_key(self) -> str:
        return "durable"

    def get_tables(self) -> list[str]:
        return [
            "durable_checkpoints",
            "durable_events",
            "durable_waits",
        ]

    @staticmethod
    def get_identifier() -> str:
        return "chancy.durable"
