import asyncio
import logging

import pytest

from chancy import Chancy, Queue, QueuedJob, Worker
from chancy.plugins.durable import (
    DurablePlugin,
    DurableContext,
    SuspendExecution,
    durable,
)
from chancy.plugins.leadership import ImmediateLeadership

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_durable_tables(chancy):
    """Drop durable tables before the chancy fixture's teardown.

    They have FK references to chancy_jobs which would block core
    table drops.
    """
    yield
    for plugin in chancy.plugins.values():
        if hasattr(plugin, "migrate_key") and plugin.migrate_key() == "durable":
            await plugin.migrate(chancy, to_version=0)


@pytest_asyncio.fixture()
async def worker(chancy):
    """Override conftest worker to declare the async queue first.

    Worker.start() auto-declares Queue("default") with ProcessExecutor.
    By declaring with AsyncExecutor beforehand, the worker's no-upsert
    declare becomes a no-op and the queue keeps AsyncExecutor.
    """
    await chancy.declare(
        Queue("default", executor=Chancy.Executor.Async), upsert=True
    )
    async with Worker(chancy, shutdown_timeout=60) as w:
        yield w


# ------------------------------------------------------------------
# Helper step functions (must be module-level for importability)
# ------------------------------------------------------------------


async def add(a: int, b: int) -> int:
    return a + b


async def multiply(a: int, b: int) -> int:
    return a * b


def sync_add(a: int, b: int) -> int:
    return a + b


# ------------------------------------------------------------------
# Durable workflows (must be module-level for importability)
# ------------------------------------------------------------------


@durable()
async def basic_workflow(ctx: DurableContext):
    a = await ctx.step(add, a=1, b=2)
    b = await ctx.step(multiply, a=a, b=3)
    return b


@durable()
async def step_name_workflow(ctx: DurableContext):
    return await ctx.step(add, a=10, b=20)


@durable()
async def explicit_name_workflow(ctx: DurableContext):
    return await ctx.step(add, a=1, b=2, name="my_add")


@durable()
async def auto_increment_workflow(ctx: DurableContext):
    r1 = await ctx.step(add, a=1, b=1)
    r2 = await ctx.step(add, a=2, b=2)
    r3 = await ctx.step(add, a=3, b=3)
    return [r1, r2, r3]


@durable()
async def sync_step_workflow(ctx: DurableContext):
    return await ctx.step(sync_add, a=5, b=7)


@durable()
async def sleep_workflow(ctx: DurableContext):
    a = await ctx.step(add, a=1, b=2)
    await ctx.sleep(0)  # Suspend and immediately re-queue
    b = await ctx.step(multiply, a=a, b=10)
    return b


@durable()
async def event_wait_workflow(ctx: DurableContext):
    a = await ctx.step(add, a=1, b=2)
    payload = await ctx.wait_for_event("test:payment")
    b = await ctx.step(multiply, a=a, b=payload["amount"])
    return b


@durable()
async def pre_emitted_workflow(ctx: DurableContext):
    payload = await ctx.wait_for_event("pre_emitted")
    return payload


@durable()
async def event_timeout_workflow(ctx: DurableContext):
    payload = await ctx.wait_for_event("never_emitted", timeout=1)
    return payload


@durable()
async def crash_workflow(ctx: DurableContext):
    r = await ctx.step(add, a=1, b=2)

    # Fails on first attempt, succeeds on retry
    fail_key = f"__fail_tracker__{ctx.job.id}"
    if fail_key not in ctx._checkpoints:
        await ctx._save_checkpoint(fail_key, True)
        ctx._checkpoints[fail_key] = True
        ctx._accessed_steps.add(fail_key)
        raise ValueError("Simulated crash")

    ctx._accessed_steps.add(fail_key)
    return r


@durable()
async def multi_suspend_workflow(ctx: DurableContext):
    a = await ctx.step(add, a=1, b=2)
    await ctx.sleep(0)
    b = await ctx.step(multiply, a=a, b=3)
    await ctx.sleep(0)
    c = await ctx.step(add, a=b, b=1)
    return c


@durable()
async def concurrent_event_workflow(ctx: DurableContext):
    event_name = ctx.kwargs["event_name"]
    return await ctx.wait_for_event(event_name)


@durable()
async def orphaned_workflow(ctx: DurableContext):
    # Manually insert a checkpoint that won't be accessed
    await ctx._save_checkpoint("old_step", "stale_data")
    ctx._checkpoints["old_step"] = "stale_data"
    return await ctx.step(add, a=1, b=2)


@durable()
async def cancel_workflow(ctx: DurableContext):
    await ctx.wait_for_event("never_coming")


@durable()
async def rewind_workflow(ctx: DurableContext):
    a = await ctx.step(add, a=1, b=2)
    b = await ctx.step(multiply, a=a, b=3)
    c = await ctx.step(add, a=b, b=10, name="final_add")
    return c


@durable()
async def simple_workflow(ctx: DurableContext):
    return await ctx.step(add, a=1, b=2)


@durable()
async def checkpoints_workflow(ctx: DurableContext):
    await ctx.step(add, a=1, b=2)
    await ctx.step(multiply, a=3, b=4)
    await ctx.step(add, a=5, b=6, name="third")
    return True


@durable()
async def kwargs_workflow(ctx: DurableContext):
    return await ctx.step(add, a=ctx.kwargs["x"], b=ctx.kwargs["y"])


@durable()
async def invalidate_workflow(ctx: DurableContext):
    result = await ctx.step(add, a=1, b=2)
    await ctx.invalidate("add")
    return result


@durable()
async def wait_cancelled_event_workflow(ctx: DurableContext):
    payload = await ctx.wait_for_event("cancelled_event")
    return payload


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DECLARE = Queue("default", executor=Chancy.Executor.Async)


def durable_chancy_config():
    return {
        "plugins": [ImmediateLeadership(), DurablePlugin(poll_interval=1)],
        "no_default_plugins": True,
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_basic_step_replay(chancy: Chancy, worker):
    """Steps complete, checkpoints are saved, results are correct."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(basic_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    # Checkpoints should exist
    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 2
    assert checkpoints[0]["step_name"] == "add"
    assert checkpoints[0]["result"] == 3
    assert checkpoints[1]["step_name"] == "multiply"
    assert checkpoints[1]["result"] == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_step_name_from_function(chancy: Chancy, worker):
    """Default step name is derived from fn.__qualname__."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(step_name_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 1
    assert checkpoints[0]["step_name"] == "add"


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_explicit_step_name(chancy: Chancy, worker):
    """Explicit name= overrides the default."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(explicit_name_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert checkpoints[0]["step_name"] == "my_add"


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_same_name_auto_increment(chancy: Chancy, worker):
    """Same function called multiple times auto-increments the name."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(auto_increment_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    names = [c["step_name"] for c in checkpoints]
    assert names == ["add", "add#1", "add#2"]
    assert checkpoints[0]["result"] == 2
    assert checkpoints[1]["result"] == 4
    assert checkpoints[2]["result"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_sync_step(chancy: Chancy, worker):
    """Sync (non-async) step functions work."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(sync_step_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_sleep_suspension(chancy: Chancy, worker):
    """ctx.sleep() suspends and resumes after the duration."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(sleep_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=15)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    step_names = [c["step_name"] for c in checkpoints]
    assert "add" in step_names
    assert "multiply" in step_names
    # Sleep checkpoint should also exist
    sleep_names = [n for n in step_names if n.startswith("__sleep__")]
    assert len(sleep_names) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_event_wait_and_emit(chancy: Chancy, worker):
    """wait_for_event suspends, emit_event resumes with payload."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(event_wait_workflow.job.with_kwargs())

    # Give the job time to reach the wait_for_event and suspend
    await asyncio.sleep(2)

    # Emit the event (scoped to the job)
    await DurablePlugin.emit_event(
        chancy, str(ref.identifier), "test:payment", {"amount": 5}
    )

    result = await chancy.wait_for_job(ref, timeout=15)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    event_checkpoints = [
        c for c in checkpoints if c["step_name"].startswith("__event__")
    ]
    assert len(event_checkpoints) == 1
    assert event_checkpoints[0]["result"] == {"amount": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_event_already_emitted(chancy: Chancy, worker):
    """If event is emitted before wait_for_event runs, returns immediately."""
    await chancy.declare(DECLARE, upsert=True)

    # Push first (job must exist for FK constraint), then emit immediately.
    # The event reaches the DB before the workflow calls wait_for_event
    # (or if the workflow already suspended, emit_event wakes it).
    # Either path leads to a successful completion.
    ref = await chancy.push(pre_emitted_workflow.job.with_kwargs())
    await DurablePlugin.emit_event(
        chancy, str(ref.identifier), "pre_emitted", {"pre": True}
    )

    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_event_timeout(chancy: Chancy, worker):
    """wait_for_event with timeout returns None when it expires."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(event_timeout_workflow.job.with_kwargs())

    # Wait for timeout expiration + poll interval + execution
    result = await chancy.wait_for_job(ref, timeout=15)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_crash_recovery(chancy: Chancy, worker):
    """Steps replay from checkpoints after a failure."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(crash_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=15)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_multiple_suspensions(chancy: Chancy, worker):
    """Function with multiple suspend/resume cycles."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(multi_suspend_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=20)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    step_checkpoints = [
        c for c in checkpoints if not c["step_name"].startswith("__")
    ]
    assert len(step_checkpoints) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_concurrent_events(chancy: Chancy, worker):
    """Multiple jobs waiting for different events all get woken."""
    await chancy.declare(DECLARE, upsert=True)

    ref1 = await chancy.push(
        concurrent_event_workflow.job.with_kwargs(event_name="evt:1")
    )
    ref2 = await chancy.push(
        concurrent_event_workflow.job.with_kwargs(event_name="evt:2")
    )

    await asyncio.sleep(3)

    await DurablePlugin.emit_event(
        chancy, str(ref1.identifier), "evt:1", {"v": 1}
    )
    await DurablePlugin.emit_event(
        chancy, str(ref2.identifier), "evt:2", {"v": 2}
    )

    r1 = await chancy.wait_for_job(ref1, timeout=15)
    r2 = await chancy.wait_for_job(ref2, timeout=15)

    assert r1 is not None and r1.state == QueuedJob.State.SUCCEEDED
    assert r2 is not None and r2.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_orphaned_checkpoint_warning(chancy: Chancy, worker, caplog):
    """Orphaned checkpoints produce a warning on completion."""
    await chancy.declare(DECLARE, upsert=True)

    with caplog.at_level(logging.WARNING, logger="chancy.plugins.durable"):
        ref = await chancy.push(orphaned_workflow.job.with_kwargs())
        result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED
    assert any("orphaned checkpoints" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_suspend_not_caught_by_except_exception():
    """SuspendExecution (BaseException) is not caught by except Exception."""
    caught = False
    try:
        try:
            raise SuspendExecution(wake_at=None)
        except Exception:
            caught = True
    except SuspendExecution:
        pass

    assert not caught, "SuspendExecution was caught by except Exception"


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_cancel_durable_job(chancy: Chancy, worker):
    """cancel_durable_job deletes waits and sets job to failed."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(cancel_workflow.job.with_kwargs())

    # Wait for suspension
    await asyncio.sleep(2)

    job = await chancy.get_job(ref)
    assert job is not None
    await DurablePlugin.cancel_durable_job(chancy, job.id)

    # Verify the job is now failed
    job = await chancy.get_job(ref)
    assert job is not None
    assert job.state == QueuedJob.State.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_rewind_to_step(chancy: Chancy, worker):
    """Rewind deletes target and subsequent checkpoints, re-queues."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(rewind_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    # Should have 3 checkpoints
    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 3

    # Rewind to "multiply" — should delete multiply and final_add
    await DurablePlugin.rewind_to_step(chancy, result.id, "multiply")

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 1
    assert checkpoints[0]["step_name"] == "add"

    # Job should complete again after rewind
    result = await chancy.wait_for_job(ref, timeout=15)
    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_rewind_invalid_step(chancy: Chancy, worker):
    """rewind_to_step raises ValueError for nonexistent checkpoint."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(simple_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None

    with pytest.raises(ValueError, match="No checkpoint"):
        await DurablePlugin.rewind_to_step(chancy, result.id, "nonexistent")


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_get_job_checkpoints(chancy: Chancy, worker):
    """get_job_checkpoints returns all checkpoints in creation order."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(checkpoints_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    assert len(checkpoints) == 3
    assert checkpoints[0]["step_name"] == "add"
    assert checkpoints[0]["result"] == 3
    assert checkpoints[1]["step_name"] == "multiply"
    assert checkpoints[1]["result"] == 12
    assert checkpoints[2]["step_name"] == "third"
    assert checkpoints[2]["result"] == 11

    # Verify ordering by created_at
    for i in range(len(checkpoints) - 1):
        assert checkpoints[i]["created_at"] <= checkpoints[i + 1]["created_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_kwargs_access(chancy: Chancy, worker):
    """ctx.kwargs returns the job's keyword arguments."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(kwargs_workflow.job.with_kwargs(x=10, y=20))
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_cleanup_is_noop(chancy: Chancy, worker):
    """cleanup() is a no-op — CASCADE from jobs table handles everything."""
    await chancy.declare(DECLARE, upsert=True)

    durable_plugin = None
    for plugin in chancy.plugins.values():
        if isinstance(plugin, DurablePlugin):
            durable_plugin = plugin
            break

    assert durable_plugin is not None
    deleted = await durable_plugin.cleanup(chancy)
    assert deleted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_cancel_event(chancy: Chancy, worker):
    """cancel_event() removes a pending event from the database."""
    await chancy.declare(DECLARE, upsert=True)

    # Push a job (needed for FK constraint on events table)
    ref = await chancy.push(wait_cancelled_event_workflow.job.with_kwargs())
    job_id = str(ref.identifier)

    # Emit an event for this job
    await DurablePlugin.emit_event(chancy, job_id, "cancelled_event", {"v": 1})

    # Verify the event exists
    async with chancy.pool.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT count(*) FROM {chancy.prefix}durable_events "
                "WHERE job_id = %s AND event_name = 'cancelled_event'",
                (job_id,),
            )
            row = await cursor.fetchone()
            assert row[0] == 1

    # Cancel the event
    await DurablePlugin.cancel_event(chancy, job_id, "cancelled_event")

    # Verify the event is gone
    async with chancy.pool.connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT count(*) FROM {chancy.prefix}durable_events "
                "WHERE job_id = %s AND event_name = 'cancelled_event'",
                (job_id,),
            )
            row = await cursor.fetchone()
            assert row[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("chancy", [durable_chancy_config()], indirect=True)
async def test_invalidate_checkpoint(chancy: Chancy, worker):
    """ctx.invalidate() deletes a checkpoint from the database."""
    await chancy.declare(DECLARE, upsert=True)

    ref = await chancy.push(invalidate_workflow.job.with_kwargs())
    result = await chancy.wait_for_job(ref, timeout=10)

    assert result is not None
    assert result.state == QueuedJob.State.SUCCEEDED

    # The "add" checkpoint should have been deleted by ctx.invalidate("add")
    checkpoints = await DurablePlugin.get_job_checkpoints(chancy, result.id)
    step_names = [c["step_name"] for c in checkpoints]
    assert "add" not in step_names
