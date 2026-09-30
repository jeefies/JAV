"""Runtime backend base classes.

A backend executes exactly one job at a time; the Supervisor enforces the
single-ACTIVE-profile mutex across the whole box.
"""
from __future__ import annotations

import asyncio


class BackendError(RuntimeError):
    pass


class BackendCrash(BackendError):
    pass


class StartTimeout(BackendError):
    pass


class BaseBackend:
    kind = "base"

    def __init__(self, profile):
        self.profile = profile
        self.pid: int | None = None

    async def start(self):
        raise NotImplementedError

    async def stop(self, reason: str = "stop"):
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError

    async def submit(self, job_id: str, task: dict) -> dict:
        """Returns {'status': 'success'|'failed', 'paths': [...], 'error': str|None}."""
        raise NotImplementedError

    async def cancel(self, job_id: str):
        """Best-effort interrupt; result is discarded by the scheduler."""
        return None

    def deliver(self, payload: dict) -> bool:
        """Internal-callback entry point; True if this backend owns it."""
        return False

    def pipeline_status(self, payload: dict) -> bool:
        return False


def _await_sync(fn, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, fn, *args)
