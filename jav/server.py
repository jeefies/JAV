"""JAV server entrypoint: FastAPI app factory + lifespan (JAV-DESIGN 17).

uvicorn --workers 1 keeps the scheduler singleton; HTTP layer is async."""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from starlette.responses import JSONResponse

from . import config
from .api import routers
from .runtime.supervisor import Supervisor
from .scheduler import EventBus, Scheduler
from .store import Store


@dataclass
class Ctx:
    store: Store
    sup: Supervisor
    sched: Scheduler
    bus: EventBus


def create_app(profiles: dict | None = None, store: Store | None = None,
               sup: Supervisor | None = None, bus: EventBus | None = None,
               sched: Scheduler | None = None) -> FastAPI:
    config.ensure_dirs()
    store = store or Store()
    sup = sup or Supervisor(store, profiles)
    bus = bus or EventBus()
    sched = sched or Scheduler(store, sup, bus)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sup.sweep_orphans()
        recovered = store.recover_interrupted()
        if recovered:
            store.log_event(None, None, f"crash_recovery_requeued:{len(recovered)}")
        await sched.start()
        try:
            yield
        finally:
            await sched.stop()
            await sup.shutdown("exit", lane="both")

    app = FastAPI(title="JAV — Jeefy Audio-Video Generation Platform",
                  version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    # 默认 /openapi.json|/docs|/redoc 关闭（未策划的 schema 会暴露 internal 端点）；
    # 公开契约只走 GET /v1/openapi.json（jav.openapi 策划视图）。
    app.ctx = Ctx(store=store, sup=sup, sched=sched, bus=bus)
    if config.API_TOKEN:
        @app.middleware("http")
        async def bearer_auth(request, call_next):
            path = request.url.path
            if (request.method == "GET" or path == "/v1/health"
                    or path.startswith("/v1/internal/")):
                return await call_next(request)
            if not hmac.compare_digest(
                    request.headers.get("authorization", ""),
                    f"Bearer {config.API_TOKEN}"):
                return JSONResponse({"detail": "invalid or missing bearer token"},
                                    status_code=401)
            return await call_next(request)
    for r in routers:
        app.include_router(r)
    return app


def main():
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=config.SERVICE_PORT, log_level="info")


if __name__ == "__main__":
    main()
