from . import assets, jobs, system  # noqa: F401

routers = (jobs.router, assets.router, system.router)
