"""Curated public OpenAPI 3.1 contract (GET /v1/openapi.json).

Post-processes the FastAPI generator: strips /v1/internal/*, tags by
resource, annotates per-operation contract summaries (zh + x-summary-en for
bilingual doc frontends), and declares the bearer scheme on every non-GET
operation (mirrors the server middleware). Request-body schemas come from
the Pydantic models, so they cannot drift; descriptions are the
human/LLM-facing contract notes (API.md remains the authoritative prose)."""
from __future__ import annotations

INFO_DESCRIPTION = (
    "JAV 统一生成服务公开契约。任务生命周期：queued → running → completed/"
    "failed/cancelled（SSE /v1/jobs/{id}/events 可订阅）。公开字段只用 `id`；"
    "ComfyUI 细节不出 API 层。非 GET 端点需 `Authorization: Bearer`（健康/查询类 GET 公开）。"
    "provider.workflow 组合、输入槽位与参数范围以服务端 `GET /v1/capabilities` 与 "
    "API.md 为准；未在工作流中声明的输入键会被 provider 白名单拒绝（400）。\n\n"
    "Dubbing note: cosyvoice tasks may transparently execute on the CPU fallback "
    "lane while the GPU is occupied by another runtime family — the task keeps "
    "`runtime_profile: cosyvoice`; the receipt meta reports the actual device."
)

TAGS = [
    {"name": "jobs", "description": "任务提交/查询/取消/产物/SSE"},
    {"name": "batches", "description": "批量任务（shared 默认 + 逐任务覆盖）"},
    {"name": "assets", "description": "素材（内容寻址 sha256，受引用保护删除）"},
    {"name": "voices", "description": "配音音色注册表（版本化，调用方自有数据）"},
    {"name": "system", "description": "能力门/运行时/队列/健康"},
]

# (method, path) -> (summary_zh, summary_en, description-or-None)
OVERLAY: dict[tuple[str, str], tuple[str, str, str | None]] = {
    ("post", "/v1/jobs"): (
        "提交单个任务", "Submit a job",
        "201 返回 job；同 `client_ref` 重放返回 200 + `idempotent_replay:true`。"
        "队列深度超限 429；provider/workflow 未过能力门 400/422。"),
    ("post", "/v1/jobs/batch"): (
        "提交批量任务", "Submit a batch",
        "`shared` 提供 provider/workflow/priority/client_ref 与 inputs/generation 的"
        "共享默认，`jobs[]` 逐条覆盖；batch 级 client_ref 同样幂等重放。"),
    ("get", "/v1/jobs"): (
        "列出任务", "List jobs",
        "过滤：status/runtime_profile/batch_id/provider/client_ref；limit≤500。"
        "queued 项带队列位置。"),
    ("get", "/v1/jobs/{job_id}"): (
        "查询单个任务", "Get a job",
        "公开字段：id/provider/workflow/status/runtime_profile/inputs 回执/generation/"
        "outputs[]（含 meta：sha256、duration_s、device 等）/client_ref/时间戳。"),
    ("delete", "/v1/jobs/{job_id}"): (
        "取消任务", "Cancel a job",
        "queued 立即 cancelled；running 置持久化 cancel_requested，由调度器在检查点"
        "落定（返回 cancelling）。已是终态 409。"),
    ("get", "/v1/jobs/{job_id}/outputs"): ("产物列表", "List outputs", None),
    ("get", "/v1/jobs/{job_id}/output"): (
        "下载产物", "Download an output file",
        "按 role/kind 选择文件（配音 WAV、视频 mp4 等）。"),
    ("get", "/v1/jobs/{job_id}/events"): (
        "SSE 事件流", "SSE event stream",
        "`text/event-stream`：任务状态变更逐条推送，终态后服务端关闭。"),
    ("get", "/v1/batches/{batch_id}"): (
        "查询批次", "Get a batch",
        "聚合进度 + 子任务列表。"),
    ("delete", "/v1/batches/{batch_id}"): (
        "取消批次", "Cancel a batch",
        "对全部非终态子任务执行与单任务取消相同的语义。"),
    ("post", "/v1/assets"): (
        "上传素材（raw body）", "Upload an asset (raw body)",
        "内容寻址（sha256 去重）。kind 由 `?kind=` 或 Content-Type/x-filename 推断，"
        "无法推断 400。"),
    ("post", "/v1/assets/upload"): (
        "上传素材（multipart）", "Upload an asset (multipart)",
        "同 POST /v1/assets，form 字段 `file`。"),
    ("get", "/v1/assets/{asset_id}"): ("素材元数据", "Asset metadata", None),
    ("delete", "/v1/assets/{asset_id}"): (
        "删除素材", "Delete an asset",
        "被任何 job/batch 引用时拒绝（409，引用保护）。"),
    ("get", "/v1/voices"): (
        "音色列表", "List voices",
        "调用方自有数据：注册表内容不属于公开文档；含 kind/license/role/version。"),
    ("post", "/v1/voices"): (
        "注册音色", "Register a voice",
        "重复 voice_id 默认 409；`replace:true` 覆盖且旧版进 history（version+1）。"
        "素材必须是已上传托管音频 + 逐字 transcript（zero-shot 参考 10-15 s）。"),
    ("get", "/v1/voices/{voice_id}"): ("音色详情", "Voice detail", None),
    ("delete", "/v1/voices/{voice_id}"): ("删除音色", "Delete a voice", None),
    ("get", "/v1/voices/{voice_id}/preview"): (
        "试听 WAV", "Preview WAV",
        "确定性短样本（canonical 路径；/sample 为兼容别名）。"),
    ("get", "/v1/voices/{voice_id}/sample"): (
        "试听 WAV（兼容别名）", "Preview WAV (compat alias)", None),
    ("get", "/v1/capabilities"): (
        "能力门", "Capability gates",
        "available = 模板已实现 + runtime 启用 + 权重在盘 + 已在实机验证。"
        "附 cosyvoice 模型信息块。"),
    ("get", "/v1/runtime"): (
        "运行时观测", "Runtime observability",
        "双通道（GPU lane + CPU lane）状态、PID、RSS、空闲卸载计时。"),
    ("post", "/v1/runtime/keepalive"): (
        "保持驻留", "Keep runtime warm",
        "阻止指定 profile 的空闲卸载窗口。"),
    ("post", "/v1/runtime/unload"): (
        "手动卸载", "Unload runtime now",
        "立即释放 runtime（无在途任务时）。"),
    ("get", "/v1/queue"): (
        "队列快照", "Queue snapshot",
        "active_profile/state + `cpu_*` 兜底通道观测 + 各 profile 排队深度。"),
    ("get", "/v1/health"): (
        "健康检查", "Liveness", "liveness + 调度器/状态机概况。公开 GET。"),
    ("get", "/v1/openapi.json"): (
        "本契约文档", "This contract",
        "OpenAPI 3.1 JSON（公开面，已剔除 internal 端点），供文档前端动态渲染。"),
}


def _tag_of(path: str) -> str:
    seg = path.strip("/").split("/")
    if len(seg) < 2:
        return "system"
    if seg[1].startswith("jobs"):
        # /v1/jobs/batch belongs to batches
        return "batches" if seg[1] == "jobs" and len(seg) > 2 and seg[2] == "batch" else "jobs"
    if seg[1].startswith("batches"):
        return "batches"
    if seg[1].startswith("assets"):
        return "assets"
    if seg[1].startswith("voices"):
        return "voices"
    return "system"


def curate(spec: dict) -> dict:
    """In-place: internal strip + tags/summary/description/security/scheme."""
    paths: dict = spec.get("paths", {})
    for p in [k for k in paths if k.startswith("/v1/internal")]:
        del paths[p]
    for p, item in paths.items():
        for method, op in item.items():
            if method not in ("get", "post", "delete", "put", "patch"):
                continue
            op["tags"] = [_tag_of(p)]
            summary, summary_en, desc = OVERLAY.get((method, p), ("", "", None))
            if summary:
                op["summary"] = summary   # curated text wins over the generated "Create Job" form
            if summary_en:
                op["x-summary-en"] = summary_en  # bilingual renderer in doc frontends
            if desc:
                op["description"] = desc
            if method != "get":
                op["security"] = [{"BearerAuth": []}]
    spec["tags"] = TAGS
    comps = spec.setdefault("components", {})
    comps.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http", "scheme": "bearer",
        "description": "JAV_API_TOKEN（操作员在部署侧下发；GET 查询面不要求）"}
    info = spec.setdefault("info", {})
    info.setdefault("description", INFO_DESCRIPTION)
    info.setdefault("contact", {"name": "JAV", "url": "https://tools.jeefy.top/docs"})
    spec.setdefault("servers", [{"url": "/", "description": "same origin"}])
    return spec


def cached_spec(app) -> dict:
    """Generate once per process; FastAPI already caches app.openapi_schema,
    but curate() must run exactly once (it mutates in place)."""
    if getattr(app, "_jav_openapi_spec", None) is None:
        app._jav_openapi_spec = curate(app.openapi())
    return app._jav_openapi_spec
