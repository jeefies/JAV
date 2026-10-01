# JAV API 接口规范

版本 v0.1 · Base URL 由服务方提供 · 内容类型 `application/json`（上传除外）

**鉴权规则**：`Authorization: Bearer <token>`。除 `GET /v1/health` 外的**写操作**（POST/DELETE）强制校验，缺失或错误返回 `401`；`GET` 查询端点开放。Token 由服务方单独发放。

**通用约定**：
- 所有对象以 `id`（字符串，全局唯一）标识；时间戳为 UTC ISO-8601。
- 输入媒体（图/视频/音频）必须先经 `/v1/assets` 上传，任务体只引用 `asset_id`。
- 错误响应统一为 `{"detail": "<原因>"}`；批量校验为 `{"detail": {"invalid_jobs": {"<下标>": "<原因>"}}}`。

---

## 1. 任务 Jobs

### 1.1 `POST /v1/jobs` — 提交任务

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `provider` | string | 是 | `zit` \| `ltx25` \| `mh3` |
| `workflow` | string | 是 | 见 §1.2 workflow 参数表 |
| `inputs` | object | 是 | workflow 专属输入（prompt、asset_id 引用等） |
| `generation` | object | 否 | `width`/`height`/`steps`/`guidance`/`strength`/`seed`/`duration`/`fps` 等，按 workflow 取用 |
| `priority` | int | 否 | 默认 0，越大越优先 |
| `client_ref` | string | 否 | 调用方自定义标签，原样回显 |

响应与逻辑：
- `201` → 任务对象，`status="queued"`，附 `runtime_profile` 与 `queue_position`
- `400` 参数/资产校验失败（资产不存在、尺寸非法等）
- `401` 鉴权失败
- `415` workflow 当前不可用（`detail` 含原因）
- `429` 队列积压达上限（默认 500）
- **幂等缓存**：`generation.seed >= 0` 时请求是确定性的；若服务端存在**完全相同参数**（prompt/尺寸/步数/seed/引用资产一致）的已完成任务，提交将立即创建一个新任务并直接置为 `completed`，复用既有产物（新 `id`，不返回原任务 id）
- `generation.seed = -1`（默认）由服务端随机，有效 seed 不回显

任务对象：

```json
{ "id": "job_…", "provider": "…", "workflow": "…", "runtime_profile": "…",
  "status": "queued|starting_runtime|running|completed|failed|cancelled",
  "batch_id": null, "client_ref": null,
  "created_at": "…", "started_at": "…", "finished_at": null,
  "error": null, "error_type": null, "retry_count": 0, "queue_position": 3,
  "outputs": [ { "id": "out_…", "kind": "image|video|audio", "asset_id": "asset_…",
                 "url": "/v1/jobs/job_…/output?asset_id=out_…" } ] }
```

状态机：`queued → starting_runtime → running → completed | failed | cancelled`；服务端故障重启后在途任务自动回到 `queued`。

### 1.2 Workflow 与 `inputs` / `generation` 规范

| provider.workflow | 必需 inputs | 可选 inputs | generation 关键项 |
|---|---|---|---|
| `zit.t2i` | `prompt` | `negative_prompt` | `width`/`height`/`steps`/`guidance`/`seed` |
| `zit.i2i` | `prompt`, `image` | `negative_prompt` | 同上 + `strength`（0..1 重绘幅度） |
| `zit.inpaint` | `prompt`, `image`, `mask` | `negative_prompt` | 同上；mask 白色=重绘、黑色=保留 |
| `ltx25.t2v` | `prompt` | `negative_prompt` | `width`/`height`/`duration`(s)/`fps`/`seed`/`mode` |
| `ltx25.i2v` | `prompt`, `first_image`（别名 `image`） | `negative_prompt` | 同上 |
| `ltx25.flf2v` | `prompt`, `first_image`（别名 `first_frame`）, `last_image`（别名 `last_frame`） | — | 同上 |
| `ltx25.a2v` | `prompt`, `audio` | `negative_prompt` | `fps`/`seed`/`mode`；输出时长 = `generation.duration`（默认 5s），音频裁剪到此长度（不自动跟随音频原始时长） |
| `ltx25.union_control` | `prompt`, `control_video` | `canny_low`/`canny_high`（0.01..0.99，默认 0.4/0.8） | `strength`、`shorter_size`；**不接受 duration/width/height** |
| `ltx25.motion_control` | `prompt`, `source_video` | — | 同上 |
| `ltx25.inpaint` | `source_video`, `mask_image` | `dilate_radius`（0..32，默认 5） | 同上；mask 白色=重绘，整段生效 |
| `ltx25.outpaint` | `source_video`, `canvas_width`, `canvas_height`（32 倍数，256..2048） | — | 同上；画布向外扩展 |
| `ltx25.ic_lora`（默认 reference 模式） | `reference_sheet`（参考拼版图） | `prompt` | `strength`、`shorter_size` |
| `ltx25.ic_lora`（`mode:"v2v"`） | `source_video`, `lora`∈`cinemagraph/clean_plate/ingredients/slow_motion` | `prompt` | 同上 |
| `mh3.t2v` / `i2v` / `fl2v` | `prompt`；i2v 加 `image`；fl2v 加 `first_frame`/`last_frame` | `turbo`(bool，8 步快速档) | `width`/`height`/`duration`/`fps`/`steps`/`seed` |
| `mh3.ref2v` | `prompt`, `reference_images`（1–9 张） | `turbo` | 同上 |
| `mh3.fun_control` | `prompt`, `control_video` | `control_strength`（默认 1.0，控制 canny/depth/pose/hed 风格视频） | 同上，默认 20 步 |
| `mh3.multiframe` | `prompt`, `keyframes`（`[{image|video, time}]`，≤8，`time` 秒）, `reference_images`（1–9） | `turbo` | 同上，默认 20 步 |

ltx25 通用扩展：`generation.mode = "fast"`（默认）| `"high"`（两段式高清，输出约 2× 分辨率，耗时更长；仅 t2v/i2v/flf2v）。
IC-LoRA 控制族（union_control/motion_control/inpaint/outpaint/ic_lora）：时长与画幅**跟随源视频/参考图**；`inputs.shorter_size` 128..768 且为 32 的倍数（默认 512）；`generation.strength` 0..1（guide 强度，默认 1.0）。

### 1.3 `POST /v1/jobs/batch` — 批量提交

```json
{ "shared": { "provider": "…", "workflow": "…", "generation": { … } },
  "jobs": [ { "inputs": { … } }, { "inputs": { … }, "generation": { … } } ],
  "client_ref": "…" }
```

- `shared` 提供各任务默认值，`jobs[]` 内字段**浅合并覆盖**（`inputs`/`generation` 同名键以 job 为准）；`client_ref` 优先级：job > shared > 批级
- **整批原子校验**：任一非法（参数/资产/workflow 不可用）→ `422 {"detail":{"invalid_jobs":{"<下标>":"<原因>"}}}`，零入队；全部合法 → `201 {"batch_id","jobs":[id…],"queue_positions":{id:n}}`
- 单批上限 64；空批或超限 → `400`；加入后将超队列上限 → `429`
- 同 `runtime_profile` 任务由调度器聚组连跑

### 1.4 `GET /v1/jobs` — 任务列表

查询参数：`status`（逗号分隔多值）、`provider`、`runtime_profile`、`batch_id`、`limit`（1..500）、`offset`（≥0）。
返回 `{ "total": n, "jobs": [任务对象] }`。

### 1.5 `GET /v1/jobs/{id}` — 任务详情

返回任务对象。`404` 不存在。

### 1.6 `DELETE /v1/jobs/{id}` — 取消任务

| 当前状态 | 响应 | 语义 |
|---|---|---|
| `queued` | `200 {"id","status":"cancelled"}` | 立即生效 |
| `starting_runtime` / `running` | `200 {"id","status":"cancelling"}` | 取消意图持久化，服务端在下一个检查点（提交前/结果落地前/重试前）落定 `cancelled` 并丢弃产物；**不会出现已接受取消但任务照常完成** |
| 终态 | `409` | `detail:"job already <status>"`，无法取消 |

### 1.7 `GET /v1/jobs/{id}/events` — SSE 状态流

`text/event-stream`：连接即推送当前状态 `event: status` + `data: {"status":"…"}`；此后每次状态变更再推一条，到达终态后服务端关闭；5s keepalive 注释帧；30 分钟无状态变化服务端会主动断开（重连即可，连接时重放当前状态）。

### 1.8 产物获取

- `GET /v1/jobs/{id}/outputs` → 产物数组（同任务对象 `outputs` 字段）
- `GET /v1/jobs/{id}/output?asset_id=<out_id>` → 文件字节流（`Content-Type` 按类型；内容寻址、可长期缓存与断点续传由客户端自行处理）
- 任务不存在：`404`；`asset_id` 不属于该任务：`404`

## 2. Batch

- `GET /v1/batches/{id}` → `{ "id", "counts": {"queued":n,"completed":n,…}, "jobs": [id…] }`
- `DELETE /v1/batches/{id}` → 对批内每个任务执行取消（语义同 §1.6），返回 `{ "batch_id", "result": { "<job_id>": "cancelled" | "cancelling" | "<已终态>" | "not_found" } }`；批不存在 `404`

## 3. 资产 Assets

### 3.1 上传

- `POST /v1/assets?kind=image|video|audio` — **raw body**：请求体即文件字节；可选头 `x-filename`（辅助推断 kind/扩展名）
- `POST /v1/assets/upload?kind=` — multipart，字段名 `file`
- `kind` 省略时按 `x-filename` 扩展名或 `Content-Type` MIME 推断；无法推断 → `400 "cannot infer asset kind; pass ?kind="`；空请求体 → `400`
- `201` → `{ "id": "asset_…", "type": "image|video|audio", "sha256": "…", "size": n }`
- **SHA-256 内容去重**：相同字节返回同一 `id`，重复上传不占额外空间
- 上限 2 GiB/文件（`Content-Length` 预检 + 传输中实时超限），超限 `413`

### 3.2 查询 / 删除

- `GET /v1/assets/{id}` → **直接返回文件字节流**（`Content-Type` 按 MIME、带 `Content-Disposition` 文件名）；资产不存在 `404`；记录在但底层文件缺失 `410`
- `DELETE /v1/assets/{id}` → `200 {"id","deleted":true}`；被非终态任务或已记录产物引用时 `409`（`detail` 说明引用数，不会破坏在途任务与产物可下载性）；不存在 `404`

## 4. 系统 System

### 4.1 `GET /v1/health`（无需鉴权）

`{ "status": "healthy", "service": "JAV", "version": "…", "auth": "bearer|disabled", "state": "STOPPED|STARTING|READY|DRAINING|STOPPING", "active_profile": null|"…", "queued": n, "scheduler_running": true }`

### 4.2 `GET /v1/capabilities`

按 provider 分组的实时能力表：

```json
{ "zit": { "workflows": { "t2i": { "available": true, "runtime": "zit" }, … } }, … }
```

`available=false` 时附 `reason`。提交前的可用性以本端点为准（权重、模板、硬件探测动态计算）。

### 4.3 `GET /v1/queue`

`{ "active_profile", "state", "queued_total", "queued_by_profile": {…}, "admission_backoff": {…} }`

### 4.4 `GET /v1/runtime`

当前模型驻留状态与资源指标（RAM/swap/VRAM）+ 最近切换事件列表。

### 4.5 `POST /v1/runtime/keepalive?ttl_s=`

刷新当前驻留模型的空闲卸载计时器（不创建任务）。返回 `{ "kept_alive": true|null, "state": "…", "idle_unload_in_s": n }`。`ttl_s` 可选，钳制到 **1..7200**（`ttl_s<=0` 或超上限都不会反向触发立即卸载）；无活跃模型时 `kept_alive=null`。

### 4.6 `POST /v1/runtime/unload`

立即释放当前驻留模型（不等空闲超时）。返回卸载前的 profile 与队列快照。有任务处于 `starting_runtime`/`running` 时返回 **409**（绝不中途打断在途任务；需先取消对应任务）。

## 5. 调度与可靠性语义（调用方可依赖的行为）

- **互斥驻留**：同一时刻仅一个模型档占用 GPU；跨档切换期间新任务保持 `queued`，任何任务**不会**被中途打断
- **冷启动**：空闲卸载后首个任务需加载权重（约 +60–90s），后续同档任务连续执行
- **亲和连跑**：同档任务最多连跑 3 个即让位；他档等待 >10min 优先切换
- **准入退避**：主机内存/显存不足时任务保持 `queued` 并指数退避（15s→4min），不会因此 `failed`
- **失败重试**：`error_type` 属于运行时类（`runtime_crash` / `timeout` / `start_error` / `oom_error` / `internal_error` / `ingest_error`）时自动重入队一次（`retry_count` 递增）；参数/构图类（`compile_error`、普通 `generation_error`）直接 `failed`，不重试。ComfyUI 侧超时同样按 `timeout` 落定（触发中断+运行时重建后重试）。`error` 为人读信息（截断至 500 字符）

## 6. 状态码汇总

| 码 | 场景 |
|---|---|
| `200` | 查询/取消/回调类成功 |
| `201` | 任务、批量、资产创建成功 |
| `400` | 参数/资产校验失败（含无法推断资产类型、空上传体） |
| `401` | 缺失或错误的 Bearer token |
| `404` | 资源不存在 |
| `409` | 状态冲突（已终态取消、运行中 unload、资产被引用） |
| `410` | 资产记录存在但底层文件缺失 |
| `413` | 上传超 2 GiB |
| `415` | workflow 不可用 |
| `422` | 批量校验失败（含 `invalid_jobs` 映射） |
| `429` | 队列积压超 500 |
