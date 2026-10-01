# JAV 统一 API

服务：`127.0.0.1:8765`（经 `ZIT-tunnel` 暴露为远端 `3001`）。
所有任务/资产响应使用 `id` 字段。输入媒体一律走 asset 上传，任务体只引用 `asset_id`。

## 鉴权（隧道暴露前必读）

- 设置环境变量 `JAV_API_TOKEN`（systemd drop-in）后，**所有非 GET 端点**要求
  `Authorization: Bearer <token>`，否则 401；GET 与 `/v1/health` 保持开放
  （`/v1/health` 响应中 `auth` 字段指示当前模式）。未设置 = 本地信任模式（仅适合
  纯回环使用）。SDK 已内置 `token=` 参数。
- `/v1/internal/*`（worker 回调）**永不**走 API token：要求
  `X-JAV-Callback: <JAV_CALLBACK_SECRET>`（默认每进程随机，经 env 注入 worker）。
  任何客户端（含隧道对端）伪造回调都会被 403 拒绝；回调携带的文件路径还受
  托管目录白名单（pending / outputs / ComfyUI output）二次约束。

### 当前状态（2026-10-01 起：Bearer 已启用）

- Token 存放于 **`~/.config/jav/env`**（chmod 600，仓库外），经 systemd drop-in
  `~/.config/systemd/user/JAV.service.d/10-auth.conf` 的 `EnvironmentFile` 注入。
  **token 值不得写入任何 git 跟踪的文件（含本文档）**。
- 本机调用：`curl -H "Authorization: Bearer $(grep -m1 '^JAV_API_TOKEN=' ~/.config/jav/env | cut -d= -f2)" …`；
  SDK 传 `token=` 或设环境变量 `JAV_API_TOKEN`。
- 轮换：编辑 `~/.config/jav/env` → `systemctl --user restart JAV.service`
  （callback 秘密随进程重新随机，无需手动同步）。

## 概念

- **provider**：`zit` | `ltx25` | `mh3`
- **workflow**：provider 内的能力名（见 `/v1/capabilities`）
- **runtime_profile**：真正占用 GPU 的资源档位（`zit` / `ltx25` / `mh3.fl2va` / `mh3.ref2va`），
  互斥调度，同一时刻仅一个常驻

| provider.workflow | runtime_profile |
|---|---|
| zit.t2i / zit.i2i / zit.inpaint | zit |
| ltx25.t2v / i2v / flf2v / a2v / union_control / motion_control / inpaint / outpaint / ic_lora / bbox_control | ltx25 |
| mh3.t2v / i2v / fl2v | mh3.fl2va |
| mh3.ref2v / fun_control / multiframe | mh3.ref2va |

已验证扩展语义：
- `ltx25`：`generation.mode = "fast"`（默认，单段 distilled 8 步）| `"high"`
  （Two-Stage：低段生成 → latent x2 上采样 → 3 步 re-sampler，输出约 2× 分辨率；t2v/i2v/flf2v 支持（flf2v stage2 在 x2 上采样前 CropGuides 剥离关键帧 token，随后 re-anchor））
- **ltx25 IC-LoRA 控制族**（`union_control` / `motion_control` / `inpaint` /
  `outpaint` / `ic_lora`）：单段 distilled + IC-LoRA guide，**时长与画幅跟随源
  视频**（不接受 duration/width/height；`generation.strength` 0..1 控制 guide
  强度，默认 1.0；`inputs.shorter_size` 128..768 且 32 倍数，默认 512）：
  - `union_control`：`inputs.control_video`（必需）→ Canny 边缘引导
    （`canny_low`/`canny_high` 0.01..0.99 默认 0.4/0.8），union-control IC-LoRA
  - `motion_control`：`inputs.source_video`（必需）→ slow-motion IC-LoRA 变换
  - `inpaint`：`source_video` + `mask_image`（白色=重绘，整段生效）+
    `dilate_radius` 0..32（默认 5），clean-plate IC-LoRA
  - `outpaint`：`source_video` + `canvas_width`/`canvas_height`（必需，32 倍数
    256..2048，向外扩展画布）
  - `ic_lora`：`mode="reference"`（默认）：`reference_sheet`（参考拼版图，
    ingredients LoRA 图生视频）；`mode="v2v"`：`source_video` +
    `lora ∈ {cinemagraph, clean_plate, ingredients, slow_motion}` 视频编辑
  - seed 缺省 -1 = 服务端随机（响应中不回显有效 seed）
- `mh3`：`inputs.turbo = true`（官方 turbo LoRA 路径，fl2v 系 8 步 / ref2v 4 步，走已下载的 turbo LoRA）
- `mh3.fun_control`：`inputs.control_video`（必需，video asset）+ `inputs.control_strength`
  （默认 1.0）；Fun ControlNet-Union patch 支持 canny/depth/pose/hed 控制视频，
  ref2va base，默认 20 步
- `mh3.multiframe`：`inputs.reference_images`（1-9）+ `inputs.keyframes`
  （`[{image|video, time}]`，≤8；`video` 走 LoadVideo clip 锚定 = 官方 continuation 语义，
  `time` 秒 → `MiniMaxH3AddGuide` 按 `round(time*fps)` clamp 到帧）；
  ref2va base，链式把每张关键帧锚定到 latent 对应帧，默认 20 步

## 任务

### POST /v1/jobs
```json
{
  "provider": "zit",
  "workflow": "t2i",
  "inputs":  {"prompt": "a red cube", "negative_prompt": "",
              "image": "<asset_id>", "mask": "<asset_id>"},
  "generation": {"width": 1024, "height": 1024, "steps": 9,
                 "guidance": 0.0, "strength": 0.8, "seed": -1},
  "priority": 0,
  "client_ref": "optional"
}
```
- 201 → `{id, status:"queued", runtime_profile, queue_position, ...}`
- 415：workflow 不可用（`{"detail":"... unavailable: <reason>"}`）
- 400：参数/资产校验失败；429：队列积压超限（500）
- `seed >= 0` 时启用缓存命中：完全相同参数的已完成任务直接返回 `status:"completed"`
- `inputs.image`：i2i 必填；`inputs.mask`：inpaint 必填（白色=重绘，黑色=保留）

### POST /v1/jobs/batch
```json
{
  "shared": {"provider":"zit","workflow":"t2i",
             "generation": {"width":512,"height":512,"steps":4}},
  "jobs":   [{"inputs": {"prompt": "one"}},
             {"inputs": {"prompt": "two"}, "generation": {"seed": 7}}],
  "client_ref": "storyboard-01"
}
```
- 整批原子校验：任一 job 非法 → 422 `{"detail":{"invalid_jobs":{index: reason}}}`，零入队
- 201 → `{batch_id, jobs:[id...], queue_positions:{id: n}}`
- 上限 64 jobs/批；同 profile 任务由调度器聚组连跑（省 runtime 切换）

### GET /v1/jobs
过滤参数：`status`（逗号分隔多值）、`runtime_profile`、`batch_id`、`provider`、`limit`、`offset`
返回 `{total, jobs:[<public job>]}`

### GET /v1/jobs/{id}
```json
{ "id":"job_...", "provider":"zit", "workflow":"t2i", "runtime_profile":"zit",
  "status":"running", "batch_id":null, "client_ref":null,
  "created_at":"...", "started_at":"...", "finished_at":null,
  "error":null, "error_type":null, "retry_count":0, "queue_position":3,
  "outputs":[{"id":"out_...","kind":"image","asset_id":"asset_...",
              "url":"/v1/jobs/job_.../output?asset_id=out_..."}] }
```
状态机：`queued → starting_runtime → running → completed | failed | cancelled`
（崩溃重启后在途任务自动回到 `queued`）

### DELETE /v1/jobs/{id}
- 排队中 → `{"status":"cancelled"}` 立即生效
- 启动中/运行中 → `{"status":"cancelling"}`：取消意图**持久化**（`cancel_requested`），
  调度器在下一个检查点（提交前 / 结果落地前 / 失败重试前）落定为
  `cancelled` 并丢弃产物——不会出现“已接受取消但任务照常完成”。
  ComfyUI 侧同时 best-effort `/interrupt`
- 已在终态 → 409
- `GET /v1/jobs`：`limit` 钳制在 1..500，`offset` ≥0（越界参数自动修正而非透传 SQL）

### GET /v1/jobs/{id}/outputs · GET /v1/jobs/{id}/output?asset_id=
列表 / 直接下载产物文件（content-addressed，可长期缓存）。

### GET /v1/jobs/{id}/events （SSE）
`event: status` + `data: {"status": "..."}`，终态后关闭；5s keepalive。

## Batch
- `GET /v1/batches/{id}` → `{id, counts:{status:n}, jobs:[id...]}`
- `DELETE /v1/batches/{id}` → 批量取消未完成项

## 资产
### POST /v1/assets?kind=image|video|audio
- **raw-body 上传**：请求体即文件字节；`x-filename` 头可选（用于推断 kind/扩展名）
- 或 `POST /v1/assets/upload?kind=` multipart（字段名 `file`）
- 201 → `{"id":"asset_...", "type":"image", "sha256":"...", "size":n}`
- SHA-256 去重：相同字节返回同一 `id`；上限 2GiB/文件（**流式**落盘 +
  `Content-Length` 预检 + 传输中超限即 413，不整读进内存）

### GET /v1/assets/{id} · DELETE /v1/assets/{id}
- DELETE 有引用保护：被非终态任务或已记录输出引用的资产返回 **409**，
  不会静默删除在途/可下载文件

## 系统
- `GET /v1/capabilities` — 每个 workflow 的 `available` 由
  权重在盘 × 模板实现 × 硬件 smoke 标志动态计算；`reason` 说明不可用原因
- `GET /v1/queue` — `{active_profile, state, streak, queued_by_profile,
  queued_total, admission_backoff}`
- `GET /v1/runtime` — 当前 runtime 进程/RAM/swap/VRAM 指标 + 最近切换事件
- `POST /v1/runtime/keepalive` — 刷新空闲卸载计时器，让当前 runtime 继续驻留
  （可选 `?ttl_s=` 临时延长本轮窗口，上限 7200s）；返回
  `{kept_alive, state, idle_unload_in_s}`。无活跃 runtime 时返回
  `{kept_alive: null}`，下个任务照常冷启动
- `POST /v1/runtime/unload` — 立即释放 active runtime（不等空闲超时），
  返回卸载前 profile 与队列快照
- `GET /v1/health` — liveness

## 调度语义
- 互斥：跨 profile 切换 = 软清理(`/free`) → SIGTERM → 显存释放确认 → 起新进程
- affinity：同 profile 连跑 ≤3 个任务；其他 profile 等待 >10min 触发切换；
  绝不在任务中途切换 runtime
- 准入（OOM 防护）：启动 runtime 前要求实时
  `MemAvailable+SwapFree ≥ RAM 预算 + 4G 地板` 且 `VRAM free ≥ 显存预算`，
  不满足则任务保持 queued 指数退避（15s→4min）。VRAM 探针失败（nvidia-smi
  异常/驱动挂死）时**拒绝准入**而非放行（fail-closed）；仅
  `JAV_VRAM_GATE=off` 显式绕过
- 空闲卸载：runtime 无任务超过 `idle_unload_s`（zit 默认 **300s**，可用
  `JAV_ZIT_IDLE_UNLOAD_S` 调整）后自动 STOPPED 释放约 20G RAM + 全部显存；
  不等的话可随时 `POST /v1/runtime/unload` 手动释放。空闲后的首个任务会
  多付约 60–90s 权重加载时间（页缓存预热后可接受）。
- 失败重试：runtime 崩溃/超时/OOM 自动重入队一次；确定性参数错误不重试

## 内部端点（仅 worker 回调，须 `X-JAV-Callback` 秘密头）
- `POST /v1/internal/task_complete` — worker 任务结果回调（路径限
  pending/outputs/ComfyUI output 三个托管目录）
- `POST /v1/internal/pipeline_status` — worker 加载/卸载状态回调

## 部署可靠性
- systemd：`StartLimitIntervalSec=600` / `StartLimitBurst=3`，持续失败会
  park 服务（不再无限 5s 重启循环；循环会反复重跑孤儿进程清扫）
- 孤儿清扫：runtime.state 在**子进程 Popen 瞬间**记录 pid + starttime；
  清扫必须同时匹配记录的 profile 脚本全路径与进程启动时刻，PID 复用不会
  误杀无关进程
- `tools/first_validation.py` 与在线服务互斥：检测到 `:8765` 存活即拒绝
  运行，并用 `data/.first_validation.lock` 防双实例（避免双 runtime OOM）

## 兼容说明
旧 ZIT-service `/v1/tasks` 系列**未保留**（用户决策 2026-09-29：JAV 完成后
jeefy-tools 直接适配 `/v1/jobs`）。
