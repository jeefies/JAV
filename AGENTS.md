# JAV — AI 编码代理须知

本目录是 JAV 统一生成服务。自然语言注释以中文为主的约定沿用 AV workspace。

## 必读上下文
- 架构设计（决策依据，先读再改）：`/mnt/data/AV/JAV-DESIGN.md`
- API 契约：`API.md`；架构原始思路：`/mnt/data/AV/struct.md`

## 关键不变式（改代码前确认）
1. **同一时刻最多一个 runtime 进程占用 GPU 大权重**。RuntimeSupervisor 的
   asyncio.Lock 是唯一入口；不要绕过它 spawn 任何推理进程。
   `tools/first_validation.py` 与在线服务互斥（health 探测 + lockfile），不要绕过。
2. **绝不中途杀死 running 任务来切换 runtime**（用户显式 cancel 除外）。
   取消是持久化 `cancel_requested` 标志 + 调度器检查点落定，勿改回内存态。
3. **RAM 准入**：启动任何 backend 前必须过 `check_admission`
   （MemAvailable+SwapFree 与 VRAM free 双检查）。本机 30G RAM 与 unichess
   训练共存，admission 是 OOM 防线，禁止调低 `JAV_MEM_FLOOR_MB`；
   VRAM 探针失败是 **fail-closed**（`JAV_VRAM_GATE=off` 仅限排障）。
4. **权重只能在 /mnt/data/AV**（用户红线）。新增权重放
   `ComfyUI/models/<分类>/`；diffusers 缓存固定 `HF_HOME=/mnt/data/AV/JAV/data/hf_home`。
   验收前跑 `python3 tools/audit_weights.py`。
5. 公开 API 字段只用 `id`；ComfyUI 细节（prompt_id、节点图）不得泄漏到 API 层，
   workflow 变更只改 `jav/workflows/*` 模板与 manifest。
6. jeefy-tools 兼容层**故意不存在**；外部适配是独立后置任务。
7. **安全边界**：`/v1/internal/*` 只认 `X-JAV-Callback` 秘密头，且回调路径必须
   落在托管目录白名单内（防伪造回调读取任意文件）。**Bearer 已启用（2026-10-01）**：
   `JAV_API_TOKEN` 在 `~/.config/jav/env`（600，仓库外）经 drop-in 注入，非 GET 端点
   强制校验；**token 值严禁写入 git 跟踪文件**。资产删除受引用保护。

## 开发
- 解释器：统一 conda env `comfyui`（服务、ZIT worker、ComfyUI backend 同一
  `/home/jeefy/miniconda3/envs/comfyui/bin/python3`；旧 image env 已删除）。
- 测试：`python3 -m pytest tests -q`（FakeBackend 驱动，秒级，不占 GPU/内存）；
  GPU 级验收见 README「当前状态」与 `tools/comfy_smoke.py`。
- 修改 profile 预算：`jav/config.py`（默认值）或 `config/profiles.yaml`（覆盖）。
- 服务：`systemctl --user {start|restart|status} JAV.service`；
  unit 在 `deploy/JAV.service`，编辑后 `cp` 到 `~/.config/systemd/user/` 并
  `daemon-reload`。cgroup 内存上限（High31/Max34/SwapMax20G）与 profile 预算需一起权衡。

## 已知坑
- ZIT worker **实测峰值 ~29.2G RSS**（i2i/inpaint 的 fp32→bf16 derived cast 瞬
  间；常驻 bf16 ~20G + fp32 副本）。cgroup `MemoryMax` 是 RAM+swap 总和，若低于
  该峰值会**永久 thrash 且冻死事件循环**（2026-09-30 实测 45min 卡死，28G cap）；
  profile `ram_budget_mb=30720` 与 High31/Max34 需同步权衡。超时重试路径必须彻底
  回收旧 worker（scheduler 已在 job_timeout 时 `sup.shutdown("job_timeout")`，
  勿改回复用）。
- SIGKILL CUDA 进程会留下驱动级“幽灵”VRAM 占用；vram 准入会让后续任务排队退避
  直至释放，属预期行为，不要为此加重试逻辑。
- ZIT derived pipeline（i2i/inpaint）首次使用需 fp32→bf16 全量 cast，首任务比
  t2i 慢数分钟，属正常。
- ComfyUI **v3 io 节点的 DynamicCombo**（如 ResizeImageMaskNode 的 resize_type）
  在 API prompt 里的合法形态是 `"resize_type": "<选项字符串>"` +
  `"resize_type.<子字段>": value` 点号键；写成嵌套 dict 会通过验证但在 execute
  时丢参（`comfy_provider._set_deep` 已自动展开，模板初值保持点号形态）。
- ComfyUI RandomNoise 等采样节点要求 seed ≥ 0：`seed=-1`（随机）由
  `models.eff_seed` 在编译期随机化，勿把 -1 直接注入图。
