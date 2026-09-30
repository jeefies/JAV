# JAV — AI 编码代理须知

本目录是 JAV 统一生成服务。自然语言注释以中文为主的约定沿用 AV workspace。

## 必读上下文
- 架构设计（决策依据，先读再改）：`/mnt/data/AV/JAV-DESIGN.md`
- API 契约：`API.md`；架构原始思路：`/mnt/data/AV/struct.md`

## 关键不变式（改代码前确认）
1. **同一时刻最多一个 runtime 进程占用 GPU 大权重**。RuntimeSupervisor 的
   asyncio.Lock 是唯一入口；不要绕过它 spawn 任何推理进程。
2. **绝不中途杀死 running 任务来切换 runtime**（用户显式 cancel 除外）。
3. **RAM 准入**：启动任何 backend 前必须过 `check_admission`
   （MemAvailable+SwapFree 与 VRAM free 双检查）。本机 30G RAM 与 unichess
   训练共存，admission 是 OOM 防线，禁止调低 `JAV_MEM_FLOOR_MB`。
4. **权重只能在 /mnt/data/AV**（用户红线）。新增权重放
   `ComfyUI/models/<分类>/`；diffusers 缓存固定 `HF_HOME=/mnt/data/AV/JAV/data/hf_home`。
   验收前跑 `python3 tools/audit_weights.py`。
5. 公开 API 字段只用 `id`；ComfyUI 细节（prompt_id、节点图）不得泄漏到 API 层，
   workflow 变更只改 `jav/workflows/*` 模板与 manifest。
6. jeefy-tools 兼容层**故意不存在**；外部适配是独立后置任务。

## 开发
- 解释器：conda env `image`（服务与 worker）；ComfyUI backend 用 env `comfyui`。
- 测试：`python3 -m pytest tests -q`（FakeBackend 驱动，秒级，不占 GPU/内存）；
  GPU 级验收见 README「当前状态」与 `tools/comfy_smoke.py`。
- 修改 profile 预算：`jav/config.py`（默认值）或 `config/profiles.yaml`（覆盖）。
- 服务：`systemctl --user {start|restart|status} JAV.service`；
  unit 在 `deploy/JAV.service`，编辑后 `cp` 到 `~/.config/systemd/user/` 并
  `daemon-reload`。cgroup 内存上限（High26/Max28/SwapMax16G）与 profile 预算需一起权衡。

## 已知坑
- ZIT worker 工作集 ~25G（RAM 19-21G + swap），MemoryHigh 若低于工作集会 thrash
  到分钟级生成；超时重试路径必须彻底回收旧 worker（scheduler 已在 job_timeout
  时 `sup.shutdown("job_timeout")`，勿改回复用）。
- SIGKILL CUDA 进程会留下驱动级“幽灵”VRAM 占用；vram 准入会让后续任务排队退避
  直至释放，属预期行为，不要为此加重试逻辑。
- ZIT derived pipeline（i2i/inpaint）首次使用需 fp32→bf16 全量 cast，首任务比
  t2i 慢数分钟，属正常。
- 旧 ZIT-service 目录保留作迁移参考；`image_service.py` 的 Flask 实现已停用，
  不要向其添加功能。
