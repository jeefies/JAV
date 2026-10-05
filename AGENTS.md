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
  **例外：cosyvoice worker 用独立 venv `/mnt/data/AV/venvs/cosyvoice`**
  （python3.12 + torch 2.7.1+cu128）——见已知坑「CosyVoice 依赖隔离」。
- 测试：`python3 -m pytest tests -q`（FakeBackend 驱动，秒级，不占 GPU/内存）；
  GPU 级验收见 README「当前状态」与 `tools/comfy_smoke.py`。
- 修改 profile 预算：`jav/config.py`（默认值）或 `config/profiles.yaml`（覆盖）。
  **profiles.yaml 是字段级合并**（2026-10-01 审计修正）：override 只替换它写出的
  key，其余继承内置默认；未知 key 直接报错。旧版是整条替换，只写预算会把
  `script`/`python_bin` 清成空串导致该 profile 全量停摆，勿改回。
- 服务：`systemctl --user {start|restart|status} JAV.service`；
  unit 在 `deploy/JAV.service`，编辑后 `cp` 到 `~/.config/systemd/user/` 并
  `daemon-reload`。cgroup 内存上限（High31/Max34/SwapMax20G）与 profile 预算需一起权衡。
- 停机预算：优雅停机最坏路径 = 调度器等待在途(90) + comfy `/free`(10) + terminate(15)
  + supervisor VRAM 确认(30) ≈ 145s，必须 < `TimeoutStopSec`（现为 **180**）。改任一
  数字都要同步改另一处，否则 systemd 会 SIGKILL CUDA 子进程留下“幽灵显存”（见已知坑）。

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
- **VRAM 与 unichess 同卡共存（2026-10-01 OOM 复盘）**：unichess 的 `app.py`
  （~916MiB 常驻）与 `Kit selfplay`（~1.16GiB，会增长）都占用 GPU 0，不在 JAV
  互斥锁管辖内。ZIT worker 真实 VRAM 峰值 ~13.4GiB（budget 12800 偏乐观）。
  防线：`VRAM_FLOOR_MB=1536`（自 256 上调，覆盖外部共存的 TOCTOU 增长窗口）+
  `expandable_segments`（**必须在 `import torch` 之前设置**，zit_worker 顶部；
  main() 里 setdefault 无效，曾因此白烧 640MiB 碎片）。selfplay 活跃期 zit
  准入会持续退避排队（预期行为，勿"优化"成放行）；显存侧与 RAM 侧同理：
  预算=实测峰值，地板=外部增长余量。
- ComfyUI **v3 io 节点的 DynamicCombo**（如 ResizeImageMaskNode 的 resize_type）
  在 API prompt 里的合法形态是 `"resize_type": "<选项字符串>"` +
  `"resize_type.<子字段>": value` 点号键；写成嵌套 dict 会通过验证但在 execute
  时丢参（`comfy_provider._set_deep` 已自动展开，模板初值保持点号形态）。
- ComfyUI RandomNoise 等采样节点要求 seed ≥ 0：`seed=-1`（随机）由
  `models.eff_seed` 在编译期随机化，勿把 -1 直接注入图。
- 音频 latent 的 `frame_rate` 必须由 manifest `audio_fps` 绑定 `gen["fps"]`
  （t2v/i2v/flf2v/bbox 及 upscale 变体，2026-10-01 已统一）；模板字面量只是
  初值，勿改回硬编码（flf2v 曾冻结在 24、其余 25，造成 A/V 映射跨 workflow 漂移）。
- GPU 子进程环境必须剥离 `JAV_API_TOKEN`（第三方 ComfyUI 节点可执行任意代码；
  ComfyUI 还额外剥 `JAV_CALLBACK_SECRET`）。ZIT worker 保留 callback 三件套但
  同样不见 token。新增 backend spawn 时照抄该 scrub。
- **CosyVoice 依赖隔离（2026-10-04 t2a 上线）**：本机是 RTX 5070 Ti（**sm_120**），
  上游 requirements 钉的 torch 2.3.1+cu121 最高只到 sm_90，装上只会 CUDA 报错——
  venv 内必须 `torch==2.7.1+cu128`（torchaudio 同步）。conda 默认 channels 走
  tuna 镜像已 403，建环境用 `uv venv --seed` + pip 走 `mirrors.aliyun.com`；
  openai-whisper 需 `--no-build-isolation`（其 setup.py 要 pkg_resources，build
  隔离环境的 setuptools≥81 已删除）。推理链真实需要：whisper/onnxruntime/
  hyperpyyaml/wetext/**lightning**（matcha.utils）/**gdown+wget**（matcha utils）/
  **x-transformers**（CV3 flow DiT）/**pyarrow+pyworld**（cosyvoice3.yaml 引用的
  dataset.processor）。缺任何一个表现为 AutoModel 加载期 pydoc locate ImportError。
- **instruct2 与注册 spk 互斥陷阱**：`frontend_instruct2` 传了 `zero_shot_spk_id`
  会直接加载 spk2info 并**丢弃 instruct 文本**（wants.md 的"逐句表演指令"会静默
  失效）。cosyvoice_worker 的实现是正确的：有 instruction → 逐句传 prompt_wav 的
  instruct2（无 spk 捷径）；无 instruction → add_zero_shot_spk 注册后的快速路径。
  勿"优化"成统一走 spk 注册。spk2info.pt 是只读快照（模型目录不回写），
  音色真相在 `config/voices.yaml`，worker 进程启动后按任务懒注册。
  VoiceCache 的缓存键含参考文件 (mtime,size)+逐字稿：`POST /v1/voices replace=true`
  换素材后 worker 不死也会自动重提特征，勿改回"按 voice_id 一次性注册"（陈旧 embedding）。
- **音色管理语义（wants.md v2，2026-10-05）**：重复注册默认 **409**，显式 `replace:true`
  才覆盖（旧版进 history，version+1，保留 20 版）；`kind/license/provenance/role/tags`
  为来源/许可元数据；`model` 不符部署名 422。试听端点正名 `/preview`（`/sample` 是
  兼容别名）。t2a inputs/generation **白名单**：未知键 400（"不支持的参数明确报错，
  不能接受后忽略"）。`client_ref` 是幂等请求编号：POST /v1/jobs(/batch) 同 ref 重放
  既有任务/批次（200 + idempotent_replay），tools/cosyvoice_smoke.py 每 case 用
  独立 ref，重跑会命中重放（200 也是成功）。
- cosyvoice 实测：加载 17s，VRAM 峰值 3.9G / RSS 7.3G（budget 6144/12288 有余量，
  仍属小档）；RTF 0.2–0.7。t2a 默认 seed=42（确定性 → §7 缓存复用）；输出 24k
  模型采样率 → 重采样到 `generation.sample_rate`（默认 48k）mono PCM16。
  wetext FST 自动缓存于 ~/.cache/modelscope/hub/pengzhendong（只含 .fst，不触发
  audit_weights 红线；缺失时官方降级为无前端，CV3 自带文本归一化，无碍）。
- 调度器 `_finish` 是取消竞态的最后闸口（completed + cancel_requested → 落定
  cancelled），勿在别处绕过它直接 set_status("completed")。取消路径的
  `Path.unlink` 与 `_ingest_outputs` 共用 `_managed_roots`/`_contained` 白名单，
  任何新增 callback 路径消费者都必须先过 containment。
