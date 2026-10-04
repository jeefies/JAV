# JAV — Jeefy Audio-Video Generation Platform

统一管理 **Z-Image-Turbo / MiniMax-H3 / LTX 2.5 Fast** 的单 GPU 生成服务：
统一 `/v1/jobs` API、batch 提交、runtime profile 互斥调度、RAM/VRAM 准入保护。

设计文档：`/mnt/data/AV/JAV-DESIGN.md`（架构与决策依据）
API 文档：`API.md`

## 布局

```
JAV/
├── jav/                  # 服务端包
│   ├── server.py         # FastAPI 入口 (uvicorn jav.server:create_app --factory)
│   ├── config.py         # 路径/profile/调度常量
│   ├── models.py         # Pydantic 域模型（provider/workflow→profile 映射）
│   ├── store.py          # SQLite(WAL): jobs/batches/assets/outputs/runtime_events
│   ├── capabilities.py   # 能力发现（权重×模板×smoke 动态 available）
│   ├── scheduler.py      # affinity + 防饿死 + 准入退避 + 空闲卸载
│   ├── api/              # jobs/assets/system 路由
│   ├── providers/        # zit/ltx25/mh3 参数校验+编译（含 ComfyUI 模板引擎）
│   ├── workflows/        # ComfyUI api.json + manifest（按 provider 分目录）
│   └── runtime/          # supervisor + zit_worker/comfy backend
├── sdk/jav/              # 笔记本 Python SDK（单文件，requests-only）
├── config/profiles.yaml  # 可选 profile 覆盖
├── data/                 # jav.db · assets/ · outputs/ · pending/ · hf_home/
├── logs/                 # worker/comfyui 运行日志
├── tools/                # wait_batch · watch_jobs · comfy_smoke · audit_weights
├── deploy/JAV.service    # systemd --user 单元（含 cgroup 内存硬顶）
└── tests/                # pytest（FakeBackend 驱动，不占 GPU）
```

## 运行

```bash
systemctl --user start JAV.service      # :8765，日志 journalctl --user -u JAV
/home/jeefy/miniconda3/envs/comfyui/bin/python3 -m pytest tests -q
python3 tools/verify_weights.py         # 全部权重（registry 47 项 ~153G：size+sha256+header），失败带修复命令
```

权重（全部在 /mnt/data/AV，红线）：
- ZIT diffusers 快照：`/mnt/data/AV/models/Z-Image-Turbo`
- LTX：`/mnt/data/AV/ComfyUI/models/{checkpoints,diffusion_models,...}`（2.5 distilled int8 + 控制 LoRA 全量在盘）
- MH3：`ComfyUI/models/` int8-convrot 套件 + turbo/fun-control LoRA

## 当前状态（2026-10-04）

| Profile | 权重 | E2E |
|---|---|---|
| zit | ✅ 在盘 | ✅ t2i batch / i2i / inpaint / 缓存 / 崩溃恢复 |
| ltx25 | ✅ LTX-2.5 distilled int8-convrot 38.7G + latent upscalers 1.26G + bbox IC-LoRA 0.33G + **5 控制 LoRA**（union-control 2.3 / clean-plate / slow-motion / ingredients / cinemagraph 2.5，共 1.9G） | ✅ t2v / i2v / flf2v / a2v / **bbox_control** 首验出片；`generation.mode:"high"` = Two-Stage（latent x2 + 3步 re-sampler）t2v/i2v/flf2v 已验证，bbox HD 实测 1536×896 出片；**控制族 6 图（union/motion/inpaint/outpaint/ic_lora×2 mode）全部真实出片**（官方 2.5 UI 样例 subgraph 展开重写，时长画幅跟随源视频） |
| mh3.fl2va | ✅ pruned int8-convrot + nvfp4 TE + 双 VAE ~50.5G + turbo LoRA 2.0G | ✅ t2v / i2v / fl2v / multiframe(含 video 关键帧=continuation) 首验出片，capability 已点亮 |
| mh3.ref2va | ✅ ref2va pruned int8 21G + turbo LoRA 2.0G + fun-controlnet union int8 6.8G（2.0 + 原版） | ✅ ref2v（双参考图 Autogrow）与 **fun_control**（ModelPatch controlnet）首验出片，capability 已点亮 |
| cosyvoice | ✅ Fun-CosyVoice3-0.5B-2512 9.1G（/mnt/data/AV/models/Fun-CosyVoice3-0.5B；worker 用独立 venv `/mnt/data/AV/venvs/cosyvoice`，torch 2.7.1+cu128——上游 pin 2.3.1 不支持本机 sm_120） | ✅ t2a 冒烟 3/3（plain/instruct2/内联试听+speed，48kHz mono PCM + duration_s + 缓存命中，VRAM 实测 3.7G / RSS 6.7G，见 `tools/cosyvoice_smoke.py`） |

20/20 工作流 available。**真实生成回归**：`tools/live_e2e.py`（对运行中的服务走
HTTP + SDK，产物用 PIL/ffprobe 做内容级断言，分阶段 `--stage zit,ltx25,control,mh3`）；
新工作流首验用 `tools/first_validation.py`（单项）或 `tools/batch_first_validation.py`
（多项同进程批量，共享一次 ComfyUI 冷启动）——两者都走 in-process 生产路径并自动
点亮 capability，且带防双 runtime 互斥锁。FakeBackend pytest 与真实出片两层缺一不可。
