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
python3 tools/verify_weights.py         # 全部 35 项权重 149.8G（size+sha256+header），失败带修复命令
```

权重（全部在 /mnt/data/AV，红线）：
- ZIT diffusers 快照：`/mnt/data/AV/models/Z-Image-Turbo`
- LTX：`/mnt/data/AV/ComfyUI/models/{checkpoints,diffusion_models,...}`（2.5 权重未到位，capability 自动 unavailable）
- MH3：待重新下载 INT8 convrot 到 ComfyUI models 目录

## 当前状态（2026-09-30）

| Profile | 权重 | E2E |
|---|---|---|
| zit | ✅ 在盘 | ✅ t2i batch / i2i / inpaint / 缓存 / 崩溃恢复 |
| ltx25 | ✅ LTX-2.5 distilled int8-convrot 38.7G + latent upscalers 1.26G + bbox IC-LoRA 0.33G | ✅ t2v / i2v / flf2v / a2v / **bbox_control** 首验出片；`generation.mode:"high"` = Two-Stage（latent x2 + 3步 re-sampler）t2v/i2v/flf2v 已验证，bbox HD 实测 1536×896 出片；官方 union/motion/inpaint IC-LoRA 仍 gated:auto（需 HF 网页端许可），bbox 为第三方 GPL 兼容版 |
| mh3.fl2va | ✅ pruned int8-convrot + nvfp4 TE + 双 VAE ~50.5G + turbo LoRA 2.0G | ✅ t2v / i2v / fl2v / multiframe(含 video 关键帧=continuation) 首验出片，capability 已点亮 |
| mh3.ref2va | ✅ ref2va pruned int8 21G + turbo LoRA 2.0G + fun-controlnet union int8 6.8G（2.0 + 原版） | ✅ ref2v（双参考图 Autogrow）与 **fun_control**（ModelPatch controlnet）首验出片，capability 已点亮 |

权重全部位于 `/mnt/data/AV`（`tools/audit_weights.py` 审计通过），
统一 Python 环境 `/home/jeefy/miniconda3/envs/comfyui`（openclaw-home image env 已删除）。
新 workflow 首验用 `tools/first_validation.py`（in-process 生产路径，无需服务在跑，成功后自动点亮 capability）。
