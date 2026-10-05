"""CosyVoice3 provider: t2a (配音对白) on the CosyVoice worker subprocess.

台词与表演指令分离传入：text 永远逐字朗读；instruction 只进
instruct 通道（"You are a helpful assistant. X<|endofprompt|>"）。多音字
hotfix 由 text 内联拼音标记（如 [j][ǐ]）直接透传，官方能力。
"""
from __future__ import annotations

from .. import voices
from ..store import cache_hash

DEFAULTS = {
    "speed": 1.0,
    "seed": 42,           # 确定性默认：同参数重放命中缓存（§7 复用）
    "sample_rate": 48000,
}
# 不支持的参数必须明确报错，不能接受后忽略——输入/生成字段
# 都走白名单校验，未知 key 直接 400。
KNOWN_INPUTS = {"text", "voice_id", "instruction", "reference_audio",
                "reference_text", "text_frontend", "duration_limit_s"}
SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000)
INSTRUCT_PREFIX = "You are a helpful assistant. "
END_PROMPT = "<|endofprompt|>"
INSTRUCT_MAX = 500


def wrap_instruct(instruction: str) -> str:
    s = instruction.strip()
    if END_PROMPT in s:
        return s
    return f"{INSTRUCT_PREFIX}{s}{END_PROMPT}"


def normalize(workflow: str, inputs: dict, generation: dict) -> dict:
    if workflow != "t2a":
        raise ValueError(f"cosyvoice: unknown workflow {workflow}")
    # reject, never silently drop, unrecognized knobs.
    unknown_in = set(inputs) - KNOWN_INPUTS
    if unknown_in:
        raise ValueError(f"cosyvoice.t2a: unsupported input(s) {sorted(unknown_in)}; "
                         f"allowed: {sorted(KNOWN_INPUTS)}")
    unknown_gen = set(generation) - set(DEFAULTS)
    if unknown_gen:
        raise ValueError(f"cosyvoice.t2a: unsupported generation key(s) "
                         f"{sorted(unknown_gen)}; allowed: {sorted(DEFAULTS)}")

    text = str(inputs.get("text", "")).strip()
    if not text:
        raise ValueError("cosyvoice.t2a: 'text' is required")
    if len(text) > 5000:
        raise ValueError("cosyvoice.t2a: 'text' too long (>5000 chars); split it")

    gen = {**DEFAULTS, **generation}
    gen["speed"] = float(gen["speed"])
    if not 0.5 <= gen["speed"] <= 2.0:
        raise ValueError("cosyvoice.t2a: speed must be in [0.5, 2.0]")
    gen["seed"] = int(gen["seed"])
    gen["sample_rate"] = int(gen["sample_rate"])
    if gen["sample_rate"] not in SAMPLE_RATES:
        raise ValueError(f"cosyvoice.t2a: sample_rate must be one of {SAMPLE_RATES}")

    dls = inputs.get("duration_limit_s")
    if dls is not None:
        dls = float(dls)
        if dls <= 0:
            raise ValueError("cosyvoice.t2a: duration_limit_s must be > 0 seconds")

    ref_asset = inputs.get("reference_audio")
    ref_text = str(inputs.get("reference_text", "")).strip()
    assets: dict[str, str] = {}
    prompt_path = ""
    if ref_asset and ref_text:
        # 内联试听参考（选声对比用）：不做 spk 注册持久化，逐句现算
        voice_id = f"inline:{ref_asset}"
        prompt_text = ref_text
        assets["reference_audio"] = str(ref_asset)
    elif ref_asset or ref_text:
        raise ValueError("cosyvoice.t2a: 'reference_audio' requires 'reference_text' (and vice versa)")
    else:
        voice_id = str(inputs.get("voice_id", "")).strip()
        if not voice_id:
            raise ValueError("cosyvoice.t2a: 'voice_id' is required "
                             "(or inline reference_audio + reference_text; GET /v1/voices)")
        v = voices.get_voice(voice_id)
        if v is None:
            raise ValueError(f"cosyvoice.t2a: unknown voice_id {voice_id!r}; see GET /v1/voices")
        prompt_text = v["prompt_text"]
        if v["asset"]:
            assets["reference_audio"] = v["asset"]
        else:
            prompt_path = v["path"]

    instruction = str(inputs.get("instruction", "")).strip()
    if len(instruction) > INSTRUCT_MAX:
        raise ValueError(f"cosyvoice.t2a: instruction too long (>{INSTRUCT_MAX}); keep it a short style phrase")
    instruct_text = wrap_instruct(instruction) if instruction else None
    # wetext 文本前端默认开（本机 FST 已缓存）；显式 false 时数字/缩写按原文
    # 送模型（CV3 自带文本归一化）
    text_frontend = bool(inputs.get("text_frontend", True))

    return {
        "provider": "cosyvoice", "workflow": "t2a",
        "text": text, "instruct_text": instruct_text,
        "instruction": instruction,   # 原样回显（回执 §3：实际表演参数）
        "voice_id": voice_id, "prompt_text": prompt_text,
        "prompt_path": prompt_path, "text_frontend": text_frontend,
        "duration_limit_s": dls,
        "generation": gen, "assets": assets,
    }


def cache_key(payload: dict) -> str | None:
    """Deterministic requests only (seed >= 0), same posture as zit."""
    gen = payload["generation"]
    if int(gen.get("seed", -1)) < 0:
        return None
    return cache_hash({
        "provider": "cosyvoice", "workflow": "t2a", "text": payload["text"],
        "instruct_text": payload["instruct_text"], "voice_id": payload["voice_id"],
        "prompt_text": payload["prompt_text"], "text_frontend": payload["text_frontend"],
        "speed": gen["speed"], "sample_rate": gen["sample_rate"], "seed": gen["seed"],
        "assets": payload["assets"],
    })


def asset_slots(payload: dict) -> dict[str, str]:
    return dict(payload["assets"])


def compile(payload: dict, asset_paths: dict[str, str], output_dir, base_dir) -> dict:
    gen = payload["generation"]
    prompt_wav = asset_paths.get("reference_audio") or payload["prompt_path"]
    if not prompt_wav:
        raise ValueError("cosyvoice.t2a: reference audio could not be resolved")
    return {
        "job_id": payload["job_id"],
        "mode": "t2a",
        "text": payload["text"],
        "instruct_text": payload["instruct_text"],
        "instruction": payload.get("instruction", ""),
        "voice_id": payload["voice_id"],
        "prompt_text": payload["prompt_text"],
        "prompt_wav": prompt_wav,
        "speed": gen["speed"], "seed": int(gen["seed"]),
        "sample_rate": gen["sample_rate"],
        "text_frontend": payload["text_frontend"],
        "duration_limit_s": payload.get("duration_limit_s"),
        "output_dir": str(output_dir),
    }
