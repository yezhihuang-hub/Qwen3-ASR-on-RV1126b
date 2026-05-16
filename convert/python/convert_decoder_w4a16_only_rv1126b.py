#!/usr/bin/env python3
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from rkllm.api import RKLLM

SRC_MODEL = Path("./qwen3asr_rv1126b_run/models/decoder_hf").resolve()
OUT_DIR = Path("./qwen3asr_rv1126b_run/models/decoder").resolve()
DATASET = Path("./qwen3asr_rv1126b_run/models/data_quant.json").resolve()

TARGET_PLATFORM = "rv1126b"
NUM_NPU_CORE = 1
MAX_CONTEXT = 1024
OPT_LEVEL = 1

QUANT_DTYPE = "w4a16"
QUANT_ALGO = "grq"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(
        prefix="decoder_as_qwen3_rv1126b_",
        dir="./build_rv1126b",
    )).resolve()

    print(f"[INFO] tmpdir = {tmpdir}")

    for f in SRC_MODEL.iterdir():
        dst = tmpdir / f.name
        if f.name == "config.json":
            continue
        if f.name.endswith(".safetensors"):
            os.symlink(f, dst)
            print(f"[LINK] {f.name}")
        elif f.is_file():
            shutil.copy2(f, dst)
            print(f"[COPY] {f.name}")

    with open(SRC_MODEL / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    print("[INFO] original model_type:", config.get("model_type"))
    print("[INFO] original architectures:", config.get("architectures"))

    config["model_type"] = "qwen3"
    config["architectures"] = ["Qwen3ForCausalLM"]
    config["rope_scaling"] = {
        "rope_type": "default",
        "type": "default",
    }
    if "vision_config" in config:
        del config["vision_config"]

    with open(tmpdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("[INFO] modified model_type:", config["model_type"])
    print("[INFO] modified architectures:", config["architectures"])

    llm = RKLLM()

    print("[1/3] load_huggingface")
    ret = llm.load_huggingface(
        model=str(tmpdir),
        device="cpu",
        dtype="float32",
    )
    if ret != 0:
        print("[ERROR] load_huggingface failed:", ret)
        return ret

    print("[2/3] build w4a16")
    ret = llm.build(
        do_quantization=True,
        optimization_level=OPT_LEVEL,
        quantized_dtype=QUANT_DTYPE,
        quantized_algorithm=QUANT_ALGO,
        target_platform=TARGET_PLATFORM,
        num_npu_core=NUM_NPU_CORE,
        dataset=str(DATASET),
        max_context=MAX_CONTEXT,
    )
    if ret != 0:
        print("[ERROR] build failed:", ret)
        return ret

    save_path = OUT_DIR / f"decoder_qwen3.{QUANT_DTYPE}.{TARGET_PLATFORM}.rkllm"

    print("[3/3] export")
    ret = llm.export_rkllm(str(save_path))
    if ret != 0:
        print("[ERROR] export_rkllm failed:", ret)
        return ret

    print("[OK] saved:", save_path)
    if hasattr(llm, 'release'):
        llm.release()
    return 0

if __name__ == "__main__":
    sys.exit(main())
