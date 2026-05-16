#!/usr/bin/env python3
"""
Re-export Qwen3-ASR decoder as standard Qwen3 (NOT Qwen3-VL) for RKLLM.

Key idea: Qwen3-ASR audio tokens use 1D sequential positions (same as text).
When model_type="qwen3_vl", RKLLM might assign 2D image positions to EMBED tokens.
By changing to model_type="qwen3" with standard RoPE, RKLLM assigns 1D positions.

Since mrope with all 3 dims equal = standard RoPE, the model weights are compatible.
"""
import os, sys, json, shutil, tempfile

from rkllm.api import RKLLM

SRC_MODEL   = "./qwen3asr_rv1126b_run/models/decoder_hf"
OUT_DIR     = "./qwen3asr_rv1126b_run/models/decoder"

def main():
    # Create a temporary modified config
    tmpdir = tempfile.mkdtemp(prefix="decoder_as_qwen3_")
    print(f"[INFO] Creating modified model in {tmpdir}")

    # Copy all files from source
    for f in os.listdir(SRC_MODEL):
        src = os.path.join(SRC_MODEL, f)
        dst = os.path.join(tmpdir, f)
        if os.path.isfile(src):
            if f == "config.json":
                continue  # we'll create modified version
            shutil.copy2(src, dst)

    # Modify config.json: change to standard Qwen3
    with open(os.path.join(SRC_MODEL, "config.json"), "r") as f:
        config = json.load(f)

    print(f"[INFO] Original model_type: {config.get('model_type')}")
    print(f"[INFO] Original architectures: {config.get('architectures')}")
    print(f"[INFO] Original rope_scaling: {config.get('rope_scaling')}")

    # Change to standard Qwen3
    config["model_type"] = "qwen3"
    config["architectures"] = ["Qwen3ForCausalLM"]

    # Remove vision_config
    if "vision_config" in config:
        del config["vision_config"]

    # Simplify rope_scaling: remove mrope fields, keep standard
    config["rope_scaling"] = {
        "rope_type": "default",
        "type": "default"
    }

    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Modified model_type: {config['model_type']}")
    print(f"[INFO] Modified architectures: {config['architectures']}")
    print(f"[INFO] Modified rope_scaling: {config['rope_scaling']}")

    # Export fp16 for RK3576
    os.makedirs(OUT_DIR, exist_ok=True)

    llm = RKLLM()
    ret = llm.load_huggingface(model=tmpdir, device="cpu", dtype="float32")
    if ret != 0:
        print(f"[ERROR] load_huggingface failed: {ret}")
        return ret

    ret = llm.build(
        do_quantization=False,
        optimization_level=1,
        target_platform="rv1126b",
        num_npu_core=1,
        max_context=4096,
    )
    if ret != 0:
        print(f"[ERROR] build failed: {ret}")
        return ret

    save_path = os.path.join(OUT_DIR, "decoder_qwen3.fp16.rv1126b.rkllm")
    ret = llm.export_rkllm(save_path)
    if ret != 0:
        print(f"[ERROR] export failed: {ret}")
        return ret

    print(f"[OK] saved: {save_path}")

    # Also export w4a16 for comparison
    print("\n[INFO] Now exporting w4a16...")
    llm2 = RKLLM()
    ret = llm2.load_huggingface(model=tmpdir, device="cpu", dtype="float32")
    if ret != 0:
        print(f"[ERROR] load_huggingface for w4a16 failed: {ret}")
        return ret

    dataset = "./qwen3asr_rv1126b_run/models/data_quant.json"
    ret = llm2.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w4a16",
        quantized_algorithm="grq",
        target_platform="rv1126b",
        num_npu_core=1,
        dataset=dataset,
        max_context=4096,
    )
    if ret != 0:
        print(f"[ERROR] build w4a16 failed: {ret}")
        return ret

    save_path2 = os.path.join(OUT_DIR, "decoder_qwen3.w4a16.rv1126b.rkllm")
    ret = llm2.export_rkllm(save_path2)
    if ret != 0:
        print(f"[ERROR] export w4a16 failed: {ret}")
        return ret

    print(f"[OK] saved: {save_path2}")

    # Cleanup
    shutil.rmtree(tmpdir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
