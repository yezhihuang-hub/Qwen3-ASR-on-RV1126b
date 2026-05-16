import os
import json
import shutil
import torch
from safetensors.torch import save_file
from qwen_asr import Qwen3ASRModel

src = "./Qwen3-ASR-0.6B"
dst = "./qwen3asr_rv1126b_run/models/decoder_hf"

os.makedirs(dst, exist_ok=True)

print("[1/4] loading Qwen3-ASR")
m = Qwen3ASRModel.from_pretrained(
    src,
    dtype=torch.float32,
    device_map="cpu",
    max_new_tokens=8,
)

thinker = m.model.thinker

print("[2/4] collecting text decoder weights")
state = {}
skipped = []

for name, tensor in thinker.state_dict().items():
    if name.startswith("model."):
        state[name] = tensor.detach().cpu().to(torch.float16).contiguous()
    elif name == "lm_head.weight":
        # lm_head.weight is tied with model.embed_tokens.weight.
        # Keep only embed_tokens to save disk.
        skipped.append(name)
    elif name.startswith("lm_head."):
        state[name] = tensor.detach().cpu().to(torch.float16).contiguous()

print("saving tensors:", len(state))
print("skipped:", skipped)
numel = sum(v.numel() for v in state.values())
print("saved params:", numel)
print("estimated fp16 GB:", numel * 2 / 1024**3)

print("[3/4] saving model.safetensors")
save_file(state, os.path.join(dst, "model.safetensors"))

print("[4/4] writing config/tokenizer assets")
with open(os.path.join(src, "config.json"), "r", encoding="utf-8") as f:
    full_cfg = json.load(f)

text_cfg = full_cfg["thinker_config"]["text_config"]

# qzxyz's key trick: export ASR decoder as standard Qwen3.
text_cfg["model_type"] = "qwen3"
text_cfg["architectures"] = ["Qwen3ForCausalLM"]
text_cfg["tie_word_embeddings"] = True
text_cfg["vocab_size"] = full_cfg["thinker_config"]["vocab_size"]
text_cfg["torch_dtype"] = "float16"

# Avoid multimodal/mrope confusion in RKLLM EMBED mode.
text_cfg["rope_scaling"] = {
    "rope_type": "default",
    "type": "default"
}

with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
    json.dump(text_cfg, f, indent=2, ensure_ascii=False)

# Prefer qzxyz tokenizer.json if present.
qz_tok_dir = "./qwen3asr_rk_code/models/decoder_hf"
for fn in [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]:
    copied = False
    p1 = os.path.join(qz_tok_dir, fn)
    p2 = os.path.join(src, fn)
    if os.path.exists(p1):
        shutil.copy2(p1, os.path.join(dst, fn))
        copied = True
    elif os.path.exists(p2):
        shutil.copy2(p2, os.path.join(dst, fn))
        copied = True
    if copied:
        print("copied", fn)

print("DONE:", dst)
