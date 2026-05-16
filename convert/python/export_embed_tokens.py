import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

src = "./Qwen3-ASR-0.6B"
out = "./qwen3asr_rv1126b_run/models/embed_tokens.npy"

m = Qwen3ASRModel.from_pretrained(
    src,
    dtype=torch.float32,
    device_map="cpu",
    max_new_tokens=8,
)

emb = m.model.thinker.model.embed_tokens.weight.detach().cpu().numpy()

print("embed_tokens:", emb.shape, emb.dtype)
print("saving fp16 to reduce disk size:", out)

np.save(out, emb.astype(np.float16))
print("DONE")
