import torch
import torch.nn.functional as F
from qwen_asr import Qwen3ASRModel


SECONDS = 5
FRAMES = SECONDS * 100
CHUNKS = FRAMES // 100
TOKENS = CHUNKS * 13


class FixedAudioTowerWrapper(torch.nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.audio_tower = audio_tower
        self.register_buffer(
            "cu_seqlens",
            torch.arange(0, TOKENS + 1, 13, dtype=torch.int32)
        )

    def forward(self, input_features):
        # input_features: [1, 128, FRAMES]
        x = input_features.squeeze(0)          # [128, FRAMES]
        x = x.transpose(0, 1)                  # [FRAMES, 128]

        # Fixed chunks: [FRAMES,128] -> [CHUNKS,100,128] -> [CHUNKS,1,128,100]
        x = x.reshape(CHUNKS, 100, 128)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = F.gelu(self.audio_tower.conv2d1(x))
        x = F.gelu(self.audio_tower.conv2d2(x))
        x = F.gelu(self.audio_tower.conv2d3(x))

        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.audio_tower.conv_out(x)       # [CHUNKS, 13, 896]

        pos = self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :]
        pos = pos.unsqueeze(0).to(x.dtype)
        x = x + pos

        hidden_states = x.reshape(TOKENS, -1)  # [TOKENS, 896]

        for layer in self.audio_tower.layers:
            hidden_states = layer(hidden_states, self.cu_seqlens)[0]

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        hidden_states = self.audio_tower.proj2(hidden_states)  # [TOKENS, 1024]

        return hidden_states.unsqueeze(0)      # [1, TOKENS, 1024]


src = "./Qwen3-ASR-0.6B"

m = Qwen3ASRModel.from_pretrained(
    src,
    dtype=torch.float32,
    device_map="cpu",
    max_new_tokens=8,
)

audio_tower = m.model.thinker.audio_tower.eval()

audio_tower.config._attn_implementation = "eager"
for layer in audio_tower.layers:
    layer.self_attn.config._attn_implementation = "eager"

wrapper = FixedAudioTowerWrapper(audio_tower).eval()
dummy = torch.randn(1, 128, FRAMES, dtype=torch.float32)

with torch.no_grad():
    y = wrapper(dummy)
    print("test output:", y.shape, y.dtype)

onnx_path = f"build_rv1126b/qwen3_asr_encoder_merged.fp16.{SECONDS}s.rv1126b.onnx"

torch.onnx.export(
    wrapper,
    dummy,
    onnx_path,
    input_names=["input_features"],
    output_names=["audio_embeds"],
    opset_version=13,
    export_params=True,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    dynamo=False,
    verbose=False,
)

print("Exported:", onnx_path)
