import numpy as np
import librosa
from pathlib import Path

sr = 16000
n_fft = 400
n_mels = 128

# Generate Mel filter bank.
# librosa output shape: [n_mels, n_fft // 2 + 1] = [128, 201]
mel = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=0,
    fmax=8000,
    htk=False,
    norm="slaney",
).astype(np.float32)

print("librosa mel:", mel.shape, mel.dtype)

# Runtime MelExtractor expects shape [201, 128].
mel_filters = mel.T.copy()

print("saved mel:", mel_filters.shape, mel_filters.dtype)

out_path = Path("/data/qwen3asr_rv1126b/models/mel_filters.npy")
out_path.parent.mkdir(parents=True, exist_ok=True)

np.save(out_path, mel_filters)

print("saved to:", out_path)
print("DONE")
