#!/usr/bin/env python3
"""
Qwen3-ASR File Transcription CLI

Transcribe audio files using RKNN encoder (NPU) + RKLLM decoder (NPU).

Usage:
    python transcribe.py --audio audio.wav
    python transcribe.py --audio audio.wav --language English --chunk-size 20
    python transcribe.py --audio audio.wav --context "新闻节目" --memory-num 3

All Parameters:
    --model-dir     Model root directory (default: auto-detect)
    --platform      rk3576 or rk3588 (default: rk3576)
    --audio         Audio file path (wav/mp3/m4a/flac)
    --language      Language hint: Chinese, English, etc. (default: Chinese)
                    Use "auto" for auto-detection
    --context       Context description for better accuracy
    --chunk-size    Audio chunk seconds (default: 30, max: encoder capacity)
    --memory-num    Chunks in sliding window (default: 2)
    --max-new-tokens  Max generated tokens per chunk (default: 500)
    --start         Start offset in seconds (default: 0)
    --duration      Duration in seconds (default: full file)
    --max-chunks    Max chunks to process (default: all)
    --cpus          CPU cores for RKLLM: 1,2,4 (default: 2)
    --top-k         Top-K sampling, 1=greedy (default: 1)
    --temperature   Sampling temperature (default: 1.0)
    --repeat-penalty Repetition penalty (default: 1.0)
    --no-itn        Disable Inverse Text Normalization
    --quiet         Minimal output
"""

import argparse
import os
import sys
import time
from pathlib import Path


def find_model_dir():
    """Auto-detect model directory."""
    candidates = [
        Path(__file__).parent.parent / "models",
        Path("/home/qztest/qwen3asr_rknn/models"),
        Path(os.environ.get("QWEN3ASR_MODEL_DIR", "")),
    ]
    for c in candidates:
        if c.exists() and (c / "mel_filters.npy").exists():
            return str(c)
    raise FileNotFoundError(
        "Cannot find model directory. Use --model-dir to specify."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR Transcription on RK3576/RK3588",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py --audio test.wav
  python transcribe.py --audio test.wav --language English
  python transcribe.py --audio podcast.m4a --context "科技新闻" --chunk-size 20
  python transcribe.py --audio lecture.wav --cpus 4 --max-chunks 10
        """
    )

    # Model
    parser.add_argument("--model-dir", default=None,
                        help="Model root directory")
    parser.add_argument("--platform", default="rv1126b",
                        choices=["rk3576", "rk3588", "rv1126b"])
    parser.add_argument("--lib-path", default=None,
                        help="Path to librkllmrt.so")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json")

    # Audio
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start offset (seconds)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration (seconds)")

    # ASR parameters
    parser.add_argument("--language", default="Chinese",
                        help="Language: Chinese, English, auto, etc.")
    parser.add_argument("--context", default="",
                        help="Context description")
    parser.add_argument("--chunk-size", type=float, default=30.0,
                        help="Chunk size in seconds (max 30)")
    parser.add_argument("--memory-num", type=int, default=2,
                        help="Sliding window chunks")
    parser.add_argument("--max-new-tokens", type=int, default=500,
                        help="Max tokens per chunk")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Max chunks to process")
    parser.add_argument("--rollback-tokens", type=int, default=5,
                        help="Prefix rollback tokens")

    # Sampling
    parser.add_argument("--cpus", type=int, default=2,
                        choices=[1, 2, 4, 8])
    parser.add_argument("--top-k", type=int, default=1,
                        help="Top-K (1=greedy)")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)

    # Output
    parser.add_argument("--no-itn", action="store_true",
                        help="Disable ITN")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    # Auto-detect model dir
    model_dir = args.model_dir or find_model_dir()

    # Handle language
    language = args.language
    if language.lower() == "auto":
        language = None

    # Import and initialize engine
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen3asr import Qwen3ASREngine

    engine = Qwen3ASREngine(
        model_dir=model_dir,
        platform=args.platform,
        lib_path=args.lib_path,
        tokenizer_path=args.tokenizer,
        enabled_cpus=args.cpus,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        verbose=not args.quiet,
    )

    try:
        result = engine.transcribe(
            audio=args.audio,
            language=language,
            context=args.context,
            chunk_size=args.chunk_size,
            memory_num=args.memory_num,
            rollback_tokens=args.rollback_tokens,
            max_new_tokens=args.max_new_tokens,
            start_second=args.start,
            duration=args.duration,
            max_chunks=args.max_chunks,
            apply_itn_flag=not args.no_itn,
        )

        print(f"\n{'=' * 60}")
        print(f"Language: {result['language']}")
        print(f"Text: {result['text']}")
        print(f"{'=' * 60}")

        stats = result["stats"]
        print(f"\nStats: audio={stats['audio_s']:.1f}s "
              f"wall={stats['wall_ms']:.0f}ms "
              f"RTF={stats['rtf']:.3f}")
        print(f"  Encoder: {stats['enc_ms']:.0f}ms")
        print(f"  Decoder: {stats['llm_ms']:.0f}ms")

    finally:
        engine.close()


if __name__ == "__main__":
    main()
