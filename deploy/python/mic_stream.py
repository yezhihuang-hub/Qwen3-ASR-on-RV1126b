#!/usr/bin/env python3
"""
Qwen3-ASR Microphone Streaming Recognition with VAD

Real-time speech recognition from microphone using arecord (ALSA).
Uses Silero VAD to skip silence periods and only process speech.

Usage:
    python mic_stream.py
    python mic_stream.py --device plughw:2,0 --chunk-size 5 --language Chinese
    python mic_stream.py --no-vad        # Disable VAD (fixed chunking)
    python mic_stream.py --list-devices

Flow:
    1. Start arecord capturing 16kHz mono PCM
    2. Feed audio to Silero VAD (RTF=0.043, negligible)
    3. When speech detected: accumulate audio, encode 5s chunks on NPU
    4. When speech ends: process remaining audio
    5. During silence: skip ASR processing (save compute)
    6. Display real-time results
    7. Ctrl+C to stop
"""

import argparse
import os
import sys
import time
import signal
import subprocess
import numpy as np
from pathlib import Path


_running = True


def signal_handler(sig, frame):
    global _running
    _running = False
    print("\n[Stopping...]")


def list_audio_devices():
    """List available ALSA capture devices."""
    print("=== ALSA Capture Devices ===")
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except FileNotFoundError:
        print("ERROR: arecord not found. Install alsa-utils.")
    except Exception as e:
        print(f"ERROR: {e}")


def find_model_dir():
    """Auto-detect model directory."""
    candidates = [
        Path(__file__).parent.parent / "models",
        Path("/home/qztest/qwen3asr_rknn/models"),
    ]
    for c in candidates:
        if c.exists() and (c / "mel_filters.npy").exists():
            return str(c)
    raise FileNotFoundError("Cannot find model directory. Use --model-dir.")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR Microphone Streaming with VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--device", default="plughw:CARD=rockchiprv1126b,DEV=0",
                        help="ALSA device (default: plughw:2,0)")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--platform", default="rv1126b",
                        choices=["rk3576", "rk3588", "rv1126b"])

    parser.add_argument("--chunk-size", type=float, default=5.0,
                        help="Encoder chunk seconds (default: 5)")
    parser.add_argument("--memory-num", type=int, default=2,
                        help="Sliding window chunks (default: 2)")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--context", default="")
    parser.add_argument("--cpus", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--decoder-quant", default="w4a16",
                        choices=["w4a16", "w4a16_g128", "w8a8"],
                        help="Decoder quantization type")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--rollback-tokens", type=int, default=2,
                        help="Rollback tokens for chunk boundary correction")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Max recording seconds")

    # VAD options
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable VAD (use fixed chunking)")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD speech threshold (0-1)")
    parser.add_argument("--vad-min-silence", type=float, default=0.5,
                        help="Min silence to end speech (seconds)")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    global _running
    signal.signal(signal.SIGINT, signal_handler)

    model_dir = args.model_dir or find_model_dir()
    language = args.language if args.language.lower() != "auto" else None

    sys.path.insert(0, str(Path(__file__).parent))
    from qwen3asr import Qwen3ASREngine

    print("Initializing ASR engine...")
    engine = Qwen3ASREngine(
        model_dir=model_dir,
        platform=args.platform,
        decoder_quant=args.decoder_quant,
        enabled_cpus=args.cpus,
        max_new_tokens=args.max_new_tokens,
        max_context_len=1024,
        top_k=1,
        temperature=1.0,
        verbose=True,
    )

    # Initialize VAD if enabled
    vad = None
    if not args.no_vad and engine.vad_model_path:
        from qwen3asr.vad import SileroVAD
        print("Initializing VAD...")
        vad = SileroVAD(
            model_path=engine.vad_model_path,
            threshold=args.vad_threshold,
            min_silence_duration=args.vad_min_silence,
            min_speech_duration=0.25,
            max_speech_duration=30.0,
        )
        print(f"  VAD ready (threshold={args.vad_threshold}, "
              f"silence={args.vad_min_silence}s)")
    elif not args.no_vad:
        print("  [WARN] VAD model not found, using fixed chunking")

    # Create streaming session
    stream = engine.create_stream(
        language=language,
        context=args.context,
        chunk_size=args.chunk_size,
        memory_num=args.memory_num,
        rollback_tokens=args.rollback_tokens,
        max_new_tokens=args.max_new_tokens,
        vad=vad,
    )

    # Start arecord.
    # RV1126B captures reliably as 48kHz stereo.
    # Keep qzxyz's original Python side: feed 16kHz mono raw PCM.
    # Use ffmpeg as a continuous stream resampler:
    #   arecord 48k stereo raw -> ffmpeg -> 16k mono raw -> Python.
    input_rate = 48000
    input_channels = 2
    asr_rate = 16000

    arecord_cmd = [
        "arecord",
        "-D", args.device,
        "-f", "S16_LE",
        "-r", str(input_rate),
        "-c", str(input_channels),
        "-t", "raw",
        "--buffer-size=16384",
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(input_rate),
        "-ac", str(input_channels),
        "-i", "pipe:0",
        "-f", "s16le",
        "-ar", str(asr_rate),
        "-ac", "1",
        "pipe:1",
    ]

    mode = "VAD+ASR" if vad else "Fixed Chunk"
    print(f"\n{'=' * 60}")
    print(f"  Qwen3-ASR Microphone Streaming ({mode})")
    print(f"  Device: {args.device}")
    print(f"  Chunk: {args.chunk_size}s, Memory: {args.memory_num}")
    print(f"  Language: {args.language}, CPUs: {args.cpus}")
    if vad:
        print(f"  VAD: threshold={args.vad_threshold}, "
              f"silence={args.vad_min_silence}s")
    print(f"  Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    try:
        arecord_proc = subprocess.Popen(
            arecord_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=arecord_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # Let ffmpeg own the pipe.
        if arecord_proc.stdout:
            arecord_proc.stdout.close()
        print("[REC] arecord + ffmpeg pipe started, microphone is recording...")
    except FileNotFoundError:
        print("ERROR: arecord or ffmpeg not found.")
        engine.close()
        return
    except Exception as e:
        print(f"ERROR starting audio pipeline: {e}")
        engine.close()
        return

    # Read audio in 0.5s chunks, exactly like qzxyz original:
    # 16kHz mono, 8000 samples, int16 -> 16000 bytes.
    read_samples = int(asr_rate * 0.5)
    read_bytes = read_samples * 2

    total_seconds = 0.0
    last_text = ""
    was_speech = False

    try:
        while _running:
            if args.max_seconds and total_seconds >= args.max_seconds:
                print(f"\n[Reached {args.max_seconds}s limit]")
                break

            raw = proc.stdout.read(read_bytes)
            if not raw:
                print("\n[Audio stream ended]")
                break

            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            total_seconds += len(pcm) / asr_rate

            result = stream.feed_audio(pcm)

            is_speech = result.get("is_speech", True)

            # Show speech indicator
            if is_speech and not was_speech:
                sys.stdout.write(f"\r\033[K[{total_seconds:.1f}s] 🎤 Listening...")
                sys.stdout.flush()
            elif not is_speech and was_speech:
                # Speech just ended, show processing indicator
                if result.get("chunks_processed", 0) > 0:
                    pass  # Text will be updated below

            was_speech = is_speech

            # Display text updates
            if result["text"] != last_text:
                last_text = result["text"]
                sys.stdout.write(
                    f"\r\033[K[{total_seconds:.1f}s] {last_text}")
                sys.stdout.flush()

            # Display idle/speech status when no text changes
            if not is_speech and result["text"] == last_text:
                utt = result.get("utterances", 0)
                if utt > 0:
                    sys.stdout.write(
                        f"\r\033[K[{total_seconds:.1f}s] "
                        f"(idle, {utt} utterances) {last_text[-60:]}")
                else:
                    sys.stdout.write(
                        f"\r\033[K[{total_seconds:.1f}s] (waiting...)")
                sys.stdout.flush()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            arecord_proc.terminate()
            arecord_proc.wait(timeout=2)
        except Exception:
            try:
                arecord_proc.kill()
            except Exception:
                pass

        print("\n\nFinalizing...")
        final = stream.finish()

        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULT")
        print(f"  Language: {final['language']}")
        print(f"  Text: {final['text']}")
        print(f"{'=' * 60}")

        stats = final.get("stats", {})
        if stats:
            print(f"\n  Audio processed: {stats.get('total_audio_s', 0):.1f}s")
            print(f"  Encoder: {stats.get('total_enc_ms', 0):.0f}ms")
            print(f"  Decoder: {stats.get('total_llm_ms', 0):.0f}ms")
            print(f"  VAD: {stats.get('total_vad_ms', 0):.0f}ms")
            print(f"  RTF: {stats.get('rtf', 0):.3f}")
            print(f"  Chunks: {stats.get('chunks', 0)}")
            print(f"  Utterances: {stats.get('utterances', 0)}")

        engine.close()


if __name__ == "__main__":
    main()
