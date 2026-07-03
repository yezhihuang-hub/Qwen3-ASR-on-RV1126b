#!/usr/bin/env python3
import time
_SCRIPT_START = time.perf_counter()

import argparse
from pathlib import Path

import sherpa_onnx

from audio_utils_board import read_wav_float_mono, resample_linear, stats, wav_duration, write_json

_IMPORTS_DONE = time.perf_counter()


def write_profile_md(path: Path, profile: dict) -> None:
    timings = profile.get("timings", {})
    total = float(timings.get("total_script_sec") or 0.0)
    rows = []
    for key, value in timings.items():
        if key.endswith("_sec") and isinstance(value, (int, float)):
            pct = (float(value) / total * 100.0) if total > 0 else 0.0
            rows.append((key, float(value), pct))
    lines = [
        "# Diarization Internal Profile",
        "",
        f"- wav: `{profile.get('wav')}`",
        f"- audio duration sec: {profile.get('audio_duration_sec')}",
        f"- inference RTF: {profile.get('inference_rtf')}",
        f"- total RTF: {profile.get('total_rtf')}",
        f"- predicted speaker count: {profile.get('predicted_speaker_count')}",
        f"- segment count: {profile.get('segment_count')}",
        "",
        "## Timing",
        "",
        "| stage | sec | percent |",
        "| --- | ---: | ---: |",
    ]
    for key, value, pct in rows:
        lines.append(f"| {key} | {value:.6f} | {pct:.2f}% |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- sherpa_onnx.OfflineSpeakerDiarization.process() does not expose separate Python timing for pyannote segmentation/VAD, internal embedding extraction, clustering, and speaker assignment.",
            "- Those internal stages are included in diarization_inference_sec.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_timeline(path: Path, segments: list[dict]) -> None:
    lines = ["# Board sherpa diarization timeline", ""]
    for seg in segments:
        lines.append(f"- {seg['start']:.3f}-{seg['end']:.3f}s {seg['speaker_label']} ({seg['duration']:.3f}s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rttm(path: Path, wav: Path, segments: list[dict]) -> None:
    file_id = wav.stem
    lines = [
        f"SPEAKER {file_id} 1 {s['start']:.3f} {s['duration']:.3f} <NA> <NA> {s['speaker_label']} <NA> <NA>"
        for s in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    timings = {
        "python_imports_sec": _IMPORTS_DONE - _SCRIPT_START,
    }
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--segmentation-model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-speakers", type=int, default=-1)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    args = parser.parse_args()
    timings["argument_parse_sec"] = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    audio_duration = wav_duration(args.wav)
    timings["audio_duration_probe_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(args.segmentation_model)
            )
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(args.embedding_model)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_speakers,
            threshold=args.cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(f"Invalid diarization config: {config}")
    timings["config_setup_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    timings["diarizer_object_init_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    audio, sample_rate = read_wav_float_mono(args.wav)
    timings["audio_load_sec"] = time.perf_counter() - t0
    timings["audio_resample_sec"] = 0.0
    if sample_rate != sd.sample_rate:
        t0 = time.perf_counter()
        audio = resample_linear(audio, sample_rate, sd.sample_rate)
        sample_rate = sd.sample_rate
        timings["audio_resample_sec"] = time.perf_counter() - t0

    started = time.perf_counter()
    result = sd.process(audio).sort_by_start_time()
    elapsed = time.perf_counter() - started
    timings["diarization_inference_sec"] = elapsed

    t0 = time.perf_counter()
    segments = []
    for item in result:
        start = float(item.start)
        end = float(item.end)
        speaker = int(item.speaker)
        segments.append(
            {
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "speaker": speaker,
                "speaker_label": f"speaker_{speaker:02d}",
            }
        )
    timings["postprocess_sec"] = time.perf_counter() - t0

    durations = [s["duration"] for s in segments]
    metrics = {
        "wav": str(args.wav),
        "audio_duration_sec": audio_duration,
        "sample_rate": sample_rate,
        "elapsed_sec": elapsed,
        "rtf": elapsed / audio_duration if audio_duration > 0 else None,
        "number_of_segments": len(segments),
        "predicted_speaker_count": len({s["speaker_label"] for s in segments}),
        "num_speakers_arg": args.num_speakers,
        "cluster_threshold": args.cluster_threshold,
        "segment_duration_stats": stats(durations),
        "segmentation_model": str(args.segmentation_model),
        "embedding_model": str(args.embedding_model),
    }

    t0 = time.perf_counter()
    write_json(args.out_dir / "diarization_segments.json", segments)
    write_timeline(args.out_dir / "diarization_timeline.md", segments)
    write_rttm(args.out_dir / "diarization.rttm", args.wav, segments)
    timings["write_outputs_sec"] = time.perf_counter() - t0
    timings["total_script_sec"] = time.perf_counter() - _SCRIPT_START

    profile = {
        "wav": str(args.wav),
        "segmentation_model": str(args.segmentation_model),
        "embedding_model": str(args.embedding_model),
        "audio_duration_sec": audio_duration,
        "sample_rate": sample_rate,
        "inference_rtf": elapsed / audio_duration if audio_duration > 0 else None,
        "total_rtf": timings["total_script_sec"] / audio_duration if audio_duration > 0 else None,
        "predicted_speaker_count": metrics["predicted_speaker_count"],
        "segment_count": len(segments),
        "timings": timings,
        "internal_breakdown_available": False,
        "folded_into_diarization_inference_sec": [
            "pyannote_segmentation_or_vad",
            "speaker_embedding_extraction",
            "fast_clustering",
            "speaker_assignment",
        ],
    }
    metrics.update(
        {
            "total_elapsed_sec": timings["total_script_sec"],
            "total_rtf": profile["total_rtf"],
            "timing_breakdown": timings,
        }
    )
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "profile_internal.json", profile)
    write_profile_md(args.out_dir / "profile_internal.md", profile)
    print(f"diarization segments={len(segments)} speakers={metrics['predicted_speaker_count']} rtf={metrics['rtf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
