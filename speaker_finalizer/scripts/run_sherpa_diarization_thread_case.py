#!/usr/bin/env python3
import argparse
import time
_SCRIPT_START = time.perf_counter()

from pathlib import Path

import sherpa_onnx

from audio_utils_board import read_wav_float_mono, resample_linear, stats, wav_duration, write_json

_IMPORTS_DONE = time.perf_counter()


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


def write_profile_md(path: Path, profile: dict) -> None:
    timings = profile["timings"]
    total = float(timings.get("total_script_sec") or 0.0)
    lines = [
        "# Diarization Thread Case Profile",
        "",
        f"- segmentation_num_threads: {profile['segmentation_num_threads']}",
        f"- embedding_num_threads: {profile['embedding_num_threads']}",
        f"- segmentation_provider: {profile['segmentation_provider']}",
        f"- embedding_provider: {profile['embedding_provider']}",
        f"- audio_duration_sec: {profile['audio_duration_sec']}",
        f"- inference_rtf: {profile['inference_rtf']}",
        f"- total_rtf: {profile['total_rtf']}",
        "",
        "| stage | sec | percent |",
        "| --- | ---: | ---: |",
    ]
    for key, value in timings.items():
        if key.endswith("_sec") and isinstance(value, (int, float)):
            pct = value / total * 100.0 if total > 0 else 0.0
            lines.append(f"| {key} | {value:.6f} | {pct:.2f}% |")
    lines.extend(
        [
            "",
            "Note: sherpa_onnx.OfflineSpeakerDiarization.process() does not expose separate segmentation/internal-embedding/clustering timing in Python.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    timings = {"python_imports_sec": _IMPORTS_DONE - _SCRIPT_START}
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--segmentation-model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-speakers", type=int, default=-1)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--segmentation-num-threads", type=int, default=1)
    parser.add_argument("--embedding-num-threads", type=int, default=1)
    parser.add_argument("--segmentation-provider", default="cpu")
    parser.add_argument("--embedding-provider", default="cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    timings["argument_parse_sec"] = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    audio_duration = wav_duration(args.wav)
    timings["audio_duration_probe_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    segmentation_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(args.segmentation_model)
        ),
        num_threads=args.segmentation_num_threads,
        debug=args.debug,
        provider=args.segmentation_provider,
    )
    embedding_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(args.embedding_model),
        num_threads=args.embedding_num_threads,
        debug=args.debug,
        provider=args.embedding_provider,
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation_cfg,
        embedding=embedding_cfg,
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

    t0 = time.perf_counter()
    result = sd.process(audio).sort_by_start_time()
    inference_sec = time.perf_counter() - t0
    timings["diarization_inference_sec"] = inference_sec

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
    t0 = time.perf_counter()
    write_json(args.out_dir / "diarization_segments.json", segments)
    write_timeline(args.out_dir / "diarization_timeline.md", segments)
    write_rttm(args.out_dir / "diarization.rttm", args.wav, segments)
    timings["write_outputs_sec"] = time.perf_counter() - t0
    timings["total_script_sec"] = time.perf_counter() - _SCRIPT_START

    metrics = {
        "wav": str(args.wav),
        "audio_duration_sec": audio_duration,
        "sample_rate": sample_rate,
        "elapsed_sec": inference_sec,
        "rtf": inference_sec / audio_duration if audio_duration > 0 else None,
        "total_elapsed_sec": timings["total_script_sec"],
        "total_rtf": timings["total_script_sec"] / audio_duration if audio_duration > 0 else None,
        "number_of_segments": len(segments),
        "predicted_speaker_count": len({s["speaker_label"] for s in segments}),
        "num_speakers_arg": args.num_speakers,
        "cluster_threshold": args.cluster_threshold,
        "segment_duration_stats": stats(durations),
        "segmentation_model": str(args.segmentation_model),
        "embedding_model": str(args.embedding_model),
        "segmentation_num_threads": args.segmentation_num_threads,
        "embedding_num_threads": args.embedding_num_threads,
        "segmentation_provider": args.segmentation_provider,
        "embedding_provider": args.embedding_provider,
        "timing_breakdown": timings,
    }
    profile = {
        **{k: metrics[k] for k in [
            "wav",
            "segmentation_model",
            "embedding_model",
            "audio_duration_sec",
            "segmentation_num_threads",
            "embedding_num_threads",
            "segmentation_provider",
            "embedding_provider",
        ]},
        "inference_rtf": metrics["rtf"],
        "total_rtf": metrics["total_rtf"],
        "predicted_speaker_count": metrics["predicted_speaker_count"],
        "segment_count": len(segments),
        "timings": timings,
    }
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "profile_internal.json", profile)
    write_profile_md(args.out_dir / "profile_internal.md", profile)
    print(
        "diarization "
        f"seg_threads={args.segmentation_num_threads} emb_threads={args.embedding_num_threads} "
        f"segments={len(segments)} speakers={metrics['predicted_speaker_count']} rtf={metrics['rtf']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
