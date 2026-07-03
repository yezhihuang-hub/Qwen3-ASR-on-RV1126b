#!/usr/bin/env python3
import time
_SCRIPT_START = time.perf_counter()

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sherpa_onnx

from audio_utils_board import read_wav_float_mono, resample_linear, wav_duration, write_json, write_wav_float_mono

_IMPORTS_DONE = time.perf_counter()


def write_profile_md(path: Path, profile: dict) -> None:
    timings = profile.get("timings", {})
    total = float(timings.get("total_script_sec") or 0.0)
    lines = [
        "# Identification Internal Profile",
        "",
        f"- wav: `{profile.get('wav')}`",
        f"- audio duration sec: {profile.get('audio_duration_sec')}",
        f"- total RTF: {profile.get('total_rtf')}",
        f"- embedding dim: {profile.get('embedding_dim')}",
        f"- cluster count: {profile.get('cluster_count')}",
        "",
        "## Timing",
        "",
        "| stage | sec | percent |",
        "| --- | ---: | ---: |",
    ]
    for key, value in timings.items():
        if key.endswith("_sec") and isinstance(value, (int, float)):
            pct = (float(value) / total * 100.0) if total > 0 else 0.0
            lines.append(f"| {key} | {float(value):.6f} | {pct:.2f}% |")
    lines.extend(["", "## Per Cluster", "", "| cluster | duration sec | extract sec | embedding sec | score sec | top1 | assigned |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for item in profile.get("clusters", []):
        lines.append(
            "| {label} | {duration:.6f} | {extract:.6f} | {embedding:.6f} | {score:.6f} | {score_val} | {assigned} |".format(
                label=item.get("label"),
                duration=float(item.get("cluster_duration_sec") or 0.0),
                extract=float(item.get("extract_sec") or 0.0),
                embedding=float(item.get("embedding_sec") or 0.0),
                score=float(item.get("score_sec") or 0.0),
                score_val=("" if item.get("top1_score") is None else f"{float(item['top1_score']):.6f}"),
                assigned=item.get("assigned_label"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_embedding(extractor, wav: Path | None = None, samples=None, sample_rate: int | None = None):
    if wav is not None:
        samples, sample_rate = read_wav_float_mono(wav)
    assert samples is not None and sample_rate is not None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=np.asarray(samples, dtype=np.float32))
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise RuntimeError("Speaker embedding stream is not ready")
    return np.asarray(extractor.compute(stream), dtype=np.float32)


def score_embedding(manager, embedding: np.ndarray, threshold: float) -> dict:
    vec = embedding.astype(np.float32).tolist()
    scores = []
    for name in manager.all_speakers:
        score = float(manager.score(name, vec))
        scores.append({"speaker": name, "score": score, "verified": bool(manager.verify(name, vec, threshold))})
    scores.sort(key=lambda x: x["score"], reverse=True)
    matched = manager.search(vec, threshold)
    return {
        "threshold": threshold,
        "matched_speaker": matched or None,
        "top1_speaker": scores[0]["speaker"] if scores else None,
        "top1_score": scores[0]["score"] if scores else None,
        "margin": scores[0]["score"] - scores[1]["score"] if len(scores) > 1 else None,
        "scores": scores,
    }


def write_identified_timeline(path: Path, segments: list[dict], scores: dict) -> None:
    lines = ["# Board sherpa identified timeline", ""]
    for seg in segments:
        label = seg["speaker_label"]
        item = scores.get(label, {})
        display = item.get("assigned_label") or label
        score = item.get("top1_score")
        score_text = f", top1_score={score:.6f}" if score is not None else ""
        lines.append(f"- {seg['start']:.3f}-{seg['end']:.3f}s {display} (anonymous={label}{score_text})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    timings = {
        "python_imports_sec": _IMPORTS_DONE - _SCRIPT_START,
    }
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--diarization", type=Path, required=True)
    parser.add_argument("--enroll-manifest", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    timings["argument_parse_sec"] = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = args.out_dir / "cluster_audio"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    audio_duration = wav_duration(args.wav)
    timings["audio_duration_probe_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments = json.loads(args.diarization.read_text(encoding="utf-8"))
    timings["load_diarization_json_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    manifest = json.loads(args.enroll_manifest.read_text(encoding="utf-8"))
    timings["load_enroll_manifest_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(args.embedding_model))
    if not extractor_config.validate():
        raise RuntimeError(f"Invalid embedding config: {extractor_config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)
    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    timings["embedding_model_init_sec"] = time.perf_counter() - t0

    enrolled = []
    timings["enrollment_audio_load_sec"] = 0.0
    timings["enrollment_embedding_sec"] = 0.0
    for item in manifest:
        name = str(item["speaker"])
        wav = Path(item["wav"])
        t0 = time.perf_counter()
        enroll_audio, enroll_sample_rate = read_wav_float_mono(wav)
        timings["enrollment_audio_load_sec"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        emb = compute_embedding(extractor, samples=enroll_audio, sample_rate=enroll_sample_rate)
        timings["enrollment_embedding_sec"] += time.perf_counter() - t0
        ok = manager.add(name, emb.tolist())
        if not ok:
            raise RuntimeError(f"Failed to enroll speaker {name}")
        enrolled.append({"speaker": name, "wav": str(wav)})

    t0 = time.perf_counter()
    audio, sample_rate = read_wav_float_mono(args.wav)
    timings["audio_load_sec"] = time.perf_counter() - t0
    timings["audio_resample_sec"] = 0.0
    by_speaker = defaultdict(list)
    for seg in segments:
        label = str(seg.get("speaker_label") or f"speaker_{int(seg['speaker']):02d}")
        start_i = max(0, int(round(float(seg["start"]) * sample_rate)))
        end_i = min(len(audio), int(round(float(seg["end"]) * sample_rate)))
        if end_i > start_i:
            by_speaker[label].append(audio[start_i:end_i])

    cluster_scores = {}
    cluster_profiles = []
    timings["cluster_audio_extract_total_sec"] = 0.0
    timings["cluster_audio_write_total_sec"] = 0.0
    timings["cluster_embedding_total_sec"] = 0.0
    timings["scoring_total_sec"] = 0.0
    for label, chunks in sorted(by_speaker.items()):
        t_extract = time.perf_counter()
        silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        parts = []
        for chunk in chunks:
            parts.append(chunk)
            parts.append(silence)
        cluster_audio = np.concatenate(parts).astype(np.float32)
        extract_sec = time.perf_counter() - t_extract
        timings["cluster_audio_extract_total_sec"] += extract_sec
        cluster_wav = cluster_dir / f"{label}.wav"
        t_write = time.perf_counter()
        write_wav_float_mono(cluster_wav, cluster_audio, sample_rate)
        write_sec = time.perf_counter() - t_write
        timings["cluster_audio_write_total_sec"] += write_sec
        t_embed = time.perf_counter()
        emb = compute_embedding(extractor, samples=cluster_audio, sample_rate=sample_rate)
        embedding_sec = time.perf_counter() - t_embed
        timings["cluster_embedding_total_sec"] += embedding_sec
        t_score = time.perf_counter()
        score = score_embedding(manager, emb, args.threshold)
        score_sec = time.perf_counter() - t_score
        timings["scoring_total_sec"] += score_sec
        score.update(
            {
                "anonymous_speaker": label,
                "assigned_label": score["matched_speaker"] or label,
                "cluster_wav": str(cluster_wav),
                "cluster_duration_sec": float(len(cluster_audio) / sample_rate),
                "segment_count": len(chunks),
            }
        )
        cluster_scores[label] = score
        cluster_profiles.append(
            {
                "label": label,
                "cluster_duration_sec": float(len(cluster_audio) / sample_rate),
                "extract_sec": extract_sec,
                "write_wav_sec": write_sec,
                "embedding_sec": embedding_sec,
                "score_sec": score_sec,
                "top1_score": score.get("top1_score"),
                "assigned_label": score.get("assigned_label"),
            }
        )

    timings["total_script_sec"] = time.perf_counter() - _SCRIPT_START
    elapsed = timings["total_script_sec"]
    metrics = {
        "wav": str(args.wav),
        "audio_duration_sec": audio_duration,
        "elapsed_sec": elapsed,
        "rtf": elapsed / audio_duration if audio_duration > 0 else None,
        "embedding_dim": int(extractor.dim),
        "enrolled": enrolled,
        "cluster_count": len(cluster_scores),
        "threshold": args.threshold,
        "embedding_model": str(args.embedding_model),
        "timing_breakdown": timings,
        "clusters": cluster_profiles,
    }
    profile = {
        "wav": str(args.wav),
        "diarization": str(args.diarization),
        "enroll_manifest": str(args.enroll_manifest),
        "embedding_model": str(args.embedding_model),
        "audio_duration_sec": audio_duration,
        "total_rtf": elapsed / audio_duration if audio_duration > 0 else None,
        "embedding_dim": int(extractor.dim),
        "cluster_count": len(cluster_scores),
        "timings": timings,
        "clusters": cluster_profiles,
    }
    t0 = time.perf_counter()
    write_json(args.out_dir / "board_mic_cluster_identification_scores.json", cluster_scores)
    write_json(args.out_dir / "cluster_identification_scores.json", cluster_scores)
    write_identified_timeline(args.out_dir / "identified_timeline.md", segments, cluster_scores)
    write_identified_timeline(args.out_dir / "final_speaker_only_timeline.md", segments, cluster_scores)
    timings["write_outputs_sec"] = time.perf_counter() - t0
    timings["total_script_sec"] = time.perf_counter() - _SCRIPT_START
    elapsed = timings["total_script_sec"]
    metrics["elapsed_sec"] = elapsed
    metrics["rtf"] = elapsed / audio_duration if audio_duration > 0 else None
    metrics["timing_breakdown"] = timings
    profile["total_rtf"] = metrics["rtf"]
    profile["timings"] = timings
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "profile_internal.json", profile)
    write_profile_md(args.out_dir / "profile_internal.md", profile)
    print(f"identification clusters={len(cluster_scores)} rtf={metrics['rtf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
