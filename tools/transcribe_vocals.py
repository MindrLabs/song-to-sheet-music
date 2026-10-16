#!/usr/bin/env python3
"""Transcribe a vocal stem into notes with GAME, openvpi's singing transcriber.

Usage: transcribe_vocals.py <vocals.wav> <out.notes.json> [model] [language]

  model     small (default), medium or large — a released ONNX build of GAME
            (Generative Adaptive MIDI Extractor, MIT), downloaded on first use
            into $GAME_CACHE or ~/.cache/game-onnx; or a directory holding one
  language  en, ja, yue or zh, the languages GAME was trained on; anything
            else (or "") leaves it unset, which is what Korean songs get

Writes the notes the way the transcribe-* workflows save basic-pitch's —
start_time / end_time / pitch / velocity, in seconds — plus `score`, GAME's
pitch before rounding. `pitch` is rounded after the song's tuning offset is
taken off (see tuning()). Prints a summary.

Why GAME and not basic-pitch
----------------------------
model-compose's transcribers are basic-pitch and piano-transcription; neither
is made for a singing voice. Against the melody line the purchased parts
print, on six songs (male and female, Korean and English, a shouted punk
song), share of notes with a matching onset and pitch (one octave allowed for
the whole song: a man's line is written an octave up):

  basic-pitch (vocal range, one note at a time)   0.53
  pYIN / RMVPE pitch tracks cut into notes        0.50 / 0.55
  GAME small / medium                             0.69 / 0.71

GAME learns where notes begin and end — a repeated syllable on one pitch, a
slide — rather than cutting a pitch track by rule, and it was trained with
accompaniment and noise mixed in, so a separated stem with leftovers suits it.
Small scores like medium (six songs) and like large (27 songs: 0.60 and
0.59) at an eighth of large's size and a fifth of its time.

How it runs
-----------
GAME's ONNX release is five modules: an encoder, a segmenter that finds note
boundaries by discrete diffusion (D3PM, eight refinement steps), an estimator
that gives each note its pitch and whether it is sung at all, and two helpers
converting between durations and boundary frames. They run on ONNX Runtime
alone. The stem is cut at silences first, as GAME's own command line does
(-40 dB, pieces of at least 1 s, gaps of 200 ms); a piece longer than
MAX_PIECE is cut again at its quietest point, since attention over one long
piece is slow.

The segmenter samples: each refinement step draws random numbers, so the
boundaries — and some notes — change from run to run. ONNX Runtime is seeded
($GAME_SEED, 0 by default) so a run can be repeated exactly. $GAME_THREADS
caps the CPU threads (default: every core) on a shared machine.
"""
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

RELEASE = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-{size}-onnx.zip"
MODULES = ("encoder", "dur2bd", "segmenter", "bd2dur", "estimator")
STEPS = 8                 # D3PM refinement steps (GAME's default)
BOUNDARY_THRESHOLD = 0.2
BOUNDARY_RADIUS = 2       # frames
PRESENCE_THRESHOLD = 0.2
SILENCE_DB = -40.0
MIN_PIECE = 1.0           # seconds
MIN_GAP = 0.2
KEEP_SILENCE = 0.1
MAX_PIECE = 30.0
MIN_NOTE = 0.06


def model_dir(name):
    if Path(name).is_dir():
        return Path(name)
    cache = Path(os.environ.get("GAME_CACHE", Path.home() / ".cache" / "game-onnx"))
    target = cache / f"GAME-1.0.3-{name}-onnx"
    if not (target / "config.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        zpath = cache / f"{target.name}.zip"
        urllib.request.urlretrieve(RELEASE.format(size=name), zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cache)
        zpath.unlink()
    return target


def pieces(y, sr):
    """[(start sample, end sample)] of the sung stretches, cut at silences."""
    hop = int(0.02 * sr)
    win = 4 * hop
    frames = max(1, (len(y) - win) // hop + 1)
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(frames)])
    loud = 20 * np.log10(rms) > SILENCE_DB
    out, start, quiet_run = [], None, 0
    for i, on in enumerate(loud):
        if on:
            if start is None:
                start = i
            quiet_run = 0
        elif start is not None:
            quiet_run += 1
            if quiet_run * hop >= MIN_GAP * sr:
                end = i - quiet_run + 1
                out.append((start, end))
                start, quiet_run = None, 0
    if start is not None:
        out.append((start, frames))
    keep = int(KEEP_SILENCE * sr / hop)
    spans = [(max(0, a - keep) * hop, min(len(y), (b + keep) * hop + win)) for a, b in out]
    merged = []                                   # a short stretch joins the one before it
    for a, b in spans:
        if merged and (b - a < MIN_PIECE * sr or merged[-1][1] - merged[-1][0] < MIN_PIECE * sr) \
                and a - merged[-1][1] < MIN_GAP * sr * 5:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    final = []
    for a, b in merged:
        final += split_long(y, sr, a, b, rms, hop)
    return final


def split_long(y, sr, a, b, rms, hop):
    if b - a <= MAX_PIECE * sr:
        return [(a, b)]
    lo, hi = (a + int(0.25 * (b - a))) // hop, (a + int(0.75 * (b - a))) // hop
    cut = (lo + int(np.argmin(rms[lo:hi]))) * hop
    return split_long(y, sr, a, cut, rms, hop) + split_long(y, sr, cut, b, rms, hop)


def transcribe(y, sr, sessions, language_id):
    ts = np.linspace(0.0, 1.0, STEPS, endpoint=False).astype(np.float32)
    has_language = "language" in [i.name for i in sessions["segmenter"].get_inputs()]
    notes = []
    for a, b in pieces(y, sr):
        w = y[a:b][None].astype(np.float32)
        dur = np.array([w.shape[1] / sr], dtype=np.float32)
        x_seg, x_est, mask = sessions["encoder"].run(None, {"waveform": w, "duration": dur})
        known = sessions["dur2bd"].run(None, {"durations": dur[None], "maskT": mask})[0]
        bounds = known
        for t in ts:
            feed = {"x_seg": x_seg, "known_boundaries": known, "prev_boundaries": bounds,
                    "t": np.array([t], np.float32), "maskT": mask,
                    "threshold": np.array(BOUNDARY_THRESHOLD, np.float32),
                    "radius": np.array(BOUNDARY_RADIUS, np.int64)}
            if has_language:
                feed["language"] = np.array([language_id], np.int64)
            bounds = sessions["segmenter"].run(None, feed)[0]
        durations, mask_n = sessions["bd2dur"].run(None, {"boundaries": bounds, "maskT": mask})
        presence, scores = sessions["estimator"].run(None, {
            "x_est": x_est, "boundaries": bounds, "maskT": mask, "maskN": mask_n,
            "threshold": np.array(PRESENCE_THRESHOLD, np.float32)})
        onset = a / sr
        for d, valid, sung, score in zip(durations[0], mask_n[0], presence[0], scores[0]):
            if valid and sung and d >= MIN_NOTE:
                notes.append({"start_time": round(onset, 4), "end_time": round(onset + float(d), 4),
                              "score": round(float(score), 2), "velocity": 0.8})
            onset += float(d)
    return notes


def tuning(notes):
    """How far the song sits from A440, in semitones (-0.5 to 0.5): the
    circular mean of every note's pitch past the semitone, by duration.

    A record need not be tuned to A440 — Queen's "Don't Stop Me Now" sang 17
    cents sharp, and rounded as it was a fifth of its notes came out a
    semitone high. Taking the offset off first raised 6 of 27 songs against
    their parts (that one 0.43 -> 0.49) and lowered none.
    """
    import math

    weights = [e["end_time"] - e["start_time"] for e in notes]
    s = sum(w * math.sin(2 * math.pi * e["score"]) for w, e in zip(weights, notes))
    c = sum(w * math.cos(2 * math.pi * e["score"]) for w, e in zip(weights, notes))
    return math.atan2(s, c) / (2 * math.pi) if notes else 0.0


def main():
    if not 3 <= len(sys.argv) <= 5:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], Path(sys.argv[2])
    # model-compose fills an empty optional input with a blank, not "".
    name = (sys.argv[3] if len(sys.argv) > 3 else "").strip() or "small"
    language = (sys.argv[4] if len(sys.argv) > 4 else "").strip()
    import librosa
    import onnxruntime as ort

    started = time.perf_counter()
    # The segmenter samples its boundaries (one RandomUniformLike, unseeded):
    # unseeded, two runs on one stem shared 211 of ~286 notes.
    seed = int(os.environ.get("GAME_SEED", "0"))
    ort.set_seed(seed)
    folder = model_dir(name)
    config = json.loads((folder / "config.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("GAME_THREADS", "0"))   # 0: every core
    sessions = {m: ort.InferenceSession(str(folder / f"{m}.onnx"), options, providers=["CPUExecutionProvider"])
                for m in MODULES}
    language_id = (config.get("languages") or {}).get(language, 0)
    y, sr = sf.read(src, always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    if sr != config["samplerate"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=config["samplerate"])
        sr = config["samplerate"]
    notes = transcribe(y, sr, sessions, language_id)
    offset = tuning(notes)
    notes = [{**e, "pitch": int(round(e["score"] - offset))} for e in notes]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(notes))
    print(json.dumps({"path": str(dst), "model": folder.name, "language": language or None, "seed": seed,
                      "tuning_cents": round(offset * 100),
                      "pieces": len(pieces(y, sr)), "notes": len(notes),
                      "seconds": round(time.perf_counter() - started, 1)}))
    sys.stdout.flush()
    # onnxruntime can abort while tearing down its thread pool on macOS
    # ("recursive_mutex lock failed") after everything is written.
    os._exit(0)


if __name__ == "__main__":
    main()
