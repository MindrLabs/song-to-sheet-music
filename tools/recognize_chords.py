#!/usr/bin/env python3
"""Recognize the song's chords — sevenths included — with BTC, on the beat grid.

Usage: recognize_chords.py <out.json> <audio> <tempo.json> <key.json>

  audio       the mix (output/01-source.flac): BTC was trained on mixes
  tempo.json  the beat tracker's output; chords change on beats and half beats
  key.json    the song's key, for spelling: E-flat, not D-sharp, in a flat key

Writes <out.json> in the form every part reads (see estimate_chords.py):
one entry per chord with start / end (seconds), bar and offset on the shared
bar grid, the Harte label ("C:min7", "G:7", "N") and the symbol to print
("Cm7", "G7", "N.C.").

The model
---------
BTC (Bi-directional Transformer for Chord recognition; Park et al., ISMIR
2019, MIT), its large-vocabulary checkpoint: 12 roots x 14 qualities — major,
minor, diminished, augmented, sixths, sevenths (maj7, 7, m7, m(maj7), dim7,
m7b5), sus2, sus4 — plus no chord. Its repository, weights included (34 MB),
is downloaded on first use into $BTC_CACHE or ~/.cache/btc-chords, pinned to
one commit. It runs on the CPU or a GPU ($DEVICE, else CUDA if present).

Against the chord symbols of purchased parts, 29 songs, share of the song
with the same chord (median over the songs):

                                           root   maj/min   sevenths
  stem chroma + triad templates (before)   0.73   0.67      0.50
  BTC on the mix, as it comes              0.75   0.71      0.55
  BTC on the mix, on our grid (this)       0.76   0.72      0.56

On the grid: each half beat takes the chord BTC gives it longest, and a
chord shorter than a beat is folded into its neighbour (MIN_BAR). A beat
scored better than the half bar the old estimator needed — BTC's short
chords are mostly real ones.
"""
import json
import os
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_chords import SUB, fold_short, names_for, runs, slots  # noqa: E402
from notes_to_score import load_key, make_clock, snap  # noqa: E402

COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
ARCHIVE = f"https://github.com/jayg996/BTC-ISMIR19/archive/{COMMIT}.tar.gz"
MIN_BAR = 0.25       # a chord shorter than a quarter of a bar (a beat in 4/4) is folded away
PITCH = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
         "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
# Harte quality -> what follows the root on the page.
SYMBOL = {"maj": "", "min": "m", "dim": "dim", "aug": "aug", "min6": "m6", "maj6": "6", "min7": "m7",
          "minmaj7": "m(maj7)", "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5",
          "sus2": "sus2", "sus4": "sus4"}


def checkout():
    cache = Path(os.environ.get("BTC_CACHE", Path.home() / ".cache" / "btc-chords"))
    root = cache / f"BTC-ISMIR19-{COMMIT}"
    if not (root / "test" / "btc_model_large_voca.pt").exists():
        cache.mkdir(parents=True, exist_ok=True)
        tgz = cache / f"{COMMIT}.tar.gz"
        urllib.request.urlretrieve(ARCHIVE, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(cache, filter="data")
        tgz.unlink()
    return root


def load_model(root):
    import torch
    import yaml

    # BTC's code predates PyYAML 6 (yaml.load without a Loader) and NumPy 1.24
    # (np.float and friends); both are patched here rather than in its files.
    plain_load = yaml.load
    yaml.load = lambda stream, Loader=yaml.SafeLoader: plain_load(stream, Loader=Loader)
    for name, kind in (("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)):
        if name not in np.__dict__:
            setattr(np, name, kind)
    sys.path.insert(0, str(root))
    from btc_model import BTC_model
    from utils.hparams import HParams
    from utils.mir_eval_modules import audio_file_to_features, idx2voca_chord

    config = HParams.load(str(root / "run_config.yaml"))
    config.feature["large_voca"] = True
    config.model["num_chords"] = 170
    wanted = os.environ.get("DEVICE", "").strip()
    device = torch.device(wanted if wanted else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BTC_model(config=config.model).to(device)
    ck = torch.load(root / "test" / "btc_model_large_voca.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return lambda path: recognize(path, config, model, ck["mean"], ck["std"], device,
                                  audio_file_to_features, idx2voca_chord()), device


def recognize(path, config, model, mean, std, device, features, names):
    """[{start, end, label}] in seconds, BTC's frames merged into runs."""
    import torch

    feature, step, _ = features(path, config)            # step: seconds per frame
    feature = (feature.T - mean) / std
    n = config.model["timestep"]
    pad = n - feature.shape[0] % n
    x = torch.tensor(np.pad(feature, ((0, pad), (0, 0))), dtype=torch.float32).unsqueeze(0).to(device)
    picks = []
    with torch.no_grad():
        for t in range(x.shape[1] // n):
            out, _ = model.self_attn_layers(x[:, n * t:n * (t + 1), :])
            p, _ = model.output_layer(out)
            picks += p.squeeze().tolist()
    picks = picks[:len(picks) - pad]
    segs = []
    for i, c in enumerate(picks):
        if segs and segs[-1]["label"] == names[c]:
            segs[-1]["end"] = (i + 1) * step
        else:
            segs.append({"start": i * step, "end": (i + 1) * step, "label": names[c]})
    return segs


def parse(label):
    """(root pitch class, quality) or None for no chord."""
    if label in ("N", "X"):
        return None
    root, _, quality = label.partition(":")
    return PITCH[root], quality or "maj"


def on_grid(segs, clock):
    """Each half beat gets the label BTC gives it longest; then runs, short ones folded."""
    edges = slots(clock)
    picks = []
    for a, b in zip(edges, edges[1:]):
        weight = defaultdict(float)
        for s in segs:
            overlap = min(b, s["end"]) - max(a, s["start"])
            if overlap > 0:
                weight[s["label"]] += overlap
        picks.append(parse(max(weight, key=weight.get)) if weight else None)
    return fold_short(runs(edges, picks), clock, MIN_BAR)


def describe(start, end, pick, clock, names):
    q = clock.quarters(start) - clock.bar_phase
    bar, offset = divmod(snap(q, SUB), clock.bar_length)
    entry = {"start": round(float(start), 3), "end": round(float(end), 3), "bar": int(bar) + 1, "offset": float(offset)}
    if pick is None:
        return {**entry, "label": "N", "symbol": "N.C."}
    root, quality = pick
    return {**entry, "label": f"{names[root]}:{quality}", "symbol": names[root] + SYMBOL[quality]}


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    dst, audio, tempo_path, key_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    started = time.perf_counter()
    clock, mode = make_clock("auto", tempo_path)
    k = load_key(key_path or None)
    names = names_for(k)
    run, device = load_model(checkout())
    segs = run(audio)
    chords = [describe(a, b, p, clock, names) for a, b, p in on_grid(segs, clock)]
    result = {
        "chords": chords,
        "source": f"BTC large vocabulary ({COMMIT[:7]}) on the mix, on the half-beat grid, chords of a beat or more",
        "key": f"{k.tonic.name.replace('-', 'b')} {k.mode}" if k is not None else None,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    played = [c for c in chords if c["label"] != "N"]
    symbols = sorted({c["symbol"] for c in played}, key=lambda s: -sum(c["end"] - c["start"] for c in played if c["symbol"] == s))
    print(json.dumps({
        "path": str(dst),
        "tempo_source": mode,
        "key": result["key"],
        "device": str(device),
        "chords": len(played),
        "no_chord_spans": len(chords) - len(played),
        "symbols": symbols,
        "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
