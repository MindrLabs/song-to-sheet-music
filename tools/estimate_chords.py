#!/usr/bin/env python3
"""Estimate the song's chords once, on the beat grid, for every part to share.

Superseded in the pipeline by tools/recognize_chords.py (BTC, sevenths
included), which agrees better with the purchased parts on 29 songs; kept for
the grid helpers both share and as an estimator that downloads nothing.

Usage: estimate_chords.py <out.json> <tempo.json> <key.json> <bass.wav> <vocals.wav> [stem.wav ...]

  tempo.json  the beat tracker's output; chords change on beats and half-beats
  key.json    the song's key (tools/estimate_key.py), for spelling: E-flat,
              not D-sharp, in a flat key. "" to spell without one
  bass.wav    the bass stem — its lowest note is taken as the chord's root
  vocals.wav  the vocal stem, or "". It helps name the chord (a sung line
              leans on its tones) but never makes one: a line sung over a
              silent band is "no chord", as the scores write it
  stem.wav    every other pitched stem: all of them but the drums

Writes <out.json> and prints a summary:
  chords   one entry per chord: start / end (seconds), bar and offset (in
           quarters) on the same bar grid as every part, the Harte label
           ("C:min", "Eb:maj", "N" for no chord) and the symbol to print
           ("Cm", "Eb", "N.C.")

Why a song-level chord track
----------------------------
Chord symbols belong to the song, like the key: the purchased parts of both
test songs print the same symbols over guitar, keys and bass. And they are
what a guitarist reads for a strummed part — a note-level transcriber got 8%
of the notes of a distorted barre-chord verse, where the chord itself is
plain in the whole band's sound.

Method
------
Constant-Q chroma of all the given stems summed, averaged over each half
beat; each half beat scored against the 24 major and minor triads (cosine)
plus a bonus for the chord whose root the bass stem is playing; a Viterbi
pass over the song keeps chords from flickering. Half beats where the band
(everything but the vocals) is well below its median level are "no chord".
Last, a chord held for less than two beats is folded into the one before it:
on both test songs those were mostly wrong (root +0.03 / +0.02 without them)
and they crowd the symbols. A song whose harmony moves every beat loses
those changes — a trade to recheck with more songs.

Scored against the chord symbols of the purchased parts of two songs (a
reference, not the truth — sellers write them down by ear), share of the
song with the same chord, as song A / song B:

                                    chord (maj/min)   root
  chroma of the stems, no bass bonus    0.70 / 0.44   0.78 / 0.79
  + bass root bonus (0.3)               0.77 / 0.55   0.85 / 0.91
  + vocals in the chroma (this)         0.75 / 0.64   0.85 / 0.90

Two songs are too few to tune on: STAY anywhere in 0.5-0.9 and a bass bonus
of 0.3-0.6 scored within 0.03 of each other, and two songs cannot choose
inside that range. More songs will.

Where the two disagree it is mostly the third, and there the recordings
themselves are ambiguous: a distorted guitar's fifth harmonic sounds the
major third over a root, and one song's F chord has almost no third at all.
Those are for ears to settle, not this script.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notes_to_score import load_key, make_clock, snap  # noqa: E402

SR, HOP = 22050, 512
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PLAIN_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QUALITIES = {"maj": ((0, 4, 7), ""), "min": ((0, 3, 7), "m")}
SUB = 2              # chord slots per beat: changes on beats and half beats
STAY = 0.9           # per beat: how strongly a chord holds from one beat to the next
BASS_WEIGHT = 0.3    # bonus for the chord whose root the bass plays (0-1)
SHARPNESS = 20.0     # how decisively each half beat's score picks a chord
QUIET_DB = -35.0     # no chord below the song's median level by this much
MIN_BAR = 0.5        # a chord shorter than half a bar is folded into the one before it
                     # (two beats of 4/4; one dotted-quarter beat of 6/8)


def names_for(k):
    if k is None:
        return PLAIN_NAMES
    return SHARP_NAMES if k.sharps > 0 else FLAT_NAMES if k.sharps < 0 else PLAIN_NAMES


def templates():
    labels, rows = [], []
    for root in range(12):
        for quality, (steps, _) in QUALITIES.items():
            t = np.zeros(12)
            t[[(root + s) % 12 for s in steps]] = 1
            labels.append((root, quality))
            rows.append(t / np.linalg.norm(t))
    return labels, np.array(rows)


def slots(clock):
    """Half-beat boundaries, in seconds."""
    t = np.asarray(clock.times, dtype=float)
    inner = [np.linspace(a, b, SUB, endpoint=False) for a, b in zip(t, t[1:])]
    return np.concatenate(inner + [t[-1:]])


def per_slot(matrix, frames):
    last = matrix.shape[1] - 1
    return np.array([matrix[:, a:b].mean(axis=1) if b > a else matrix[:, min(a, last)]
                     for a, b in zip(frames, frames[1:])])


def viterbi(logp, stay):
    k = logp.shape[1]
    trans = np.full((k, k), np.log((1 - stay) / (k - 1)))
    np.fill_diagonal(trans, np.log(stay))
    score, back = logp[0].copy(), []
    for row in logp[1:]:
        m = score[:, None] + trans
        back.append(m.argmax(axis=0))
        score = m.max(axis=0) + row
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]


def estimate(bass_path, vocals_path, stem_paths, clock):
    import librosa

    load = lambda p: librosa.load(p, sr=SR, mono=True)[0]
    ys = [load(p) for p in [bass_path, *stem_paths]]
    vocals = load(vocals_path) if vocals_path else None
    n = min(len(y) for y in ys + ([vocals] if vocals is not None else []))
    band = sum(y[:n] for y in ys)
    mix = band + vocals[:n] if vocals is not None else band
    chroma = librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=HOP, bins_per_octave=36)
    # The bass stem's own chroma, B0 up three octaves: which pitch class the
    # bass is on, i.e. the root, in all but inverted chords.
    bass = librosa.feature.chroma_cqt(y=ys[0][:n], sr=SR, hop_length=HOP, bins_per_octave=36,
                                      fmin=librosa.midi_to_hz(23), n_octaves=3)
    rms = librosa.feature.rms(y=band, hop_length=HOP)[0]

    edges = slots(clock)
    frames = librosa.time_to_frames(edges, sr=SR, hop_length=HOP)
    x, b = per_slot(chroma, frames), per_slot(bass, frames)
    level = np.array([20 * np.log10(rms[a:max(c, a + 1)].mean() + 1e-9) for a, c in zip(frames, frames[1:])])

    labels, tmpl = templates()
    roots = np.array([r for r, _ in labels])
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    b = b / (b.max(axis=1, keepdims=True) + 1e-9)
    fit = x @ tmpl.T + BASS_WEIGHT * b[:, roots]

    quiet = level < np.median(level) + QUIET_DB
    logp = np.where(quiet[:, None], -np.inf, SHARPNESS * fit)
    logp = np.hstack([logp, np.where(quiet, 0.0, -np.inf)[:, None]])     # last column: no chord
    path = viterbi(logp, STAY ** (1 / SUB))
    return edges, [labels[i] if i < len(labels) else None for i in path]


def runs(edges, picks):
    out = []
    for i, pick in enumerate(picks):
        if out and out[-1][2] == pick:
            out[-1][1] = edges[i + 1]
        else:
            out.append([edges[i], edges[i + 1], pick])
    return out


def fold_short(spans, clock, min_bar=MIN_BAR):
    """Fold chords shorter than `min_bar` of a bar into their neighbour, shortest first."""
    min_beats = min_bar * clock.bar_length                 # in quarters, as clock.quarters counts
    spans = [list(s) for s in spans]
    beats = lambda s: clock.quarters(s[1]) - clock.quarters(s[0])
    while len(spans) > 1:
        i = min(range(len(spans)), key=lambda j: beats(spans[j]))
        if beats(spans[i]) >= min_beats - 1e-6:
            break
        j = i - 1 if i > 0 else 1                  # the chord before it, or after for the first
        lo, hi = min(i, j), max(i, j)
        keep = spans[j][2]
        spans[lo:hi + 1] = [[spans[lo][0], spans[hi][1], keep]]
        for k in (lo - 1, lo):                     # neighbours that now match
            if 0 <= k < len(spans) - 1 and spans[k][2] == spans[k + 1][2]:
                spans[k:k + 2] = [[spans[k][0], spans[k + 1][1], spans[k][2]]]
    return spans


def describe(start, end, pick, clock, names):
    q = clock.quarters(start) - clock.bar_phase
    bar, offset = divmod(snap(q, SUB), clock.bar_length)
    entry = {"start": round(float(start), 3), "end": round(float(end), 3),
             "bar": int(bar) + 1, "offset": float(offset)}
    if pick is None:
        return {**entry, "label": "N", "symbol": "N.C."}
    root, quality = pick
    name = names[root]
    return {**entry, "label": f"{name}:{quality}", "symbol": name + QUALITIES[quality][1]}


def main():
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    dst, tempo_path, key_path, bass_path, vocals_path = Path(sys.argv[1]), *sys.argv[2:6]
    stems = [p for p in sys.argv[6:] if p]
    clock, mode = make_clock("auto", tempo_path)
    k = load_key(key_path or None)
    names = names_for(k)

    edges, picks = estimate(bass_path, vocals_path, stems, clock)
    chords = [describe(a, b, p, clock, names) for a, b, p in fold_short(runs(edges, picks), clock)]
    result = {
        "chords": chords,
        "source": "estimated: constant-Q chroma of the stems + bass-stem root, "
                  "major/minor triads, Viterbi on half beats",
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
        "chords": len(played),
        "no_chord_spans": len(chords) - len(played),
        "symbols": symbols,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
