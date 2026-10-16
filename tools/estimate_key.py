#!/usr/bin/env python3
"""Estimate the song's key once, for every part to share.

Usage: estimate_key.py <audio> <out.json> [key]

  audio  the full mix
  key    `auto` (default) to estimate, or the key itself — "C minor", "Ab",
         "F# major" — when you know it better than the estimate does

Writes <out.json> and prints the same JSON:
  tonic, mode, name       e.g. "Eb", "minor", "Eb minor"
  fifths                  the key signature: flats negative, sharps positive
  confidence              winning correlation minus the best key with a
                          different signature (a relative key does not
                          count as a rival: it prints the same signature)
  low_confidence          confidence under 0.1 — worth checking by ear
  alternatives            the next keys with other signatures

Why a song-level key
--------------------
A key signature belongs to the song, not to a part: bass, piano and guitar
parts of one song must print the same one. Estimated from the bass line
alone, one test song came out F minor (4 flats) where its parts are written
in C minor (3 flats) — the bass sits on F about as much as on C. So the key
is read once from the mix and saved next to tempo.json.

Method
------
librosa's harmonic part of the mix (percussion taken out by HPSS), constant-Q
chroma averaged over the song, correlated with every rotation of the
Krumhansl-Kessler major and minor profiles. Tried on the two songs whose
parts we have: this got both signatures right. The full mix with the same
profile got the Dorian-coloured one wrong (F minor), as did Albrecht-
Shanahan's pop profiles on every input; model-compose's music-analyzer (CENS
chroma on the mix) called it C major. Weighting the tonic by the bass stem's
root distribution did not help. That song's margin stayed small (0.05) under
every method — hence `confidence`, and the override.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
# Key signature of each major tonic (pitch class); a minor key takes its
# relative major's. F# major (6 sharps) rather than G-flat (6 flats).
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LOW_CONFIDENCE = 0.1


def fifths(tonic, mode):
    return MAJOR_FIFTHS[tonic if mode == "major" else (tonic + 3) % 12]


def tonic_name(tonic, mode):
    """Spell the tonic the way its own signature does (A-flat, not G-sharp)."""
    return (SHARP_NAMES if fifths(tonic, mode) > 0 else FLAT_NAMES)[tonic]


def describe(tonic, mode, **extra):
    name = tonic_name(tonic, mode)
    return {"tonic": name, "mode": mode, "name": f"{name} {mode}", "fifths": fifths(tonic, mode), **extra}


def parse_key(text):
    """ "C minor", "Ab", "f# major", "Bbm" -> (pitch class, mode)."""
    m = re.fullmatch(r"\s*([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m)?\s*", text)
    if not m:
        raise SystemExit(f"cannot read key {text!r}; try e.g. 'C minor' or 'Ab major'")
    pc = (LETTERS[m.group(1).upper()] + {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(m.group(2), 0)) % 12
    mode = "minor" if (m.group(3) or "").lower() in ("minor", "min", "m") else "major"
    return pc, mode


def estimate(path):
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    harmonic = librosa.effects.harmonic(y, margin=2.0)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=2048).mean(axis=1)
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            scores.append((float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]), tonic, mode))
    scores.sort(reverse=True)
    best, tonic, mode = scores[0]
    rivals, seen = [], {fifths(tonic, mode)}
    for score, t, m in scores[1:]:
        if fifths(t, m) not in seen:
            seen.add(fifths(t, m))
            rivals.append((score, t, m))
    confidence = best - rivals[0][0]
    return describe(
        tonic, mode,
        correlation=round(best, 3),
        confidence=round(confidence, 3),
        low_confidence=confidence < LOW_CONFIDENCE,
        alternatives=[describe(t, m, correlation=round(s, 3)) for s, t, m in rivals[:3]],
        source="estimated: HPSS harmonic part, chroma_cqt, Krumhansl-Kessler",
    )


def main():
    if not 3 <= len(sys.argv) <= 4:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], Path(sys.argv[2])
    given = sys.argv[3] if len(sys.argv) > 3 else "auto"
    if given.strip().lower() in ("", "auto"):
        result = estimate(src)
    else:
        result = describe(*parse_key(given), source="given")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
