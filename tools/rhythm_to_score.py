#!/usr/bin/env python3
"""Write the guitar's rhythm chart: the song's chord symbols over rhythm slashes.

Usage: rhythm_to_score.py <guitar.wav> <out.musicxml> <tempo.json> [key.json] [chords.json] [--title=SONG]

  guitar.wav   the guitar stem; each slash is a strum read off its attacks
  tempo.json   the beat tracker's output; same bar rule as every other part
  key.json     the song's key, for the key signature
  chords.json  the song's chords (tools/recognize_chords.py), printed above

Also writes <out>.mid (a click on each strum, for listening).

Why a second guitar part
------------------------
The guitar part (guitar_to_score.py) writes the notes a transcriber hears,
and on a strummed, distorted guitar it hears few of them right. Purchased
parts write such a passage the other way: the chord's name, and slashes for
the rhythm. Against the slashes of the three purchased parts that do (145
bars), F of strums matched within a 16th (median of songs):

  the guitar part's note onsets            0.41
  a slash on every 8th note                0.91
  the stem's attacks on 8th notes (this)   0.93

Telling strummed bars from riffs, to write each its own way in one part, did
not work: one stem holds every guitar of the song while the parts write
them apart. So both parts are written, the reader takes the one that fits.

How it reads the stem
---------------------
Onsets of the stem's percussive part (librosa: HPSS, onset strength), on the
bar grid in 8th notes — 16ths and stronger-onsets-only both scored lower. A
strum on a stem far under its typical level (GATE_DB) is bleed and dropped.
Slashes sit on the middle line; lengths and rests split at the beat, as on
the drum part, so a bar reads beat by beat.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_to_score import GATE_DB  # noqa: E402
from notes_to_score import (bar_origin, chord_symbols, chords_end, load_key, make_clock, metronome, notate,  # noqa: E402
                            place_symbols, pop_title, rests, set_title, skip_part, snap, write_part)

SR, HOP = 22050, 512
GRID = 2            # strums on 8th notes
PLAIN = {Fraction(n, 4) for n in (1, 2, 3, 4, 6, 8, 12)}


def strums(path):
    """[(seconds, level dB)] of the stem's attacks."""
    import librosa

    y = librosa.load(path, sr=SR, mono=True)[0]
    perc = librosa.effects.percussive(y, margin=2.0)
    env = librosa.onset.onset_strength(y=perc, sr=SR, hop_length=HOP)
    times = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="time")
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    level = lambda t: float(20 * np.log10(rms[min(len(rms) - 1, int(t * SR / HOP)):][:5].max() + 1e-9))
    return [(float(t), level(t)) for t in times]


def on_grid(found, clock):
    """Sorted strum positions in quarters, on 8ths; bleed under GATE_DB dropped."""
    if not found:
        return [], 0
    floor = float(np.median([lv for _, lv in found])) - GATE_DB
    kept = [t for t, lv in found if lv >= floor]
    return sorted({snap(clock.quarters(t), GRID) for t in kept}), len(found) - len(kept)


def build(onsets, clock, k, song_chords=()):
    from music21 import clef, instrument, key, meter, note, stream

    part = stream.Part()
    gtr = instrument.ElectricGuitar()
    gtr.partName, gtr.partAbbreviation = "Guitar (rhythm)", "Gtr."
    part.insert(0, gtr)
    part.insert(0, clef.TrebleClef())
    if k is not None:
        part.insert(0, key.KeySignature(k.sharps))
    part.insert(0, meter.TimeSignature(clock.time_signature))
    part.insert(0, metronome(clock))

    # The middle line's B as the key signature has it (B-flat under flats), or
    # every bar's first slash carries a natural.
    altered = key.KeySignature(k.sharps).accidentalByStep("B") if k is not None else None
    middle = "B" + (altered.modifier if altered is not None else "") + "4"
    shift, pickup_bars = bar_origin(clock, onsets[0] if onsets else None, GRID)
    beat, pulse, L = clock.beat_length, clock.pulse, clock.bar_length
    free = Fraction(0)
    for i, onset in enumerate(onsets):
        at = onset + shift
        nxt = onsets[i + 1] - onset if i + 1 < len(onsets) else beat
        # As on the drum part: a strum's slash lasts to the next one but not
        # past the beat — a tie would read as a second strum.
        length = max(Fraction(1, GRID), min(nxt, beat - at % beat if at % beat else beat,
                                             pulse - at % pulse if at % pulse else beat))
        while length > Fraction(1, GRID) and length not in PLAIN:
            length -= Fraction(1, GRID)
        for r_at, r_len in rests(free, at, L, beat, pulse):
            part.insert(float(r_at), note.Rest(quarterLength=r_len))
        n = note.Note(middle, quarterLength=length)
        n.notehead = "slash"
        n.stemDirection = "up"
        part.insert(float(at), n)
        free = at + length

    notated = notate(part, pickup_bars, chords_end(song_chords, clock, GRID * 2, shift))
    place_symbols(notated, chord_symbols(song_chords, clock, GRID * 2, shift), GRID * 2)
    score = stream.Score()
    score.insert(0, notated)
    return score, pickup_bars


def write_midi(found, path):
    try:
        import pretty_midi
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0, is_drum=True, name="Strums")
    for t in found:
        track.notes.append(pretty_midi.Note(velocity=90, pitch=37, start=t, end=t + 0.05))   # side stick
    pm.instruments.append(track)
    pm.write(str(path))
    return str(path)


def main():
    title = pop_title(sys.argv)
    if not 4 <= len(sys.argv) <= 6:
        raise SystemExit(__doc__)
    audio, dst, tempo_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    k = load_key(sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None)
    chords_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    song_chords = json.loads(Path(chords_path).read_text())["chords"] if chords_path else []

    clock, mode = make_clock("auto", tempo_path)
    found = strums(audio)
    onsets, gated = on_grid(found, clock)
    if not onsets:
        print(json.dumps({**skip_part(dst, "No guitar rhythm chart: no strums in the guitar stem "
                                           "(most likely the song has no guitar).", [dst.with_suffix(".mid")]),
                          "attacks": len(found), "gated_quiet": gated}))
        return
    score, pickup_bars = build(onsets, clock, k, song_chords)
    set_title(score, "Guitar (rhythm)", title)
    write_part(score, dst)
    midi = write_midi([t for t, _ in found], dst.with_suffix(".mid"))
    print(json.dumps({
        "path": str(dst),
        "midi": midi,
        "tempo_source": mode,
        "bpm": round(clock.bpm, 3),
        "time_signature": clock.time_signature,
        "key": f"{k.tonic.name.replace('-', 'b')} {k.mode}" if k is not None else None,
        "pickup_bars": pickup_bars,
        "grid": f"1/{GRID} quarter",
        "attacks": len(found),
        "gated_quiet": gated,
        "strums": len(onsets),
        "chord_symbols": len(chord_symbols(song_chords, clock, GRID * 2, 0)) if song_chords else 0,
        "measures": len(score.parts[0].getElementsByClass("Measure")),
    }))


if __name__ == "__main__":
    main()
