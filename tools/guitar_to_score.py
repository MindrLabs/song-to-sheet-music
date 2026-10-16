#!/usr/bin/env python3
"""Write a guitar part — chords kept — from a transcriber's note events.

Usage: guitar_to_score.py <events.json> <out.musicxml> <tempo.json> [grid] [audio] [key.json] [chords.json] [--title=SONG]
                          [--tab=false]

  events.json  basic-pitch notes (start_time / end_time / pitch / velocity),
               as the transcribe-guitar workflow saves them
  tempo.json   the beat tracker's output; same bar rule as every other part
  grid         subdivisions per quarter note; 4 = 16ths (default)
  audio        the guitar stem. With it, notes read off a near-silent stem are
               dropped (as for the bass)
  key.json     the song's key (tools/estimate_key.py)
  chords.json  the song's chords (tools/estimate_chords.py), printed above the
               staff. A strummed part is read from these more than from its
               notes, which a note transcriber gets least right on exactly
               those bars

Also writes <out>.mid, for listening.

Unlike the bass, a guitar strums chords, so every pitch that starts on one
grid position stays, as a chord. It is written on a treble clef with an 8
below, the guitar's convention: the page reads an octave above the sound.
One stem carries every guitar in the song, so the part is all of them.

Under the staff, the same notes as tablature — a string and fret for each
(tools/tab.py), one part with two staves as notation programs show a guitar
with tab. A guitarist who checked these parts asked for it. --tab=false leaves
it out (some practise without one).
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_to_score import GATE_DB, Stem, onset_latency  # noqa: E402
from notes_to_score import (bar_origin, chord_symbols, chords_end, load_key, make_clock, metronome, notate, place_symbols, pop_tab, pop_title,  # noqa: E402
                            set_title, skip_part, snap, spelled, write_part)
import tab  # noqa: E402

# Sounding range: a drop-D or half-step-down low string (D2) up to the 24th
# fret of the high string (E6), with a semitone of slack either side.
LOW, HIGH = 37, 89


def pick_chords(events, clock, grid, latency):
    """(onset, [pitches], [events]) per grid position, low to high."""
    groups = defaultdict(list)
    for e in events:
        if LOW <= e["pitch"] <= HIGH:
            groups[snap(clock.quarters(e["start_time"] - latency), grid)].append(e)
    return [(onset, sorted({e["pitch"] for e in evs}), evs) for onset, evs in sorted(groups.items())]


def gate(chords, stem):
    """Drop chords read off a stem that is all but silent there."""
    if stem is None or not chords:
        return chords, 0
    levels = [stem.rms_db(min(e["start_time"] for e in evs)) for _, _, evs in chords]
    floor = float(np.median(levels)) - GATE_DB
    kept = [c for c, lv in zip(chords, levels) if lv >= floor]
    return kept, len(chords) - len(kept)


def build(chords, clock, grid, latency, k, song_chords=(), with_tab=True):
    from music21 import chord, clef, instrument, key, meter, note, stream, tempo

    part = stream.Part()
    gtr = instrument.ElectricGuitar()
    gtr.partName, gtr.partAbbreviation = "Guitar", "Gtr."
    part.insert(0, gtr)
    part.insert(0, clef.Treble8vbClef())
    if k is not None:
        part.insert(0, key.KeySignature(k.sharps))
    part.insert(0, meter.TimeSignature(clock.time_signature))
    part.insert(0, metronome(clock))

    shift, pickup_bars = bar_origin(clock, chords[0][0] if chords else None, grid)
    for i, (onset, pitches, evs) in enumerate(chords):
        own = snap(clock.quarters(max(e["end_time"] for e in evs) - latency), grid) - onset
        nxt = chords[i + 1][0] - onset if i + 1 < len(chords) else own
        named = [spelled(p, k) for p in pitches] if k is not None else pitches
        el = note.Note(named[0]) if len(named) == 1 else chord.Chord(named)
        el.quarterLength = max(Fraction(1, grid), min(own, nxt))
        part.insert(float(onset + shift), el)

    notated = notate(part, pickup_bars, chords_end(song_chords, clock, grid, shift))
    place_symbols(notated, chord_symbols(song_chords, clock, grid, shift), grid)

    if with_tab:
        return tab.with_tab_staff(notated, "Guitar", "Gtr."), pickup_bars
    score = stream.Score()
    score.insert(0, notated)
    return score, pickup_bars


def write_midi(chords, path):
    try:
        import pretty_midi
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=29, name="Guitar")   # GM 30, overdriven guitar
    for i, (_, _, evs) in enumerate(chords):
        cut = chords[i + 1][2][0]["start_time"] if i + 1 < len(chords) else None
        for e in evs:
            end = e["end_time"] if cut is None else min(e["end_time"], max(cut, e["start_time"] + 0.05))
            track.notes.append(pretty_midi.Note(velocity=int(np.clip(e.get("velocity", 0.7) * 127, 30, 127)),
                                                pitch=int(e["pitch"]), start=e["start_time"], end=end))
    pm.instruments.append(track)
    pm.write(str(path))
    return str(path)


def main():
    title = pop_title(sys.argv)
    with_tab = pop_tab(sys.argv)
    if not 4 <= len(sys.argv) <= 8:
        raise SystemExit(__doc__)
    src, dst, tempo_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    grid = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    audio = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    k = load_key(sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None)
    chords_path = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None
    song_chords = json.loads(Path(chords_path).read_text())["chords"] if chords_path else []

    clock, mode = make_clock("auto", tempo_path)
    events = json.loads(src.read_text())
    in_range = [e for e in events if LOW <= e["pitch"] <= HIGH]
    latency, latency_r = onset_latency(in_range, clock, grid)
    chords = pick_chords(events, clock, grid, latency)
    chords, gated = gate(chords, Stem(audio) if audio else None)
    if not chords:
        print(json.dumps({**skip_part(dst, "No guitar part: the guitar stem gave no notes "
                                           "(most likely the song has no guitar).", [dst.with_suffix(".mid")]),
                          "events_in": len(events), "gated_quiet": gated}))
        return
    score, pickup_bars = build(chords, clock, grid, latency, k, song_chords, with_tab)
    set_title(score, "Guitar", title)
    write_part(score, dst)
    tabbed = tab.add_tab(dst) if with_tab else None
    midi = write_midi(chords, dst.with_suffix(".mid"))

    sizes = [len(p) for _, p, _ in chords]
    print(json.dumps({
        "path": str(dst),
        "midi": midi,
        "tempo_source": mode,
        "bpm": round(clock.bpm, 3),
        "time_signature": clock.time_signature,
        "key": f"{k.tonic.name.replace('-', 'b')} {k.mode}" if k is not None else None,
        "pickup_bars": pickup_bars,
        "grid": f"1/{grid} quarter",
        "onset_latency_ms": round(latency * 1000, 1),
        "onset_phase_r": round(latency_r, 3),
        "events_in": len(events),
        "out_of_range": len(events) - len(in_range),
        "onsets": len(chords),
        "gated_quiet": gated,
        "chords": sum(1 for n in sizes if n > 1),
        "chord_symbols": len(chord_symbols(song_chords, clock, grid, 0)) if song_chords else 0,
        "tab": tabbed,
        "measures": len(score.parts[0].getElementsByClass("Measure")),
    }))


if __name__ == "__main__":
    main()
