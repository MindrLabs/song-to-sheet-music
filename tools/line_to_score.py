#!/usr/bin/env python3
"""Write a single-line part — the bass, or the sung melody — from a
transcriber's note events.

Usage: line_to_score.py <events.json> <out.musicxml> <tempo.json> <part> [grid] [audio] [key.json] [--chords=PATH]
                        [--lyrics=PATH] [--title=SONG] [--tab=false]

  events.json  notes (start_time / end_time / pitch / velocity, in seconds):
               basic-pitch's for the bass, as the transcribe-bass workflow
               saves them; tools/transcribe_vocals.py's for the voice
  tempo.json   the beat tracker's output; bar lines come from the same rule
               every other part uses, so bar numbers agree across parts
  part         `bass` or `voice`
  grid         subdivisions per quarter note; 4 = 16ths (default)
  audio        the stem the events came from. With it, each note's octave is
               checked against the stem's spectrum (see below); without it the
               transcriber's pitches are kept as they are
  key.json     the song's key (tools/estimate_key.py), shared with the other
               parts. Without it the key is guessed from this part's notes,
               which a bass line alone gets wrong (F minor for a C minor song)
  --chords     the song's chords (tools/estimate_chords.py), printed above the
               staff — with the voice part, a lead sheet
  --lyrics     the sung words with their times (the lyrics-transcriber component),
               set under the notes, one syllable to a note (tools/lyrics.py)

The bass part carries a TAB staff under the notes, as the guitar part does
(tools/tab.py): four strings, five with a low B where the line goes below a
4-string's E (band parts are often 5-string lines rearranged — the line is not
moved for it). --tab=false leaves it out; the voice never has one.

Also writes <out>.mid at sounding pitch, for listening.

The voice
---------
The sung melody comes from GAME (tools/transcribe_vocals.py), which already
decides where notes start and end and whether a stretch is sung at all, so
none of the bass's checks against the stem apply. It is written on a treble
clef, with an 8 below when that puts fewer notes on ledger lines — a man's
voice, which lead sheets print an octave up; the clef says so. Rests are
split at beats as on the drum part: a melody's rhythm is read beat by beat.

The bass
--------
Numbers below are from one song (a K-pop track with an overdriven bass), measured against a
purchased bass part corrected by a bassist's ear: the part says it was
rearranged for a 4-string bass, and the record does play D-flat, E-flat and E
an octave lower (D-flat1, E-flat1, E1). The part is a reference, not the
truth, so "agrees" is the claim, never "is correct".

One line, one note at a time
----------------------------
A bass line is a single voice, but a transcriber reports what it hears, and
it hears overtones: basic-pitch gave two or three pitches at one onset 99
times in 680. At each onset only the lowest pitch is kept, and a note is cut
where the next one starts. Notes read off a stem that is all but silent are
dropped (GATE_DB): where the bass rests, basic-pitch still found 56 notes.

The octave check
----------------
Low notes come back an octave or an octave and a fifth high: basic-pitch
reports the second or third harmonic when the fundamental is weak, and on
this stem the lowest octave is filtered — D-flat1's fundamental sat 20 to 35
dB under its octave, while its third harmonic, A-flat2, was only 1 to 3 dB
under. A-flat2 looked like a fifth over D-flat2 until a bassist heard D-flat1.

So a lower candidate replaces the reported pitch when its octave partial is
present and either its fundamental (FUND_DB) or its third harmonic (THIRD_DB)
is too. The third harmonic is the test that separates octaves: it belongs to
the lower note's series and not the upper one's, and near 100 Hz a 0.3 s
window resolves it where 35 Hz is not resolved at all. Candidates go down to
B0, a 5- or 6-string's low string. pYIN's own f0 did worse than doing nothing
(61% vs 67% agreeing).

Then a run of the same note name takes one octave (even_octaves): deciding
note by note left 22 bars with one note in two octaves, which a bassist
reading the part called the real problem; after it, none.

Onset latency
-------------
basic-pitch's onsets sit about 48 ms behind the beat grid (the drum part,
from onsets the stem itself shows, needs no such shift). Snapped as they
are, half-way to the next sixteenth, a straight eighth turns into dotted
rhythms. The lag is measured per run from the events alone — the circular
mean of their phase within a grid cell — and taken off before snapping. On
the test song that agreed with the reference's own timing to 2 ms.

Repeated notes
--------------
basic-pitch merges a run of the same pitch into one long note: 92 of the 166
reference notes it missed were inside a longer note of that pitch. And it
adds notes that are not there, on the 16ths between eighths: the octave
overtone heard alone, or one eighth reported as two 16ths. The stem's level
decides both (check_attacks): a note is split at an eighth where the level
steps up, and an unattacked 16th of those two kinds goes.

Together, against the corrected reference: notes within a sixteenth 81% ->
85%, on the exact sixteenth 60% -> 79%, same pitch 55% -> 81% (verse 39% -> 88%),
notes where the bass rests 56 -> 6. Rhythm is the weak part: the chorus's
repeated eighths on an overdriven bass come back as one to three notes a bar.

A second song with a clean-toned bass (a Korean rock song)
shows how much of that is the tone: basic-pitch alone put 99% of the notes
within a sixteenth and 86% on the right pitch, its onsets lagged 1.8 ms, not
48. There the octave check costs a little (86% -> 84%) where it gained 25
points on the overdriven song; it stays, and even_octaves was narrowed so it
no longer flattens that song's C2-C3 leaps.
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notes_to_score import (bar_origin, chord_symbols, chords_end, load_key, make_clock, metronome, notate,  # noqa: E402
                            place_symbols, pop_chords, pop_tab, pop_title, rests, set_title, skip_part, snap, spelled,
                            write_part)
import lyrics  # noqa: E402
import tab  # noqa: E402

PROFILES = {
    # Bass guitar is written an octave above where it sounds. Range: a 5- or
    # 6-string's low B0 up to about G4, in sounding MIDI numbers, and the octave
    # check may move a note anywhere in it. Do not narrow this to a 4-string's
    # E1 to agree with a published part: band parts are often rearranged for 4
    # strings (the test song's says so), so a lower note is not an error.
    # Ends on 8ths, as the voice's (below): a note let go a 16th early came out
    # as a 16th and a 16th rest, again and again (16th rests on the clean-toned
    # test song 57 -> 2, the overdriven one 110 -> 31; ties 4 -> 15 and 1 -> 2).
    "bass": dict(name="Electric Bass", label="Bass", abbreviation="Bass", instrument="ElectricBass", program=33,
                 clef="bass", written_shift=12, low=23, high=67, stem_checks=True, check_octave=True, octave_floor=23,
                 beat_rests=False, end_grid=2, tab=True),
    # A singer: C2 to C7 covers a bass's low notes and a soprano's whistle
    # tones. Written where it sounds; the clef gets an 8 below for a low line.
    # Notes start on 16ths but end on 8ths (end_grid, per quarter): a sung
    # note held a 16th past the beat came out as a tied 16th and a dotted
    # 8th rest, again and again. A shorter gap than that before the next
    # note is closed instead.
    "voice": dict(name="Voice", label="Vocals", abbreviation="Voc.", instrument="Vocalist", program=53,
                  clef="auto", written_shift=0, low=36, high=96, stem_checks=False, check_octave=False,
                  beat_rests=True, end_grid=2, tab=False),
}
# A lower candidate c for a reported pitch counts as the real note when its
# octave partial (c + 12) is there and one of these holds, both relative to
# that partial:
#   - its own fundamental is within FUND_DB — F1 and A-flat1 on the test song
#     sat 5 to 18 dB under their octave;
#   - its third harmonic (c + 19) is within THIRD_DB. A stem with the lowest
#     octave filtered away still shows it: D-flat1's fundamental was down at
#     -20 to -35 dB, its third harmonic A-flat2 at -1 to -3 dB, E-flat1's at -6
#     to -8 dB. Where the reported note is right (C2), the would-be third
#     harmonic of C1 sat at -24 to -29 dB. The third harmonic is also near
#     100 Hz, where a 0.3 s window resolves a semitone; 35 Hz is not.
FUND_DB = 20.0
THIRD_DB = 15.0
# The octave partial itself must stand within this of the loudest probe.
PARTIAL_DB = 20.0
# Notes on a stem this far under the typical note are not the instrument:
# where the test song's bass rests, basic-pitch still read 56 "notes" off a
# stem 79 dB down, while every real note sat within 12 dB of the median.
GATE_DB = 30.0



class Stem:
    """Spectral level of candidate pitches near a note's start."""

    def __init__(self, path):
        import soundfile as sf

        y, self.sr = sf.read(path, always_2d=True, dtype="float32")
        self.y = y.mean(axis=1)

    def levels(self, start, end, pitches):
        sr = self.sr
        # Skip the pick attack, read up to 0.3 s of the body, at least 80 ms.
        a = int((start + 0.02) * sr)
        b = max(int(min(end, start + 0.30) * sr), a + int(0.08 * sr))
        seg = self.y[a:b]
        if len(seg) < 16:
            return np.full(len(pitches), -240.0)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 16))
        freqs = np.fft.rfftfreq(1 << 16, 1 / sr)
        out = []
        for m in pitches:
            lo, hi = 440 * 2 ** ((m - 69.4) / 12), 440 * 2 ** ((m - 68.6) / 12)
            band = spec[(freqs > lo) & (freqs < hi)]
            out.append(20 * np.log10(band.max() + 1e-12) if len(band) else -240.0)
        return np.array(out)


    def rms_db(self, start, seconds=0.25):
        a = int(start * self.sr)
        seg = self.y[a:a + int(seconds * self.sr)]
        return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12) if len(seg) else -240.0


def fundamental(stem, start, end, pitch, floor):
    """The note this pitch is an overtone of, judged by the harmonic series.

    The reported pitch may be the second or third harmonic of the real note
    (an octave, or an octave and a fifth, above it). Lowest plausible
    candidate wins; see FUND_DB / THIRD_DB.
    """
    cands = [c for c in (pitch - 19, pitch - 12) if c >= floor]
    if not cands:
        return pitch
    probe = sorted({pitch} | {m for c in cands for m in (c, c + 12, c + 19)})
    lv = dict(zip(probe, stem.levels(start, end, probe)))
    loudest = max(lv.values())
    for c in cands:
        octave = lv[c + 12]
        if octave < loudest - PARTIAL_DB:
            continue
        if lv[c] >= octave - FUND_DB or lv[c + 19] >= octave - THIRD_DB:
            return c
    return pitch


def gate_quiet(line, stem):
    """Drop notes read off a stem that is all but silent there (GATE_DB)."""
    if not line:
        return line, 0
    levels = [stem.rms_db(e["start_time"]) for _, _, e in line]
    floor = float(np.median(levels)) - GATE_DB
    kept = [n for n, lv in zip(line, levels) if lv >= floor]
    return kept, len(line) - len(kept)


def even_octaves(line, max_gap=Fraction(1)):
    """One octave per run of the same note name — for the notes the octave
    check moved.

    Consecutive notes of one pitch class, each within a beat of the last, are
    one figure; the octave check decides note by note and can flip within it.
    A bassist reading the part called that flip-flopping the real problem, the
    wrong octave itself a lesser one. The run's most common octave (the lower
    on a tie) is given to every note in it that the check moved; a note the
    transcriber reported and the check left alone keeps its octave. Evening
    every note flattened real octave figures: on a song whose bass leaps
    C2-C3 as its hook, agreement with the part fell from 84% to 82%, while
    the overdriven song gained nothing more from it than from this.
    """
    out, changed, i = list(line), 0, 0
    while i < len(out):
        j = i + 1
        while (j < len(out) and out[j][1] % 12 == out[i][1] % 12
               and out[j][0] - out[j - 1][0] <= max_gap):
            j += 1
        if j - i > 1:
            counts = defaultdict(int)
            for _, p, _ in out[i:j]:
                counts[p] += 1
            best = min(counts, key=lambda p: (-counts[p], p))
            for k in range(i, j):
                if out[k][1] != best and out[k][2].get("octave_moved"):
                    out[k] = (out[k][0], best, out[k][2])
                    changed += 1
        i = j
    return out, changed


def onset_latency(events, clock, grid):
    """How far the transcriber's onsets sit behind the beat grid, in seconds.

    The circular mean of every onset's phase within a grid cell: a cell is a
    cycle, so an onset just past one line and one just before the next average
    to the line, not to the middle of the cell. Returns (seconds, R), R being
    how concentrated the phases are (0 = spread evenly, 1 = all at one phase);
    below 0.2 there is no consistent lag to remove and none is.
    """
    import math

    if not events:
        return 0.0, 0.0
    phase = float(clock.bar_phase)
    angles = np.array([2 * math.pi * ((clock.quarters(e["start_time"]) - phase) * grid % 1)
                       for e in events])
    s, c = np.sin(angles).mean(), np.cos(angles).mean()
    r = math.hypot(s, c)
    if r < 0.2:
        return 0.0, r
    quarters = math.atan2(s, c) / (2 * math.pi) / grid
    return quarters * 60.0 / clock.bpm, r


def pick_line(events, clock, grid, profile, stem, latency=0.0):
    """One note per onset: (onset in quarters, sounding pitch, event).

    `latency` comes off the times before they are snapped to the grid; the
    event keeps its real times, which the octave check and the MIDI use.
    """
    groups = defaultdict(list)
    for e in events:
        if profile["low"] <= e["pitch"] <= profile["high"]:
            groups[snap(clock.quarters(e["start_time"] - latency), grid)].append(e)

    in_range = [e for groups_at in groups.values() for e in groups_at]
    line, moved = [], 0
    for onset in sorted(groups):
        pitch = min(e["pitch"] for e in groups[onset])
        # The kept pitch's own event — not the onset's earliest, which may be a
        # quiet overtone ending a 16th later and cutting the note short.
        first = min((e for e in groups[onset] if e["pitch"] == pitch), key=lambda e: e["start_time"])
        if stem is not None and profile["check_octave"]:
            fixed = fundamental(stem, first["start_time"], first["end_time"], pitch, profile["octave_floor"])
            moved += fixed != pitch
            # Remember which octaves were a judgement call: even_octaves may
            # only touch those. Copied, so the caller's events stay as given.
            first = {**first, "octave_moved": fixed != pitch}
            pitch = fixed
        # The note rings while an event of its pitch that began by now goes on:
        # an overtone heard alone over a long note is that note still sounding.
        t0 = first["start_time"]
        end = max([first["end_time"]] + [e["end_time"] for e in in_range
                                         if e["pitch"] == pitch and e["start_time"] <= t0 + 0.03 < e["end_time"]])
        line.append((onset, pitch, {**first, "end_time": end}))
    return line, moved


class Attacks:
    """How much the stem's level steps up at a moment: a new note's attack.

    The rise in dB of the RMS level, 35 ms after against 35 ms before. On the
    bars of one song a bassist heard as straight repeated eighths, it was 4.3
    dB (median) at the eighths and at most 1.4 dB on the 16ths between them —
    0.95 of them told apart, where librosa's onset strength managed 0.85.
    """

    def __init__(self, stem, clock):
        import librosa

        hop = 256
        rms = librosa.feature.rms(y=stem.y, frame_length=1024, hop_length=hop)[0]
        level = 20 * np.log10(rms + 1e-9)
        k = 3                                            # 3 frames of 5.8 ms
        self.rise = np.r_[np.zeros(k), level[2 * k:] - level[:-2 * k], np.zeros(k)]
        self.frames = np.arange(len(level)) * hop / stem.sr
        # The clock goes seconds -> quarters; the grid needs the way back.
        self.times = np.arange(0.0, len(stem.y) / stem.sr, 0.002)
        self.quarters = np.array([clock.quarters(t) for t in self.times])

    def seconds(self, q):
        return float(np.interp(float(q), self.quarters, self.times))

    def at(self, q, window=0.035):
        t = self.seconds(q)
        a, b = np.searchsorted(self.frames, [t - window, t + window])
        return float(self.rise[a:b + 1].max()) if b > a else 0.0


OVERTONES = {12, 19, 24, 28, 31}   # semitones over a note: its 2nd to 6th harmonics
ATTACK_SHARE = 0.5   # an attack: a rise of this share of the part's median at its onsets
OFFBEAT_X = 2.0      # and on a 16th between eighths, this many times that
LATE = 0.2           # quarters: an unattacked 16th that began this soon after an eighth is that eighth, heard late


def check_attacks(line, stem, clock, grid, latency=0.0):
    """Notes the stem's attacks deny, and attacks the notes missed.

    A run of repeated eighths often comes back from basic-pitch as one long
    note — the pitch never changes, so nothing tells it a new note began — and,
    the other way, with notes that are not there: the octave overtone heard on
    its own a 16th into a note, or one eighth reported as two 16ths. All of it
    lands on the 16ths between eighths, where a bass line rarely plays a note
    without an attack. So, by the stem's level (Attacks):

    - a note on a 16th between eighths with no attack is dropped when it is an
      overtone — one the octave check moved, or one over the note still
      sounding — or a short restart a 16th before the same note; a slurred
      real 16th is none of these, and stays
    - a note is split at each eighth inside it where the stem attacks
    - an unattacked note on a 16th between eighths that began within LATE of
      the eighth before is that eighth, reported late (soft attacks come
      back 100-125 ms behind): it goes to the eighth, or goes if the eighth
      has the same note already

    Against the purchased bass parts of 25 songs: precision 0.67 -> 0.76,
    recall 0.76 -> 0.74, F better on 17 songs and worse on 2 (the late-eighth
    pass alone: 8 better, none worse). The overdriven song: notes found 85% ->
    92%, on the right pitch 81% -> 88%; the bars a bassist heard as straight
    eighths 73 -> 79 of 80. The share: at 0.3 the
    overdriven song found 94%, but on the clean song, where the stem steps up
    on the snare's backbeat too (drums bleed into it), 35 of 56 splits had no
    note in the part; at 0.5, 6 more stray notes than before instead of 32.
    Dropping every unattacked 16th instead cost the clean song's slurred 16ths
    (99% -> 88%); splitting on 16ths too doubled the overdriven song's notes.
    """
    if not line:
        return line, 0, 0, 0, 0.0
    attacks = Attacks(stem, clock)
    rises = [attacks.at(onset) for onset, _, _ in line]
    threshold = ATTACK_SHARE * float(np.median(rises))
    step = Fraction(1, grid)
    on_eighth = lambda q: q * 2 == int(q * 2)
    needed = lambda q: threshold if on_eighth(q) else threshold * OFFBEAT_X

    kept, dropped = [], 0
    for i, ((onset, pitch, e), rise) in enumerate(zip(line, rises)):
        nxt = line[i + 1] if i + 1 < len(line) else None
        # (not one that began just after the eighth before: that is the eighth
        # itself, late, and the last pass below puts it back)
        restart = (nxt is not None and nxt[1] == pitch and nxt[0] - onset == step
                   and clock.quarters(e["end_time"]) - clock.quarters(e["start_time"]) < 0.35
                   and clock.quarters(e["start_time"] - latency) - float(onset - step) >= LATE)
        # an overtone of the note still sounding, heard on its own (two octaves
        # up escapes the octave check, which looks one octave and a 12th down)
        overtone = bool(kept) and pitch - kept[-1][1] in OVERTONES and kept[-1][2]["end_time"] > e["start_time"]
        if kept and not on_eighth(onset) and rise < needed(onset) and (e.get("octave_moved") or restart or overtone):
            before = kept[-1]
            if before[1] == pitch:
                kept[-1] = (before[0], pitch, {**before[2], "end_time": max(before[2]["end_time"], e["end_time"])})
            dropped += 1
            continue
        kept.append((onset, pitch, e))

    out, split = [], 0
    for i, (onset, pitch, e) in enumerate(kept):
        out.append((onset, pitch, e))
        stop = min(clock.quarters(e["end_time"]) - float(step) / 2,
                   float(kept[i + 1][0]) if i + 1 < len(kept) else float("inf"))
        q = onset + step
        while q < stop:
            if on_eighth(q) and attacks.at(q) >= needed(q):
                out.append((q, pitch, {**e, "start_time": attacks.seconds(q)}))
                split += 1
            q += step
    out.sort(key=lambda n: n[0])

    final, late = [], 0
    for onset, pitch, e in out:
        eighth = onset - step
        began = clock.quarters(e["start_time"] - latency) - float(eighth)
        if not on_eighth(onset) and began < LATE and attacks.at(onset) < needed(onset):
            if final and final[-1][0] == eighth and final[-1][1] == pitch:
                before = final[-1]           # the eighth has it already: one note, reported twice
                final[-1] = (eighth, pitch, {**before[2], "end_time": max(before[2]["end_time"], e["end_time"])})
                late += 1
                continue
            if not final or final[-1][0] < eighth:
                final.append((eighth, pitch, e))   # the eighth is free: the note belongs there
                late += 1
                continue
        final.append((onset, pitch, e))
    return final, dropped, split, late, threshold


class Contour:
    """The stem's pitch over time, by pYIN (MIDI, NaN where unvoiced)."""

    def __init__(self, stem):
        import librosa

        y = librosa.resample(stem.y, orig_sr=stem.sr, target_sr=22050) if stem.sr != 22050 else stem.y
        f0, _, _ = librosa.pyin(y, fmin=30, fmax=400, sr=22050, frame_length=2048, hop_length=256)
        self.midi = librosa.hz_to_midi(f0)
        self.times = np.arange(len(f0)) * 256 / 22050

    def values(self, t0, t1):
        a, b = np.searchsorted(self.times, [t0, t1])
        v = self.midi[a:b]
        return v[~np.isnan(v)]

    def at(self, t0, t1):
        v = self.values(t0, t1)
        return float(np.median(v)) if len(v) else float("nan")


SLIDE_MIN = 3.0      # semitones a glide must cover to be a slide
# Which slides are written: only up an octave, on a path that keeps going up.
# A bassist judged 14 bars of one song with every slide the path shows
# drawn (41 in the song): the three heard as written were all an octave up,
# with 0.88 or more of the path's steps upward; the six heard as separate notes
# were fourths, fifths, and octaves down. Up an octave at 0.85 keeps 10 of the
# 41, none of them where no slide was heard (two sit a note early or late).
# Slides down or by smaller steps exist too (four of the bars), but nothing in
# the path tells them from separate notes yet.
SLIDE_UP = 12
SLIDE_ALONG = 0.85


def glides(values, start, end, along_min=0.7):
    """A pitch path that moves steadily from `start` to `end` (not a jump): it
    covers SLIDE_MIN or more, passes through two or more frames clear of both
    ends, and keeps one direction for along_min of its steps."""
    if len(values) < 4 or abs(end - start) < SLIDE_MIN:
        return False
    lo, hi = sorted((start, end))
    between = np.sum((values > lo + 0.7) & (values < hi - 0.7))
    steps = np.diff(values)
    steps = steps[np.abs(steps) > 0.05]
    along = np.mean(np.sign(steps) == np.sign(end - start)) if len(steps) else 0.0
    return between >= 2 and along >= along_min


def octave_up(pitch, land, target):
    """The slide goes up an octave: where the path lands (a semitone either way —
    it may still be bending) or the next note as transcribed."""
    return (not np.isnan(land) and abs(round(land) - pitch - SLIDE_UP) <= 1) or target - pitch == SLIDE_UP


def find_slides(line, stem, clock, grid):
    """Slides, from the stem's pitch path (Contour).

    A slide is lost on the way through the transcriber: the pitch never stops
    long enough to be a note. A bassist heard one bar — two low As, then a
    slide up an octave; the last A sliding down — come back as one low A, a
    rest, and an A-sharp cut short. So, between a note and the next:

    - an attack in the gap after a note, where the path sits on that note's
      pitch and then glides to the next note: that note struck again, with a
      slide to the next (which takes the pitch the path lands on — the octave
      check had moved one such landing a twelfth down)
    - a note whose path glides into the next note: a slide between them
    Only slides up an octave are written (SLIDE_UP, SLIDE_ALONG). A note that
    slides is held to where it goes, and gets the pitch its path held before
    moving (a semitone off, otherwise, from the bend). Falls — a path dropping
    off a note into nothing — are not written: verovio does not draw them, and
    none could be judged.
    """
    if len(line) < 2:
        return line, 0, 0
    contour = Contour(stem)
    attacks = Attacks(stem, clock)
    threshold = ATTACK_SHARE * float(np.median([attacks.at(onset) for onset, _, _ in line]))
    step = Fraction(1, grid)
    out, added, slid = [], 0, 0
    for i, (onset, pitch, e) in enumerate(line):
        if i + 1 >= len(line):
            out.append((onset, pitch, e))
            break
        nxt_onset, nxt_pitch, nxt_e = line[i + 1]
        t_i, t_j = attacks.seconds(onset), attacks.seconds(nxt_onset)
        land = contour.at(t_j + 0.02, t_j + 0.08)
        held = contour.at(t_i + 0.02, t_i + 0.10)
        end = clock.quarters(e["end_time"])
        start = None
        q = onset + 2 * step
        while q < nxt_onset:                    # an attack in the gap, on the note's pitch, then a glide
            t = attacks.seconds(q)
            if (q * 2 == int(q * 2) and q >= end - float(step) / 2 and attacks.at(q) >= threshold
                    and abs(contour.at(t, t + 0.04) - pitch) <= 1.0 and octave_up(pitch, land, nxt_pitch)
                    and glides(contour.values(t + 0.06, t_j + 0.02), pitch, land, SLIDE_ALONG)):
                start = q
                break
            q += step
        moved_land = lambda: (round(land) if not np.isnan(land) and round(land) != nxt_pitch
                              and (round(land) == nxt_e["pitch"] or round(land) % 12 == nxt_pitch % 12) else None)
        if start is not None:
            out.append((onset, pitch, e))
            out.append((start, pitch, {**e, "start_time": attacks.seconds(start), "end_time": t_j, "slide_next": True,
                                       "octave_moved": False}))
            added += 1
            slid += 1
        elif (not np.isnan(held) and octave_up(round(held) if abs(round(held) - pitch) == 1 else pitch, land, nxt_pitch)
              and glides(contour.values(max(t_i + 0.08, t_j - 0.5), t_j + 0.02), held, land, SLIDE_ALONG)):
            fixed = round(held) if abs(round(held) - pitch) == 1 else pitch
            # The path has the slide starting at this pitch: settled, not for even_octaves to move
            # (it raised a G1 sliding to G2 into a G2 "sliding" to itself).
            out.append((onset, fixed, {**e, "end_time": max(e["end_time"], t_j), "slide_next": True, "octave_moved": False}))
            slid += 1
        else:
            out.append((onset, pitch, e))
            continue
        fix = moved_land()
        if fix is not None:
            line[i + 1] = (nxt_onset, fix, {**nxt_e, "octave_moved": False})
        if (fix if fix is not None else nxt_pitch) == out[-1][1]:     # no slide to the same note
            last = out[-1]
            out[-1] = (last[0], last[1], {k: v for k, v in last[2].items() if k != "slide_next"})
            slid -= 1
    return sorted(out, key=lambda n: n[0]), added, slid


def estimate_key(line, clock):
    from music21 import note, stream

    s = stream.Stream()
    for onset, pitch, e in line:
        n = note.Note(pitch)
        n.quarterLength = max(0.25, clock.quarters(e["end_time"]) - float(onset))
        s.append(n)
    return s.analyze("key")


def pick_clef(profile, line):
    from music21 import clef

    if profile["clef"] == "bass":
        return clef.BassClef()
    if profile["clef"] == "auto" and line:
        # The staff that leaves less of the line on ledger lines: treble, E4 to
        # F5, or the same an octave down. A median against middle C picked plain
        # treble for a man singing mostly F3 to F4 (median C4 exactly).
        pitches = np.array([p for _, p, _ in line])
        def ledger(low, high):
            return int(np.maximum(0, np.maximum(low - pitches, pitches - high)).sum())
        if ledger(52, 65) < ledger(64, 77):
            return clef.Treble8vbClef()
    return clef.TrebleClef()


def build(line, clock, grid, profile, k, latency=0.0, song_chords=(), with_tab=False):
    from music21 import instrument, interval, key, meter, note, spanner, stream

    part = stream.Part()
    inst = getattr(instrument, profile["instrument"])()
    inst.partName = profile["name"]
    inst.partAbbreviation = profile["abbreviation"]   # music21's own is "Elec b"
    # music21's ElectricBass carries no transposition; without one the page
    # pitch would also be what the MIDI plays, an octave too high.
    if profile["written_shift"]:
        inst.transposition = interval.Interval(-profile["written_shift"])
    part.insert(0, inst)
    part.insert(0, pick_clef(profile, line))
    part.insert(0, key.KeySignature(k.sharps))
    part.insert(0, meter.TimeSignature(clock.time_signature))
    part.insert(0, metronome(clock))
    part.atSoundingPitch = False

    shift, pickup_bars = bar_origin(clock, line[0][0] if line else None, grid)
    free = Fraction(0)
    written = []
    for i, (onset, pitch, e) in enumerate(line):
        # Snapped where the bar puts it: bar 1 may start a 16th off the clock's 8ths.
        end = clock.quarters(e["end_time"] - latency) + float(shift)
        own = snap(end, profile["end_grid"] or grid) - shift - onset
        if profile["end_grid"]:
            # Ending on 8ths, none is shorter than one unless the next note cuts
            # it: a note let go before the 8th's middle snapped to nothing.
            own = max(own, Fraction(1, profile["end_grid"]))
        nxt = line[i + 1][0] - onset if i + 1 < len(line) else own
        if profile["end_grid"] and 0 < nxt - own < Fraction(1, profile["end_grid"]):
            own = nxt     # a gap under the end grid, before a note off it, is no rest
        n = note.Note(spelled(pitch + profile["written_shift"], k))
        n.quarterLength = max(Fraction(1, grid), min(own, nxt))
        if e.get("lyric"):
            text, syllabic = e["lyric"]
            n.lyrics.append(note.Lyric(text=text, syllabic=syllabic))
        at = onset + shift
        if profile["beat_rests"]:
            for r_at, r_len in rests(free, at, clock.bar_length, clock.beat_length, clock.pulse):
                part.insert(float(r_at), note.Rest(quarterLength=r_len))
            free = at + n.quarterLength
        part.insert(float(at), n)
        written.append((n, e))
    for (n, e), (n2, _) in zip(written, written[1:]):
        if e.get("slide_next") and n.pitch.midi != n2.pitch.midi:
            g = spanner.Glissando(n, n2)
            g.lineType = "solid"
            part.insert(0, g)

    notated = notate(part, pickup_bars, chords_end(song_chords, clock, grid, shift))
    place_symbols(notated, chord_symbols(song_chords, clock, grid, shift), grid)
    if with_tab:
        return tab.with_tab_staff(notated, profile["name"], profile["abbreviation"]), pickup_bars
    score = stream.Score()
    score.insert(0, notated)
    return score, pickup_bars


def set_lyrics(line, path):
    """Put each note's syllable(s) on its event as "lyric"; returns counts."""
    sylls = lyrics.syllables(json.loads(Path(path).read_text()))
    pairs = lyrics.attach([e["start_time"] for _, _, e in line], sylls)
    for i, parts in pairs.items():
        line[i][2]["lyric"] = lyrics.lyric_text(parts)
    return {"syllables": len(sylls), "notes_with_lyric": len(pairs), "notes": len(line),
            "syllables_placed": sum(len(v) for v in pairs.values())}


def write_midi(line, path, name, program):
    try:
        import pretty_midi
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=program, name=name)   # GM 34 electric bass (finger), 54 voice oohs
    for i, (_, pitch, e) in enumerate(line):
        end = e["end_time"]
        if i + 1 < len(line):
            end = min(end, line[i + 1][2]["start_time"])
        track.notes.append(pretty_midi.Note(velocity=int(np.clip(e.get("velocity", 0.7) * 127, 30, 127)),
                                            pitch=int(pitch), start=e["start_time"], end=max(end, e["start_time"] + 0.05)))
    pm.instruments.append(track)
    pm.write(str(path))
    return str(path)


def pop_lyrics(argv):
    """Take `--lyrics=PATH` out of argv; empty, absent or missing -> None."""
    for i, arg in enumerate(argv):
        if arg.startswith("--lyrics="):
            del argv[i]
            path = arg[len("--lyrics="):].strip()
            return path if path and Path(path).is_file() else None
    return None


def main():
    title = pop_title(sys.argv)
    tab_wanted = pop_tab(sys.argv)
    chords_path = pop_chords(sys.argv)
    lyrics_path = pop_lyrics(sys.argv)
    if not 5 <= len(sys.argv) <= 8:
        raise SystemExit(__doc__)
    src, dst, tempo_path, part_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    grid = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 4
    audio = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
    song_key = load_key(sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None)
    song_chords = json.loads(Path(chords_path).read_text())["chords"] if chords_path else []
    if part_name not in PROFILES:
        raise SystemExit(f"unknown part {part_name!r}; one of {', '.join(PROFILES)}")
    profile = PROFILES[part_name]

    clock, mode = make_clock("auto", tempo_path)
    events = json.loads(src.read_text())
    stem = Stem(audio) if audio and profile["stem_checks"] else None
    in_range = [e for e in events if profile["low"] <= e["pitch"] <= profile["high"]]
    latency, latency_r = onset_latency(in_range, clock, grid)
    line, moved = pick_line(events, clock, grid, profile, stem, latency)
    gated = dropped = split = late = evened = slide_starts = slides = 0
    threshold = None
    if stem is not None:
        line, gated = gate_quiet(line, stem)
        line, dropped, split, late, threshold = check_attacks(line, stem, clock, grid, latency)
        line, slide_starts, slides = find_slides(line, stem, clock, grid)
        line, evened = even_octaves(line)
    if not line:
        part = profile["label"].lower()
        print(json.dumps({**skip_part(dst, f"No {part} part: the {part} stem gave no notes "
                                           f"(most likely the song has none).", [dst.with_suffix(".mid")]),
                          "part": part_name, "events_in": len(events), "gated_quiet": gated}))
        return
    k = song_key or estimate_key(line, clock)
    sung = set_lyrics(line, lyrics_path) if lyrics_path else None
    with_tab = tab_wanted and profile["tab"]
    score, pickup_bars = build(line, clock, grid, profile, k, latency, song_chords, with_tab)
    set_title(score, profile["label"], title)
    write_part(score, dst)
    tabbed = None
    if with_tab:
        # Four strings, unless the line goes below a 4-string's low E: then five, with the low B.
        tuning = tab.BASS5 if min(p for _, p, _ in line) < tab.BASS4[-1] else tab.BASS4
        tabbed = {**tab.add_tab(dst, tuning=tuning), "strings": len(tuning)}
    midi = write_midi(line, dst.with_suffix(".mid"), profile["name"], profile["program"])

    kept = sum(1 for e in events if profile["low"] <= e["pitch"] <= profile["high"])
    staff = pick_clef(profile, line)
    print(json.dumps({
        "path": str(dst),
        "midi": midi,
        "part": part_name,
        "tempo_source": mode,
        "bpm": round(clock.bpm, 3),
        "time_signature": clock.time_signature,
        "key": f"{k.tonic.name.replace('-', 'b')} {k.mode}",
        "key_source": "key.json" if song_key is not None else "this part's notes",
        "pickup_bars": pickup_bars,
        "grid": f"1/{grid} quarter",
        "onset_latency_ms": round(latency * 1000, 1),
        "onset_phase_r": round(latency_r, 3),
        "events_in": len(events),
        "out_of_range": len(events) - kept,
        "notes": len(line),
        "octave_checked": stem is not None and profile["check_octave"],
        "octave_moved": moved,
        "gated_quiet": gated,
        "attack_threshold_db": round(threshold, 2) if threshold is not None else None,
        "unattacked_dropped": dropped,
        "split_at_attacks": split,
        "late_to_eighth": late,
        "slides": slides,
        "slide_starts_added": slide_starts,
        "octave_evened": evened,
        "clef": staff.sign + ("8vb" if staff.octaveChange else ""),
        "chord_symbols": len(chord_symbols(song_chords, clock, grid, 0)) if song_chords else 0,
        "lyrics": sung,
        "tab": tabbed,
        "measures": len(score.parts[0].getElementsByClass("Measure")),
    }))


if __name__ == "__main__":
    main()
