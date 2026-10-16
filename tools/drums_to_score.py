#!/usr/bin/env python3
"""Write a drum part from DrumSep's per-piece stems.

Usage: drums_to_score.py <pieces_dir> <out.musicxml> <tempo.json> [grid] [presence_db] [hit_db]
                         [--title=SONG] [--chords=chords.json]

  pieces_dir   kick / snare / toms / hh / ride / crash .wav, as the
               split-drums stage writes them (or .flac)
  tempo.json   the beat tracker's output; bar lines come from its downbeats,
               through the same rule the pitched parts use, so every part
               numbers its bars alike
  grid         subdivisions per quarter note; 4 = 16ths (default)
  presence_db  a piece whose peak sits this far below the loudest piece is
               treated as absent (default -30)
  hit_db       within a present piece, a hit quieter than this relative to
               the piece's own peak is dropped (default -24)
  chords.json  the song's chords (estimate_chords.py): sections that play the
               same chords get the same rehearsal letter, so the marks read as
               the song's form. Without it the letters just run A, B, C

Also writes <out>.mid — General MIDI drums on channel 10, for listening.

Why two gates
-------------
Onset detection normalises each signal, so a stem that is almost silent still
yields a full set of "hits" from its noise floor: on one test song DrumSep's
ride carried 0.0% of the kit's energy and still produced 473 onsets. Against
AlbumDB's separate kick and snare tracks the real hits sit within 3 dB of their
piece's peak and the false ones 40 dB below it, while pieces that are not in
the song peak 35-39 dB under the loudest one and present ones stay within 24.
So a whole-piece presence gate, then a per-hit gate. On that test: kick 100%
precision / 87% recall, snare 100% / 100%, and no toms in a song without toms.

Layout
------
Hands (snare, toms, cymbals) stems up, feet (kick) stems down, as drum charts
are written. A rehearsal mark where a crash follows a fill: against the
double barlines of purchased parts (29 songs) that finds section starts with
F 0.41, where model-compose's music-segment-detector on the mix scores 0.31
(its default) to 0.33, and a mark every 8 bars 0.24. Sections whose chords
agree beat by beat on 60% or more share a letter; against the same
sections labelled by the purchased part's own chords (and its repeats) that
gives pairwise F 0.70, precision 1.00 (median of 24 songs).
"""
import json
import math
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notes_to_score import (bar_origin, make_clock, metronome, notate, pop_chords, pop_title, rests, set_title,  # noqa: E402
                            skip_part, snap, write_part)

SR = 44100
PIECES = ["kick", "snare", "toms", "hh", "ride", "crash"]
# The kit as written: DrumSep's pieces, with the toms split into the floor
# tom and the rest (split_toms).
KIT = ["kick", "snare", "toms", "floor", "hh", "ride", "crash"]
HANDS, FEET = KIT[1:], KIT[:1]
# Lengths one note can show, in quarters: 16th to dotted half.
PLAIN = {Fraction(n, 4) for n in (1, 2, 3, 4, 6, 8, 12)}

# Where each piece sits on a five-line percussion staff, in treble-clef terms,
# and its notehead — the common drum-set key: kick in the bottom space, snare
# in the third, cymbals as crosses above. DrumSep gives every tom in one
# stem; split_toms puts the floor tom in the second space, the others on the
# mid-tom line.
STAFF = {
    "kick":  ("F4", "normal"),
    "snare": ("C5", "normal"),
    "toms":  ("D5", "normal"),
    "floor": ("A4", "normal"),
    "hh":    ("G5", "x"),
    "ride":  ("F5", "x"),
    "crash": ("A5", "x"),
}
# General MIDI percussion keys, for the listening file and the MusicXML.
GM = {"kick": 36, "snare": 38, "toms": 47, "floor": 43, "hh": 42, "ride": 51, "crash": 49}
GM_NAME = {"kick": "Bass Drum 1", "snare": "Acoustic Snare", "toms": "Low-Mid Tom", "floor": "High Floor Tom",
           "hh": "Closed Hi-Hat", "ride": "Ride Cymbal 1", "crash": "Crash Cymbal 1"}
# Hi-hat strokes the onset detector misses: an eighth where the hi-hat piece's
# level steps up this much (dB, 35 ms after against before) while the hi-hat
# sounds (within HAT_SOUNDING_DB of its typical stroke). An open hi-hat rings
# on, so the next stroke changes the spectrum too little to stand out.
HAT_RISE_DB = 2.0
HAT_SOUNDING_DB = 12.0
# A tom hit ringing below this is the floor tom. Against the tom positions the
# drum parts write (12 songs, 414 hits): floor-or-not agreed on 0.80 of them,
# where writing every tom as the floor tom agrees on 0.63; 110 and 140 Hz gave
# 0.77 and 0.69, so the line is not sharp. A candidate for drummers' ears.
FLOOR_HZ = 125.0
# A bar keeps the beat on the hi-hat when the hi-hat piece sounds louder than
# the ride piece by more than this at the cymbal strokes of the RIDE_SPAN bars
# around it (median), else on the ride.
RIDE_DB = 0.0
RIDE_SPAN = 4
# A crash or tom quieter than these, against the song's typical snare stroke
# (dB, the louder 100 ms after each), is another piece leaking into its own.
CRASH_DB = -31.0
TOM_DB = -12.0


def load(path):
    path = Path(path)
    if not path.exists() and path.suffix == ".wav":     # a run's pieces kept compressed
        path = path.with_suffix(".flac")
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def detect(pieces_dir, presence_db, hit_db):
    """Hit times per piece, after the presence and per-hit gates."""
    import librosa

    wave = {p: load(Path(pieces_dir) / f"{p}.wav") for p in PIECES}
    peak = {p: float(np.percentile(np.abs(y), 99.9)) for p, y in wave.items()}
    loudest = max(peak.values()) or 1.0
    hits, report = {}, {}
    for p, y in wave.items():
        level = 20 * np.log10(peak[p] / loudest + 1e-12)
        if level < presence_db:
            hits[p] = []
            report[p] = {"present": False, "level_db": round(level, 1), "hits": 0}
            continue
        times = librosa.onset.onset_detect(y=y, sr=SR, units="time", delta=0.1)
        kept = []
        for t in times:
            a = int(t * SR)
            seg = np.abs(y[max(0, a - int(0.01 * SR)): a + int(0.05 * SR)])
            if len(seg) and 20 * np.log10(seg.max() / (peak[p] or 1.0) + 1e-12) >= hit_db:
                kept.append(float(t))
        hits[p] = kept
        report[p] = {"present": True, "level_db": round(level, 1),
                     "candidates": int(len(times)), "hits": len(kept)}
    return hits, report


def voice(groups, pieces, shift, clock, grid, stem, end):
    """One voice as a flat part: the hits of these pieces, stems one way, and
    the rests between them up to `end`."""
    from music21 import meter, note, percussion, stream

    bar_length, beat, pulse = clock.bar_length, clock.beat_length, clock.pulse
    part = stream.Part()
    part.insert(0, meter.TimeSignature(clock.time_signature))
    onsets = sorted(o for o in groups if groups[o] & pieces)
    free = Fraction(0)
    for i, onset in enumerate(onsets):
        # A hit rings until the next one in its voice, but never past the next
        # beat. Drum parts mark when a piece is struck, not how long it sounds,
        # so the length only keeps the rhythm readable beat by beat — a note
        # across a beat comes out dotted or tied, and a tie reads as a second
        # hit. Bars are whole beats from offset 0, so this stops at bar lines too.
        # Off the eighths of a 6/8 beat it stops at the next eighth, as a
        # 1¼-quarter note can only be written tied.
        at = onset + shift
        nxt = onsets[i + 1] if i + 1 < len(onsets) else onset + 1
        length = max(Fraction(1, grid), min(nxt - onset, beat - at % beat, pulse - at % pulse if at % pulse else beat))
        while length > Fraction(1, grid) and length not in PLAIN:
            length -= Fraction(1, grid)                   # 1¼ quarters (6/8) only exists tied
        heads = []
        for p in sorted(groups[onset] & pieces, key=KIT.index):
            display, head = STAFF[p]
            u = note.Unpitched(displayName=display)
            u.notehead = head
            heads.append(u)
        el = heads[0] if len(heads) == 1 else percussion.PercussionChord(heads)
        el.quarterLength = length
        el.stemDirection = stem
        for r_at, r_len in rests(free, at, bar_length, beat, pulse):
            part.insert(float(r_at), note.Rest(quarterLength=r_len))
        part.insert(float(at), el)
        free = at + length
    for r_at, r_len in rests(free, end, bar_length, beat, pulse):
        part.insert(float(r_at), note.Rest(quarterLength=r_len))
    return part


def two_voices(hands, feet):
    """Hands' bars, each split into voice 1 (hands, stems up) and voice 2 (feet,
    stems down) — the layout drum charts use, so a bar reads as two rhythms."""
    from music21 import stream

    for mh, mf in zip(hands.getElementsByClass("Measure"), feet.getElementsByClass("Measure")):
        for m, vid in ((mh, "1"), (mf, "2")):
            v = stream.Voice(id=vid)
            for el in list(m.notesAndRests):
                at = el.getOffsetBySite(m)
                m.remove(el)
                v.insert(at, el)
            mh.insert(0, v)
    return hands


LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"                      # rehearsal letters skip I and O
PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Two sections are the same when their chord roots agree on this share of
# beats. Roots only: the chord recognizer (BTC) moves between C, Cm and C7 on
# one root from beat to beat, and comparing major/minor too took the form
# letters (pairwise F against the parts' sections, 25 songs) from 0.70 to
# 0.67; roots at half the beats gave 0.71.
SAME_SECTION = 0.5


def drum_sections(groups, shift, bar_length, gap=4):
    """[bar index from offset 0] where sections start, read from the drums
    alone: a crash on the first beat right after a fill bar (toms, or four or
    more snare hits in its second half), at least `gap` bars after the last
    start. Bar 0 always starts one."""
    toms, snares, crashes = set(), defaultdict(int), set()
    for onset, pieces in groups.items():
        k, pos = divmod(onset + shift, bar_length)
        if "toms" in pieces or "floor" in pieces:
            toms.add(k)
        if "snare" in pieces and pos >= Fraction(bar_length, 2):
            snares[k] += 1
        if "crash" in pieces and pos <= Fraction(1, 4):
            crashes.add(k)
        elif "crash" in pieces and pos >= bar_length - Fraction(1, 4):
            crashes.add(k + 1)                            # a crash a 16th early still opens the bar
    fills = toms | {k for k, n in snares.items() if n >= 4}
    starts = [0]
    for k in sorted(crashes):
        if k - 1 in fills and k - starts[-1] >= gap:
            starts.append(k)
    return starts


def harmony(label):
    """"F#:maj" -> "6", "Gb:min7" -> "6", "N" -> "N": the root's pitch class,
    so the same chord spelled two ways still matches (see SAME_SECTION)."""
    if label in ("N", "X"):
        return "N"
    root = label.partition(":")[0]
    return str((PITCH[root[0]] + root[1:].count("#") - root[1:].count("b")) % 12)


def form_letters(starts, end_bar, chords_path, clock, shift, bar_length):
    """A letter per section start, by the song's form: the chord root on each beat
    of a section, from its first bar, against every earlier section's; the one
    agreeing on the most beats (over the shorter of the two) gives its letter
    if that is SAME_SECTION or more, otherwise the section takes the next new
    letter. Without chords the letters just run A, B, C."""
    if not chords_path:
        return [LETTERS[i % 24] for i in range(len(starts))]
    spans = [(clock.quarters(c["start"]) + shift, clock.quarters(c["end"]) + shift, harmony(c["label"]))
             for c in json.loads(Path(chords_path).read_text())["chords"]]

    def beats(a, b):
        return [next((h for s, e, h in spans if s <= q < e), "N") for q in range(a * bar_length, b * bar_length)]

    def agree(x, y):
        n = min(len(x), len(y))
        return sum(p == q for p, q in zip(x[:n], y[:n])) / n if n else 0.0

    letters, seen = [], []
    for k, nxt in zip(starts, starts[1:] + [end_bar]):
        seq = beats(k, nxt)
        best, who = max(((agree(seq, s), i) for i, s in enumerate(seen)), default=(0.0, None))
        letters.append(letters[who] if best >= SAME_SECTION else LETTERS[len(set(letters)) % 24])
        seen.append(seq)
    return letters


def drop_hat_bleed(groups, shift, bar_length, steady=6):
    """Remove hi-hat and ride hits on the 16th offbeats of a bar whose hats play
    the 8ths — `steady` of every 8 eighth positions — while its 16th offbeats
    are only scattered (fewer than `steady` of 8). Those are other sounds
    leaking into the hi-hat piece: a drummer, asked about two such songs, heard
    plain 8ths. A bar with the offbeats filled too is a real 16th groove and
    keeps them. Returns how many hits it removed.

    Against 20 purchased drum parts, hi-hat F went from 0.797 to 0.819 (median)
    and 0.787 to 0.809 (all hits pooled): 14 songs better, 2 by -0.01. Without
    the 16th-groove exception one song with real 16th hats fell 0.93 to 0.73.
    The same kind of rules for kicks — weaker than their neighbours, under a
    bass note or a tom — gained nothing.
    """
    hats = {"hh", "ride"}
    bars = defaultdict(list)
    for onset, pieces in groups.items():
        if pieces & hats:
            k, pos = divmod(onset + shift, bar_length)
            bars[k].append((onset, pos))
    removed = 0
    per_bar = Fraction(bar_length, 4)                 # 8 eighths in 4/4, 6 in 6/8
    for k, hs in bars.items():
        eighths = {pos for onset, pos in hs if pos % Fraction(1, 2) == 0}
        offbeats = [onset for onset, pos in hs if pos % Fraction(1, 2) != 0]
        if len(eighths) >= steady * per_bar and len({o for o in offbeats}) < steady * per_bar:
            for onset in offbeats:
                removed += len(groups[onset] & hats)
                groups[onset] -= hats
                if not groups[onset]:
                    del groups[onset]
    return removed


def build(hits, clock, grid, chords=None, cues=None):
    """The score from the hits; section marks from `cues` (the hits before
    quiet_toms and quiet_crashes, when given). A crash or tom too quiet to
    write still tells where a section starts: read from the filtered hits,
    the marks fell from 10 to 7 a song and their F from 0.41 to 0.36 against
    29 purchased parts' section lines."""
    from music21 import clef, expressions, instrument, stream, tempo

    def grouped(hs):
        out = defaultdict(set)
        for p, times in hs.items():
            for t in times:
                out[snap(clock.quarters(t), grid)].add(p)
        return out

    groups = grouped(hits)

    shift, pickup_bars = bar_origin(clock, min(groups) if groups else None, grid)
    bar = clock.bar_length
    dropped = drop_hat_bleed(groups, shift, bar)
    end = (math.floor((max(groups) + shift) / bar) + 1) * bar if groups else 0   # the last hit's bar, whole
    hands = voice(groups, set(HANDS), shift, clock, grid, "up", end)
    feet = voice(groups, set(FEET), shift, clock, grid, "down", end)
    kit = instrument.Percussion()
    # Not "Percussion"; and "Drumset", with with_kit()'s instruments, is what
    # MuseScore imports as its drum set.
    kit.partName, kit.partAbbreviation = "Drumset", "Dr."
    hands.insert(0, kit)
    hands.insert(0, clef.PercussionClef())
    hands.insert(0, metronome(clock))
    starts = drum_sections(grouped(cues) if cues is not None else groups, shift, bar)
    marks = list(zip(starts, form_letters(starts, int(end / bar), chords, clock, shift, bar)))
    for k, letter in marks:
        hands.insert(k * clock.bar_length, expressions.RehearsalMark(letter))

    score = stream.Score()
    score.insert(0, two_voices(notate(hands, pickup_bars, end), notate(feet, pickup_bars, end)))
    return score, groups, pickup_bars, [[k + 1 - pickup_bars, letter] for k, letter in marks], dropped


def ring_hz(y, t):
    """The strongest frequency, 50-400 Hz, 30-180 ms after a hit: the tom's
    own ring, past the stick's attack."""
    seg = y[int((t + 0.03) * SR):int((t + 0.18) * SR)]
    if len(seg) < 64:
        return None
    n = 1 << 14
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    band = (freqs >= 50) & (freqs <= 400)
    return float(freqs[band][int(np.argmax(spec[band]))])


class Level:
    """A piece's RMS level (dB), and how much it steps up at a moment."""

    def __init__(self, y, hop=256, k=3):
        import librosa

        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
        self.db = 20 * np.log10(rms + 1e-9)
        self.rise = np.r_[np.zeros(k), self.db[2 * k:] - self.db[:-2 * k], np.zeros(k)]
        self.frames = np.arange(len(self.db)) * hop / SR

    def _span(self, t0, t1):
        a, b = np.searchsorted(self.frames, [t0, t1])
        return a, max(b, a + 1)

    def step(self, t, window=0.035):
        a, b = self._span(t - window, t + window)
        return float(self.rise[a:b].max())

    def peak(self, t0, t1):
        a, b = self._span(t0, t1)
        return float(self.db[a:b].max())


def recover_hats(hits, pieces_dir, clock):
    """Add the hi-hat strokes the onset detector missed (HAT_RISE_DB).

    Mostly an open hi-hat's: in the two songs with hand-read parts, the
    sections the drummer plays open came back with 12-40% of their strokes,
    closed ones with 85-89%. At the strokes the parts write there the hi-hat
    piece steps up 5-7 dB (median), between them 1 dB. Only eighths are tried,
    and none within a 16th of a stroke already found. Against 20 purchased
    drum parts: hi-hat recall 0.84 -> 0.88 (all strokes pooled), precision
    0.80 -> 0.80, F better on 11 songs and worse on 1 (by 0.01); 2 dB edged
    out 3. The two hand-read songs' hi-hat F 0.62 -> 0.88 and 0.63 -> 0.84.
    """
    found = sorted(hits.get("hh", []))
    if len(found) < 8:
        return hits, 0
    level = Level(load(Path(pieces_dir) / "hh.wav"))
    typical = float(np.median([level.peak(t, t + 0.1) for t in found]))
    times = np.arange(0.0, level.frames[-1], 0.002)
    quarters = np.array([clock.quarters(t) for t in times])
    taken = {round(clock.quarters(t) * 4) for t in found}
    added = []
    q = math.ceil(quarters[0] * 2) / 2
    while q < quarters[-1]:
        if not {round(q * 4) + d for d in (-1, 0, 1)} & taken:
            t = float(np.interp(q, quarters, times))
            if level.step(t) >= HAT_RISE_DB and level.peak(t, t + 0.1) >= typical - HAT_SOUNDING_DB:
                added.append(t)
                taken.add(round(q * 4))
        q += 0.5
    return {**hits, "hh": sorted(found + added)}, len(added)


def snare_level(hits, pieces_dir):
    """The song's typical snare stroke, dB — what quiet_toms and quiet_crashes
    measure against, so a quieter recording keeps its toms and crashes. None
    with fewer than 8 snare strokes."""
    snares = hits.get("snare", [])
    if len(snares) < 8:
        return None
    snare = Level(load(Path(pieces_dir) / "snare.wav"))
    return float(np.median([snare.peak(t, t + 0.1) for t in snares]))


def quiet_toms(hits, pieces_dir, snare):
    """Drop the tom hits too quiet to be one (TOM_DB): the kick and snare
    leaking into the tom piece.

    Against 20 purchased drum parts, two thirds of the toms we wrote were not
    in the part — most of them where the part has a kick (and we wrote one
    too) or a snare. The ones it has sit at a median 2 dB under the song's
    snare, the others at 23 dB under (AUC 0.89). Dropping those below -12
    dB: tom precision 0.34 -> 0.64, recall 0.61 -> 0.58 (all hits pooled), F
    median 0.50 -> 0.56; 14 songs better, 2 worse by 0.02. Of the real toms
    lost, 15 of 36 are one song whose toms sit low in the mix.
    -20 and -16 dB kept more (recall 0.63, 0.62) but left most of the
    leaking: in a song read by hand, whose chorus a drummer heard without
    toms, 57 and 32 of our 75 tom hits stayed where the part has 8; at -12,
    11. The 25 toms a drummer confirmed tom by tom all stay, and in a
    second hand-read song 41 of 45 with 39 of its 40.
    Returns the hits and how many it dropped.
    """
    toms = hits.get("toms", [])
    if not toms or snare is None:
        return hits, 0
    level = Level(load(Path(pieces_dir) / "toms.wav"))
    kept = [t for t in toms if level.peak(t, t + 0.1) >= snare + TOM_DB]
    return {**hits, "toms": kept}, len(toms) - len(kept)


def quiet_crashes(hits, pieces_dir, snare):
    """Drop the crashes too quiet to be one (CRASH_DB).

    Against 20 purchased drum parts the crashes we wrote that the part does
    not have sit at a median 34 dB under the song's snare, the ones it has at
    16 dB under (AUC 0.85). Dropping those below -31 dB: crash precision 0.55
    -> 0.73, recall 0.65 -> 0.63 (all hits pooled), F median 0.62 -> 0.70; 15
    songs better, 3 worse by up to 0.04. -34 to -27 dB all came within 0.01.
    Against the snare rather than in absolute dB (as good: -45 dB), so a
    quieter recording keeps its crashes. Returns the hits and how many it
    dropped.
    """
    crashes = hits.get("crash", [])
    if not crashes or snare is None:
        return hits, 0
    crash = Level(load(Path(pieces_dir) / "crash.wav"))
    kept = [t for t in crashes if crash.peak(t, t + 0.1) >= snare + CRASH_DB]
    return {**hits, "crash": kept}, len(crashes) - len(kept)


def hat_or_ride(hits, pieces_dir, clock, grid):
    """Write each bar's beat-keeping cymbal as the hi-hat or the ride, not both.

    DrumSep's hi-hat and ride pieces each catch some of the other, and both
    pass the gates: against 20 purchased drum parts, 30% of the strokes we
    wrote on a hi-hat or ride carried both heads at once, and 15% only the
    ride where the part writes the hi-hat. Per stroke the louder piece is the
    one played (AUC 0.86); the decision takes a few bars' median, as a
    drummer keeps one cymbal for a stretch. Of the strokes the parts write on
    one of the two (the ride as diamond heads or under "Ride →") we now put
    0.87 on the same one (hi-hat 0.94, ride 0.67), against 0.51 before (both
    heads counted wrong; hi-hat 0.57, ride 0.36); 18 songs better, 2 the
    same. Leaning to the ride by 6 dB per bar scored a little higher but
    turned an open hi-hat, which leaks into the ride piece, into ride: 34 of
    77 bars of one song with a hand-read part and no ride. At 0 dB over 4
    bars, 4.
    Returns the hits and how many bars went to the ride.
    """
    times = sorted(hits.get("hh", []) + hits.get("ride", []))
    if not times:
        return hits, 0
    level = {p: Level(load(Path(pieces_dir) / f"{p}.wav")) for p in ("hh", "ride")}
    phase, bar = snap(clock.bar_phase, grid), clock.bar_length
    at = {}                                       # grid position -> its earliest stroke
    for t in times:
        at.setdefault(snap(clock.quarters(t), grid), t)
    bars, diffs = defaultdict(list), defaultdict(list)
    for q, t in at.items():
        k = math.floor((q - phase) / bar)
        bars[k].append(q)
        diffs[k].append(level["hh"].peak(t, t + 0.1) - level["ride"].peak(t, t + 0.1))
    out, ride_bars = {"hh": [], "ride": []}, 0
    for k, qs in bars.items():
        around = [d for j in range(k - RIDE_SPAN // 2, k + RIDE_SPAN - RIDE_SPAN // 2) for d in diffs.get(j, [])]
        piece = "hh" if np.median(around) > RIDE_DB else "ride"
        ride_bars += piece == "ride"
        out[piece] += [at[q] for q in qs]
    return {**hits, "hh": sorted(out["hh"]), "ride": sorted(out["ride"])}, ride_bars


def split_toms(hits, pieces_dir):
    """The floor tom's hits apart from the other toms', by the pitch each rings at (FLOOR_HZ)."""
    times = hits.get("toms", [])
    if not times:
        return hits
    y = load(Path(pieces_dir) / "toms.wav")
    floor = [t for t in times if (f := ring_hz(y, t)) is not None and f < FLOOR_HZ]
    rest = [t for t in times if t not in set(floor)]
    return {**hits, "toms": rest, "floor": floor}


def with_kit(path):
    """Name each drum piece in the MusicXML and point every note at its own.

    music21 writes one "Percussion" instrument and bare <unpitched> staff
    positions. Notation programs then guess the sounds — MuseScore imports a
    generic percussion part, or the first piece alone. A <score-instrument>
    and General MIDI key per piece, and an <instrument> on each note, are
    MusicXML's own way to say which drum a note is; with the part named
    "Drumset", MuseScore 4 opens it as its drum set, every hit on its sound.
    """
    xml = Path(path).read_text()
    part = re.search(r'<score-part id="([^"]+)">.*?</score-part>', xml, re.S)
    ids = {k: f"{part.group(1)}-I{GM[k] + 1}" for k in STAFF}
    pieces = "".join(f"""
      <score-instrument id="{ids[k]}">
        <instrument-name>{GM_NAME[k]}</instrument-name>
      </score-instrument>""" for k in STAFF)
    midi = "".join(f"""
      <midi-instrument id="{ids[k]}">
        <midi-channel>10</midi-channel>
        <midi-unpitched>{GM[k] + 1}</midi-unpitched>
      </midi-instrument>""" for k in STAFF)
    head = re.sub(r"\s*<score-instrument .*?</score-instrument>", "", part.group(0), flags=re.S)
    head = re.sub(r"\s*<midi-instrument .*?</midi-instrument>", "", head, flags=re.S)
    head = head.replace("</score-part>", pieces + midi + "\n    </score-part>")
    xml = xml[:part.start()] + head + xml[part.end():]
    at = {STAFF[k][0]: ids[k] for k in STAFF}           # every piece has its own staff position

    def note(m):
        s = m.group(0)
        u = re.search(r"<display-step>(\w)</display-step>\s*<display-octave>(\d)</display-octave>", s)
        if not u or u.group(1) + u.group(2) not in at:
            return s
        # After <duration> and any <tie>, as the schema orders a note's children.
        return re.sub(r"(</duration>(?:\s*<tie [^>]*/>)*)", rf'\1\n        <instrument id="{at[u.group(1) + u.group(2)]}" />',
                      s, count=1)

    Path(path).write_text(re.sub(r"<note\b[^>]*>.*?</note>", note, xml, flags=re.S))


def write_midi(hits, path):
    try:
        import pretty_midi
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI()
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for p, times in hits.items():
        for t in times:
            kit.notes.append(pretty_midi.Note(velocity=100, pitch=GM[p], start=t, end=t + 0.1))
    pm.instruments.append(kit)
    pm.write(str(path))
    return str(path)


def main():
    title = pop_title(sys.argv)
    chords = pop_chords(sys.argv)
    if not 4 <= len(sys.argv) <= 7:
        raise SystemExit(__doc__)
    pieces_dir, dst, tempo_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    grid = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    presence_db = float(sys.argv[5]) if len(sys.argv) > 5 else -30.0
    hit_db = float(sys.argv[6]) if len(sys.argv) > 6 else -24.0

    clock, mode = make_clock("auto", tempo_path)
    hits, report = detect(pieces_dir, presence_db, hit_db)
    cues = hits
    snare = snare_level(hits, pieces_dir)
    hits, quiet_tom_hits = quiet_toms(hits, pieces_dir, snare)
    hits = split_toms(hits, pieces_dir)
    hits, recovered = recover_hats(hits, pieces_dir, clock)
    hits, quiet = quiet_crashes(hits, pieces_dir, snare)
    hits, ride_bars = hat_or_ride(hits, pieces_dir, clock, grid)
    if not any(hits.values()):
        print(json.dumps({**skip_part(dst, "No drum part: no hits in the drum stem "
                                           "(most likely the song has no drums).", [dst.with_suffix(".mid")]),
                          "gates_db": {"presence": presence_db, "hit": hit_db}, "pieces": report}))
        return
    score, groups, pickup_bars, marks, dropped = build(hits, clock, grid, chords, cues)
    set_title(score, "Drums", title)
    write_part(score, dst)
    with_kit(dst)
    midi = write_midi(hits, dst.with_suffix(".mid"))

    print(json.dumps({
        "path": str(dst),
        "midi": midi,
        "tempo_source": mode,
        "bpm": round(clock.bpm, 3),
        "time_signature": clock.time_signature,
        "bar_agreement": (round(clock.bar_agreement, 3)
                          if clock.bar_agreement is not None else None),
        "pickup_bars": pickup_bars,
        "grid": f"1/{grid} quarter",
        "gates_db": {"presence": presence_db, "hit": hit_db},
        "pieces": report,
        "onsets": len(groups),
        "sections": marks,
        "hat_bleed_dropped": dropped,
        "floor_tom_hits": len(hits.get("floor", [])),
        "hat_strokes_recovered": recovered,
        "quiet_toms_dropped": quiet_tom_hits,
        "quiet_crashes_dropped": quiet,
        "ride_bars": ride_bars,
        "measures": len(score.parts[0].getElementsByClass("Measure")),
    }))


if __name__ == "__main__":
    main()
