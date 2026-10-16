#!/usr/bin/env python3
"""Build a readable score from raw transcription events.

Usage: notes_to_score.py <events> <out.musicxml> <bpm|auto> [min_pitch] [grid] [tempo.json] [key.json] [--title=SONG]

  events     the transcriber's notes as JSON, or the MIDI file it writes
             alongside them. Both carry the same events; MIDI is the
             practical choice inside a workflow because file-store can
             persist it while the notes object is not serializable.

The transcriber emits note events in seconds. Converting the MIDI it also
writes is a dead end: music21 reads that MIDI at its 120 BPM default, and
rescaling afterwards produces durations no notation program can render
(4096th notes). Going straight from the events lets the seconds-to-beats
conversion use the real tempo once, with nothing to undo later.

  bpm        `auto` to follow the beat grid in tempo.json (preferred), or a
             number to force a fixed tempo
  min_pitch  drop notes below this MIDI pitch (bass bleed)
  grid       subdivisions per quarter note; 4 = 16ths (default)
  key.json   the song's key (tools/estimate_key.py): key signature and
             note spelling, shared with every other part. Without it the
             score has no key signature and music21 spells with sharps
  tempo.json the beat tracker's output; required when bpm is `auto`. Beat
             This! (`music-beat-tracking`) marks each bar's first beat with
             `is_downbeat`, which places the bar lines. A plain beat list (the
             older `music-analyzer` output) still works; bar 1 then starts at
             the first detected beat and the meter defaults to 4/4.

Why a beat grid rather than one BPM number
------------------------------------------
A single tempo converts seconds to beats as `t * bpm / 60`, so any error in
bpm accumulates over the track. Holding a 16th-note grid to within half a cell
over four minutes needs the tempo right to about 0.03 BPM, and a human
performance of this track drifts 0.45 BPM. No fixed number can hold alignment
to the end, and chasing decimal places does not help.

The beat times say where each beat actually landed, so interpolating between
them absorbs the drift. A fixed bpm stays available for when detection picks
the wrong metrical level and there is nothing better to hand.
"""
import bisect
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

SPLIT_PITCH = 60  # C4: at or above goes to the treble staff

FLAT = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def snap(value: float, grid: int) -> Fraction:
    """Snap to the nearest 1/grid of a quarter note, as an exact fraction."""
    return Fraction(round(value * grid), grid)


def pop_tab(argv):
    """Take `--tab=...` out of argv: false/no/off/0 -> False; empty or absent -> True.
    The guitar and bass parts carry a TAB staff unless told not to — some
    players practise without one."""
    for i, arg in enumerate(argv):
        if arg.startswith("--tab="):
            del argv[i]
            return arg[len("--tab="):].strip().lower() not in ("false", "no", "off", "0")
    return True


def pop_title(argv):
    """Take `--title=...` out of argv, wherever it is; empty or absent -> None.

    A flag rather than one more positional: the engravers' positional
    arguments are already long, and a title is optional for all of them.
    """
    for i, arg in enumerate(argv):
        if arg.startswith("--title="):
            del argv[i]
            return arg[len("--title="):].strip() or None
    return None


def set_title(score, part, song=None):
    """Head the page "song — part", or the part's name alone.

    Without this music21 prints "Music21 Fragment" and names "Music21" as the
    composer. verovio draws the movement title and not the work title, so
    the whole heading goes in the movement title; the work title carries the
    song for other notation programs.
    """
    from music21 import metadata

    md = metadata.Metadata()
    md.title = song or part
    md.movementName = f"{song} — {part}" if song else part
    md.composer = ""
    score.metadata = md


def write_part(score, dst):
    """Write a finished part as MusicXML, in place of any skip note.

    At export music21 makes the accidentals again, on its own copy, and
    treats a chord symbol's notes like any other: a G chord in A major gets a
    natural (<root-alter>0) wherever no G-sharp comes before it in the bar, and
    prints as "G♮". An alter of 0 says nothing a chord symbol needs, so it
    goes. (makeNotation=False would skip that pass, but also the splitting of
    durations MusicXML cannot write, which some parts need.)
    """
    import re

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(dst))
    dst.write_text(re.sub(r"\s*<(root|bass)-alter>0</(root|bass)-alter>", "", dst.read_text()))
    tidy_beams_and_accidentals(dst)
    anchor_chord_symbols(dst)
    skipped_note(dst).unlink(missing_ok=True)


def anchor_chord_symbols(path):
    """Put a chord that changes under a rest before the rest, at a positive offset.

    music21 writes it after the rest, its offset counted back from the rest's
    end. That is valid MusicXML, but Verovio draws such a chord in a part with
    a second staff (the TAB) between the staves and large — 9 bars of one
    song's bass part. Counted forward from the rest's start, it sits above
    the staff where it belongs.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    moved = 0
    for measure in tree.getroot().iter("measure"):
        kids = list(measure)
        for i, el in enumerate(kids):
            offset = el.find("offset") if el.tag == "harmony" else None
            if offset is None or int(offset.text) >= 0:
                continue
            j = i - 1
            while j >= 0 and kids[j].tag == "harmony":
                j -= 1
            note = kids[j] if j >= 0 else None
            if note is None or note.tag != "note" or note.find("chord") is not None:
                continue
            ahead = int(note.findtext("duration", "0")) + int(offset.text)
            if ahead < 0:
                continue
            offset.text = str(ahead)
            measure.remove(el)
            measure.insert(list(measure).index(note), el)
            moved += 1
    if moved:
        tree.write(path, encoding="UTF-8", xml_declaration=True)


SHARP_ORDER = "FCGDAEB"


def tidy_beams_and_accidentals(path):
    """Two things music21's export writes that a player reads wrong.

    - Four 16ths in a beat of 4/4 or 3/4 come out with the second beam broken
      in the middle (2+2), as if two eighths; music21 does join them in 6/8.
      Within each beamed group, notes that carry a beam level next to each
      other are joined on it.
    - Accidentals the bar does not need read as if the key signature were
      missing. music21 repeats one on the same note later in the bar (C-natural,
      G, C-natural again), and reminds of the key after a bar that altered the
      note in any octave. A reminder is kept, in parentheses, where a player
      needs one: the same note altered otherwise in the bar before (a C-sharp
      in E major after a C-natural), or another octave of it altered otherwise
      earlier in this bar. The rest go.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    for part in tree.getroot().iter("part"):
        fifths, before = {}, {}                      # staff -> fifths in effect; staff -> the last bar's {(step, octave): alter}
        for measure in part.iter("measure"):
            groups, altered, last = {}, {}, {}       # (staff, voice) -> open beam group; staff -> {(step, octave): alter}, {step: alter}
            for el in measure:
                if el.tag == "attributes":
                    for key in el.iter("key"):
                        if key.findtext("fifths") is not None:
                            staves = [key.get("number")] if key.get("number") else list(fifths) or ["1"]
                            for s in staves:
                                fifths[s] = int(key.findtext("fifths"))
                    continue
                if el.tag != "note":
                    continue
                staff = el.findtext("staff", "1")
                pitch = el.find("pitch")
                if pitch is not None and el.find("grace") is None:
                    step, octave = pitch.findtext("step"), pitch.findtext("octave")
                    alter = int(float(pitch.findtext("alter", "0")))
                    k = fifths.get(staff, fifths.get("1", 0))
                    in_key = (1 if SHARP_ORDER.index(step) < k else 0) if k >= 0 else (-1 if SHARP_ORDER[::-1].index(step) < -k else 0)
                    seen, latest = altered.setdefault(staff, {}), last.setdefault(staff, {})
                    accidental = el.find("accidental")
                    if accidental is not None and accidental.get("parentheses") is None and alter == seen.get((step, octave), in_key):
                        if latest.get(step, alter) != alter or ((step, octave) not in seen
                                                                and before.get(staff, {}).get((step, octave), alter) != alter):
                            accidental.set("parentheses", "yes")
                        else:
                            el.remove(accidental)
                    if not any(t.get("type") == "stop" for t in el.findall("tie")):
                        seen[(step, octave)] = alter
                        latest[step] = alter
                if el.find("chord") is not None:
                    continue                         # beams sit on a chord's first note
                first = el.find("beam[@number='1']")
                who = (staff, el.findtext("voice", "1"))
                if first is None:
                    if who in groups:
                        groups[who].append(None)     # a rest inside the group: no beam joins across it
                    continue
                if first.text == "begin":
                    groups[who] = []
                groups.setdefault(who, []).append(el)
                if first.text == "end":
                    _join_beams(groups.pop(who))
            for staff_, seen_ in altered.items():
                before[staff_] = seen_
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def _join_beams(notes):
    for level in range(2, 7):
        beams = [n.find(f"beam[@number='{level}']") if n is not None else None for n in notes]
        i = 0
        while i < len(beams):
            j = i
            while beams[i] is not None and j + 1 < len(beams) and beams[j + 1] is not None:
                j += 1
            if j > i:
                for k in range(i, j + 1):
                    beams[k].text = "begin" if k == i else "end" if k == j else "continue"
            i = j + 1


def skipped_note(dst):
    """Where skip_part says why a part has no score: piano.skipped.txt."""
    return Path(dst).with_suffix(".skipped.txt")


def skip_part(dst, reason, also=()):
    """Write no score for a part with nothing in it, and a note saying why.

    An empty part — a bar of rests — reads as a failure, and in a song with no
    piano it is noise beside the parts that are there. render_score.py finds
    the note and draws nothing. An earlier run's files for the part (`dst`, and
    `also`) go, so they are not taken for this run's. Returns the fields for
    the engraver's summary.
    """
    for f in (dst, *also):
        Path(f).unlink(missing_ok=True)
    note = skipped_note(dst)
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(reason + "\n")
    return {"path": None, "skipped": reason, "note": str(note)}


def load_key(path):
    """The song's key from key.json (tools/estimate_key.py) as a music21 Key.

    One key per song, shared by every part so their signatures agree. None
    when there is no file; a part then keeps its own fallback.
    """
    if not path or not Path(path).exists():
        return None
    from music21 import key

    data = json.loads(Path(path).read_text())
    tonic = data["tonic"]
    if len(tonic) > 1 and tonic[1] == "b":      # "Bb" -> music21's "B-"
        tonic = tonic[0] + "-"
    return key.Key(tonic, data["mode"])


def spelled(midi, k):
    """Name a MIDI pitch the way the key would: its own scale degrees first,
    then flats in a flat key and sharps in a sharp one."""
    from music21 import pitch

    names = {p.pitchClass: p.name for p in k.getScale().getPitches()}
    name = names.get(midi % 12) or (FLAT if k.sharps < 0 else SHARP)[midi % 12]
    p = pitch.Pitch(name)
    p.octave = midi // 12 - 1
    while p.midi < midi:
        p.octave += 1
    while p.midi > midi:
        p.octave -= 1
    return p


HALF_GAP = 0.7      # a beat gap this short against the median is the tracker doubling: half a beat
DUP_GAP = 0.3       # and this short, the same beat detected twice: dropped


class BeatGrid:
    """Maps seconds to quarter-note positions along detected beat times.

    One detected beat is taken as one quarter note — or, in compound meter,
    one eighth (see _init_bars). Beat trackers drop beats in
    quiet passages, so a gap is measured against the median interval and
    credited the whole number of beats it spans — otherwise two beats either
    side of a rest would be treated as adjacent and everything after would
    shift.
    """

    def __init__(self, times, downbeats=()):
        if len(times) < 2:
            raise ValueError("a beat grid needs at least two beats")
        times = sorted(float(t) for t in times)
        gaps = [b - a for a, b in zip(times, times[1:])]
        median = sorted(gaps)[len(gaps) // 2]
        # A beat 80 ms after another is the same beat twice (two songs: 55 of
        # them, and 30): each was counted a beat, a bar gained every four.
        kept = times[:1]
        for t in times[1:]:
            if t - kept[-1] >= DUP_GAP * median:
                kept.append(t)
        times = kept
        gaps = [b - a for a, b in zip(times, times[1:])]

        # The tracker also runs at double speed for a stretch — the eighths of a
        # piano intro or a busy chorus (one song, 20-89 s: 70 gaps of
        # half the median). Counted as beats, each bar there was written as two
        # at twice the tempo, every bar number after it moved (65 for the
        # part's 55), and the song's tempo read 77 for 71. A gap under
        # HALF_GAP of the median is half a beat.
        positions = [0.0]
        for gap in gaps:
            positions.append(positions[-1] + (0.5 if gap < HALF_GAP * median else max(1, round(gap / median))))

        self.times = times
        self.positions = positions
        self.median = median
        # Tempo from the whole span, not the median gap. Beat This! snaps beats
        # to 50 fps frames, so every gap is a multiple of 20 ms and the median
        # lands on a frame boundary — 90.91 for a 92 BPM song. The span
        # averages that quantisation away.
        self.bpm = 60.0 * (positions[-1] - positions[0]) / (times[-1] - times[0])
        # Edge slopes, in quarters per second, for times outside the grid.
        self._head = (positions[1] - positions[0]) / gaps[0]
        self._tail = (positions[-1] - positions[-2]) / gaps[-1]

        self._init_bars(sorted(float(t) for t in downbeats))

    def _init_bars(self, downbeats):
        """Derive bar length and phase from the downbeats, if there are any.

        Bar lines are placed on a fixed grid — length from the most common
        bar, phase from the downbeats' most common position within it — rather
        than on every detected downbeat. Trackers emit the odd spurious
        downbeat: on one test song a single four-beat bar near the end came
        back as 3 + 1. Taken literally that engraves a 3/4 bar and a 1/4 bar
        that are not in the music; on the fixed grid the split sums back to 4
        and nothing after it moves.

        The phase is a vote, not the first downbeat: the first one is as
        likely to be spurious as any, and it set every bar line in the song.
        On 9 of 26 songs scored against purchased parts it sat one to three
        beats off the phase all the others agreed on (Muse's "Uprising": 1
        downbeat against 131) and every bar line was misplaced. Bar 1 is the
        earliest bar line on the winning phase from the first downbeat on,
        whether or not a downbeat was detected there.

        `bar_agreement` reports how many detected downbeats the grid honours,
        so a real change of meter shows up as a drop instead of being hidden.

        Compound meter: in 6/8 Beat This! follows the eighth notes (one song: 176
        a minute against the part's dotted quarter = 58) and puts a downbeat
        every six of them, or every three. Taken as quarters that engraves
        6/4 at quarter = 174, every note twice its length. A bar of six or
        twelve beats is therefore read as 6/8 or 12/8: each beat an eighth,
        positions halved, so `bar_length` stays in quarters (3 or 6) for
        everything downstream. `pulse` is the tracked beat in quarters,
        `beat_length` the counted beat (a dotted quarter in 6/8), which is
        what rests and note lengths should not cross.
        """
        self.bar_length = 4
        self.bar_phase = self.positions[0]
        self.bar_agreement = None
        self.downbeats = len(downbeats)
        self.pulse, self.beat_length, self.time_signature = 1, 1, "4/4"
        if not downbeats:
            return

        marks = [self.quarters(t) for t in downbeats]
        lengths = [round(b - a) for a, b in zip(marks, marks[1:]) if round(b - a) > 0]
        if lengths:
            self.bar_length = max(set(lengths), key=lengths.count)
        residues = [round(q) % self.bar_length for q in marks]
        winner = max(set(residues), key=residues.count)
        on_phase = next(q for q, r in zip(marks, residues) if r == winner)
        self.bar_phase = on_phase - self.bar_length * math.floor((on_phase - marks[0]) / self.bar_length)

        def on_grid(q):
            off = (q - self.bar_phase) % self.bar_length
            return min(off, self.bar_length - off) < 0.25

        self.bar_agreement = sum(on_grid(q) for q in marks) / len(marks)

        if self.bar_length in (6, 12):
            self.time_signature = f"{self.bar_length}/8"
            self.pulse, self.beat_length = Fraction(1, 2), Fraction(3, 2)
            self.positions = [p / 2 for p in self.positions]
            self.bar_phase /= 2
            self.bar_length //= 2
            self.bpm /= 2
            self._head /= 2
            self._tail /= 2
        else:
            self.time_signature = f"{self.bar_length}/4"

    def quarters(self, t: float) -> float:
        if t <= self.times[0]:
            return self.positions[0] + (t - self.times[0]) * self._head
        if t >= self.times[-1]:
            return self.positions[-1] + (t - self.times[-1]) * self._tail
        i = bisect.bisect_right(self.times, t) - 1
        span = self.times[i + 1] - self.times[i]
        frac = (t - self.times[i]) / span if span else 0.0
        return self.positions[i] + frac * (self.positions[i + 1] - self.positions[i])

    @property
    def coverage(self):
        return self.times[0], self.times[-1]


class FixedTempo:
    """Constant-tempo fallback: the old `t * bpm / 60` behaviour."""

    def __init__(self, bpm: float):
        self.bpm = float(bpm)
        self._per_second = self.bpm / 60.0
        # No downbeats to go on: bar 1 starts at t = 0, in 4/4.
        self.bar_length = 4
        self.pulse, self.beat_length, self.time_signature = 1, 1, "4/4"
        self.bar_phase = 0.0
        self.bar_agreement = None
        self.downbeats = 0

    def quarters(self, t: float) -> float:
        return t * self._per_second

    @property
    def coverage(self):
        return None, None


def load_events(src: Path):
    """Read transcription events from notes JSON or from the MIDI."""
    if src.suffix.lower() == ".json":
        return json.loads(src.read_text())

    from music21 import converter

    # The transcriber writes note times in seconds and declares no tempo, so
    # music21 reads the file at its 120 BPM default: one quarter = 0.5s.
    # Undo that here and everything downstream works in real seconds.
    score = converter.parse(str(src))
    per_quarter = 0.5
    events = []
    for el in score.flatten().notes:
        start = float(el.offset) * per_quarter
        dur = float(el.duration.quarterLength) * per_quarter
        pitches = el.pitches if el.isChord else (el.pitch,)
        for pit in pitches:
            events.append({
                "start_time": start,
                "end_time": start + dur,
                "pitch": pit.midi,
                "velocity": (el.volume.velocityScalar or 0.5),
            })
    return events


def build(events, clock, min_pitch: int, grid: int, k=None):
    from music21 import chord, clef, instrument, key, meter, note, stream, tempo

    kept = [e for e in events if e["pitch"] >= min_pitch]

    # Group by snapped onset so simultaneous pitches become one chord rather
    # than a pile of overlapping single notes. Both ends of a note go through
    # the clock, so a note spanning a tempo change gets the right length.
    groups = defaultdict(list)
    for e in kept:
        start = clock.quarters(e["start_time"])
        onset = snap(start, grid)
        dur = snap(clock.quarters(e["end_time"]) - start, grid)
        groups[onset].append((e["pitch"], max(dur, Fraction(1, grid))))

    score = stream.Score()
    treble, bass = stream.Part(), stream.Part()
    treble.insert(0, instrument.Piano())
    treble.insert(0, clef.TrebleClef())
    bass.insert(0, clef.BassClef())
    for part in (treble, bass):
        if k is not None:
            part.insert(0, key.KeySignature(k.sharps))
        part.insert(0, meter.TimeSignature(clock.time_signature))
    # Rounded for the printed marking only — nobody engraves a score as
    # "quarter = 122.283". The grid above keeps the full precision, so this
    # loses nothing but the ink. The exact value is in tempo.json.
    treble.insert(0, metronome(clock))

    shift, pickup_bars = bar_origin(clock, min(groups) if groups else None, grid)

    for onset in sorted(groups):
        for part, picked in (
            (treble, [g for g in groups[onset] if g[0] >= SPLIT_PITCH]),
            (bass, [g for g in groups[onset] if g[0] < SPLIT_PITCH]),
        ):
            if not picked:
                continue
            length = max(d for _, d in picked)
            pitches = sorted({p for p, _ in picked})
            if k is not None:
                pitches = [spelled(p, k) for p in pitches]
            el = (note.Note(pitches[0]) if len(pitches) == 1
                  else chord.Chord(pitches))
            el.quarterLength = length
            part.insert(float(onset + shift), el)

    for part in (treble, bass):
        score.insert(0, notate(part, pickup_bars))

    return score, kept, groups, pickup_bars


def bar_origin(clock, first_onset, grid: int):
    """Where bar 1 goes: returns (offset shift, number of pickup bars).

    Bar 1 sits on the first downbeat. The origin comes from the beat grid,
    never from where one part's notes happen to start: every part built from
    the same tempo.json must number its bars the same way, or the part scores
    stop lining up with each other. Shared by every part's engraver for that
    reason — two copies of this rule would drift apart. Notes ahead of bar 1
    get pickup bars numbered 0 and below.
    """
    phase = snap(clock.bar_phase, grid)
    bar = Fraction(clock.bar_length)
    first = phase if first_onset is None else first_onset
    pickup_bars = math.ceil((phase - first) / bar) if first < phase else 0
    return -(phase - pickup_bars * bar), pickup_bars


def notate(part, pickup_bars: int, end=None):
    """makeNotation, then renumber so bar 1 stays on the first downbeat.

    makeNotation numbers from 1 at offset 0; pickup bars come first.

    The gaps between notes are filled with rests first. makeNotation leaves
    them empty, and music21's MusicXML export then fills them with rests it
    marks print-object="no" — every part came out without a single printed
    rest (the bass part: 112 rests, all hidden).

    end pads with rests up to that offset, so parts notated separately (the
    drum part's two voices) come out with the same bars.
    """
    part.makeRests(refStreamOrTimeRange=[0.0, max(part.highestTime, float(end or 0))], fillGaps=True,
                   inPlace=True, hideRests=False)
    notated = part.makeNotation()
    if pickup_bars:
        for i, m in enumerate(notated.getElementsByClass("Measure")):
            m.number = i + 1 - pickup_bars
    return notated


def rests(start, stop, bar_length, beat=1, pulse=1):
    """[(offset, length)] of rests filling [start, stop), split the way drum
    parts write them: the rest of a beat on its own, then whole beats (a
    dotted quarter in 6/8) — a whole empty bar as one rest, two empty beats
    from beat 1 or 3 of a 4/4 bar as a half rest. music21 fills a gap with a
    single rest, which across beats comes out double-dotted and hides where
    the beat falls. In 6/8 a rest off the eighths first runs to the next
    eighth: a 16th and a quarter, not one 1¼-quarter rest."""
    out, t = [], start
    while t < stop:
        if t % pulse:
            nxt = min(stop, t - t % pulse + pulse)        # to the next tracked beat (eighth in 6/8)
        elif t % beat:
            nxt = min(stop, t - t % beat + beat)          # to the end of this beat
        elif t % bar_length == 0 and stop >= t + bar_length:
            nxt = t + bar_length
        elif bar_length == 4 and beat == 1 and t % 2 == 0 and stop >= t + 2:
            nxt = t + 2
        else:
            nxt = min(stop, t + beat)
        out.append((t, nxt - t))
        t = nxt
    return out


def chord_symbols(song_chords, clock, grid, shift):
    """(offset, element) for each change of chord symbol; N.C. as plain text.

    music21's NoChord writes a <harmony> that verovio prints as "C", so
    "no chord" goes in as a text direction instead.
    """
    from music21 import expressions, harmony

    out, prev = [], None
    for c in song_chords:
        if c["symbol"] == prev:
            continue
        prev = c["symbol"]
        at = snap(clock.quarters(c["start"]), grid) + shift
        if at < 0:
            continue
        if c["label"] == "N":
            el = expressions.TextExpression("N.C.")
            el.placement = "above"
        else:
            root, quality = c["label"].split(":")
            el = harmony.ChordSymbol(root=root.replace("b", "-"), kind=CHORD_KINDS[quality])
            # What follows the root on the page, as chords.json spells it ("m7",
            # "maj7", "m7b5"), not music21's or the renderer's own abbreviations.
            el.chordKindStr = c["symbol"][len(root):]
        out.append((at, el))
    return out


# Harte quality (chords.json) -> MusicXML <kind>.
CHORD_KINDS = {"maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented",
               "maj6": "major-sixth", "min6": "minor-sixth", "7": "dominant-seventh", "maj7": "major-seventh",
               "min7": "minor-seventh", "minmaj7": "major-minor", "dim7": "diminished-seventh",
               "hdim7": "half-diminished-seventh", "sus2": "suspended-second", "sus4": "suspended-fourth"}


def chords_end(song_chords, clock, grid, shift):
    """Where a part printing the chord symbols has to run to: the end of the bar
    the last chord starts in (as part offsets), or None without chords.

    A part ends with its last note; the chords go on — an outro the guitar or
    the voice sits out. Padded with rests to here, every chord gets its bar;
    otherwise they all piled into the last one (sixteen in one bar of a Queen
    song's rhythm chart).
    """
    starts = [snap(clock.quarters(c["start"]), grid) + shift for c in song_chords if c["label"] != "N"]
    if not starts:
        return None
    bar = Fraction(clock.bar_length)
    return (math.floor(max(starts) / bar) + 1) * bar


def place_symbols(notated, symbols, grid):
    """Put (offset, element) pairs — chord symbols — into the finished bars.

    Into the bars, not the part before makeNotation: a chord symbol is a
    zero-length chord to music21, and mixed in with the notes it made
    makeNotation split every bar into voices (numbered from 0, which verovio
    cannot attach anything to).
    """
    bars = list(notated.getElementsByClass("Measure"))
    for at, el in symbols:
        home = next((m for m in reversed(bars) if m.offset <= at), None)
        if home is not None:
            home.insert(float(at - Fraction(home.offset).limit_denominator(grid * 64)), el)


def pop_chords(argv):
    """Take `--chords=PATH` out of argv; empty, absent or missing -> None."""
    for i, arg in enumerate(argv):
        if arg.startswith("--chords="):
            del argv[i]
            path = arg[len("--chords="):].strip()
            return path if path and Path(path).is_file() else None
    return None


def metronome(clock):
    """The tempo mark: quarter = bpm, or dotted quarter = bpm / 1.5 in 6/8."""
    from music21 import duration, tempo

    if clock.beat_length == 1:
        return tempo.MetronomeMark(number=round(clock.bpm))
    return tempo.MetronomeMark(number=round(clock.bpm / clock.beat_length),
                               referent=duration.Duration(float(clock.beat_length)))


def make_clock(bpm_arg: str, tempo_path):
    """Pick the seconds-to-quarters mapping from the CLI arguments."""
    if bpm_arg.lower() != "auto":
        return FixedTempo(float(bpm_arg)), "fixed"

    if not tempo_path:
        raise SystemExit("bpm=auto needs a tempo.json argument")
    path = Path(tempo_path)
    if not path.exists():
        raise SystemExit(f"tempo file not found: {path}")

    data = json.loads(path.read_text())
    beats = data.get("beats", [])
    times = [b["time"] if isinstance(b, dict) else b for b in beats]
    downbeats = [b["time"] for b in beats if isinstance(b, dict) and b.get("is_downbeat")]
    if len(times) < 2:
        # Detection can come back with nothing useful; a declared bpm in the
        # same file is still better than failing the whole engrave.
        if data.get("bpm"):
            return FixedTempo(float(data["bpm"])), "fixed (no beats in tempo.json)"
        raise SystemExit(f"{path} carries neither beats nor a bpm")

    return BeatGrid(times, downbeats), ("beat grid + downbeats" if downbeats else "beat grid")


def main() -> None:
    title = pop_title(sys.argv)
    if not 4 <= len(sys.argv) <= 8:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    bpm_arg = sys.argv[3]
    min_pitch = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    grid = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    tempo_path = sys.argv[6] if len(sys.argv) > 6 else None
    k = load_key(sys.argv[7] if len(sys.argv) > 7 else None)

    clock, mode = make_clock(bpm_arg, tempo_path)
    events = load_events(src)
    if not any(e["pitch"] >= min_pitch for e in events):
        print(json.dumps({**skip_part(dst, "No piano part: the piano stem gave no notes "
                                           "(most likely the song has no piano)."),
                          "events_in": len(events)}))
        return
    score, kept, groups, pickup_bars = build(events, clock, min_pitch, grid, k)
    set_title(score, "Piano", title)
    write_part(score, dst)

    result = {
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "tempo_source": mode,
        "bpm": round(clock.bpm, 3),
        "key": f"{k.tonic.name.replace('-', 'b')} {k.mode}" if k is not None else None,
        "time_signature": clock.time_signature,
        "downbeats": clock.downbeats,
        # Share of detected downbeats that fall on the fixed bar grid. Well
        # below 1.0 means the meter really changes, and a single time
        # signature is the wrong model for this song.
        "bar_agreement": (round(clock.bar_agreement, 3)
                          if clock.bar_agreement is not None else None),
        "pickup_bars": pickup_bars,
        "grid": f"1/{grid} quarter",
        "events_in": len(events),
        "events_kept": len(kept),
        "dropped_low": len(events) - len(kept),
        "onsets": len(groups),
        "measures": len(score.parts[0].getElementsByClass("Measure")),
    }
    first, last = clock.coverage
    if first is not None:
        result["beats"] = len(clock.times)
        result["beat_span"] = [round(first, 3), round(last, 3)]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
