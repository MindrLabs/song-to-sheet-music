"""Tablature: a string and fret for every note of the guitar or bass part.

The engraver writes the part twice — standard notation and a TAB staff, one
part with two staves, as notation programs show a guitar with tab
(with_tab_staff()) — and add_tab() then fills in the TAB staff's MusicXML:
<string>/<fret> on each note, and the staff's lines with its tuning. (music21
writes a string and fret only on single notes, not on a chord's notes, hence
the XML.) A part written an octave up (the bass) is fingered at the pitch it
sounds, from the part's <transpose>.

Fingering: every onset's notes (a chord) get distinct strings in standard
tuning, frets 0-MAX_FRET. Among a chord's ways:
- a stretch past the hand's reach costs SPAN_COST a fret (fretted notes only;
  open strings are free). The reach depends on where the hand is — frets
  narrow up the neck: three frets in the first positions, four in the middle,
  five past the 12th (reach(); a guitarist who checked the TAB)
- a string skipped inside a chord of GAP_MIN notes or more costs GAP_COST — a
  pick strikes it, or the hand must mute it (the same guitarist, on an arpeggio
  played with a pick) — unless the chord lies in one of its standard shapes,
  where the skipped string is the shape's own
- with the song's chord symbol known there, a note of a SHAPE_MIN-note chord off
  that chord's standard shapes costs SHAPE_COST (shapes(): the E, A, D, C and G forms, open position
  included) — strummed parts are played in those shapes
- every fret a little (lower positions first)
Between onsets, moving the hand costs MOVE_COST a fret. The cheapest path over
the whole part (Viterbi over each onset's best few ways) is kept. A note no string can play — below the low E,
or past six notes at once — is left off the TAB staff (hidden where a whole
onset is unplayable).
"""
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

# Standard tunings, string 1 (highest) first, sounding MIDI.
GUITAR = [64, 59, 55, 50, 45, 40]
BASS4 = [43, 38, 33, 28]                # G D A E
BASS5 = [43, 38, 33, 28, 23]            # and a low B, for a line below the E
TUNING = GUITAR
MAX_FRET = 20
# Against the guitar parts of the 34-song batch, notes held as they are (medians
# of songs; before: a reach of four frets anywhere, no skipped strings or shapes):
# chords stretched past reach() 0.08 -> 0.04, chords of three or more notes in
# one standard shape of the song's chord 0.27 -> 0.48, notes past the 12th fret
# 0.032 -> 0.034; the same string as purchased TABs on the notes both have
# 0.410 -> 0.415 (2,242 notes).
SPAN_COST = 5.0
GAP_COST = 1.0
GAP_MIN = 3          # notes at once before a skipped string counts: a bass and a melody note pluck apart
SHAPE_COST = 2.0
SHAPE_MIN = 3        # notes at once before the chord's shapes count: a strum, not a bass under a melody
FRET_COST = 0.03
MOVE_COST = 0.3
SLIDE_COST = 1000.0  # a slide's or a tie's two notes on different strings: it cannot be played as one
WAYS = 12

# Standard chord shapes: (root string, root's fret over the shape's base,
# {string: fret over the base}). At base 0 they are the open chords (E, A, D,
# C, G and their minors and sevenths); higher up, the open strings become the
# barre.
FORMS = {
    "E": (6, 0, {"major": {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}, "minor": {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0},
                 "dom7": {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0}, "m7": {6: 0, 5: 2, 4: 0, 3: 0, 2: 0, 1: 0},
                 "maj7": {6: 0, 5: 2, 4: 1, 3: 1, 2: 0, 1: 0}, "sus4": {6: 0, 5: 2, 4: 2, 3: 2, 2: 0, 1: 0},
                 "power": {6: 0, 5: 2, 4: 2}}),
    "A": (5, 0, {"major": {5: 0, 4: 2, 3: 2, 2: 2, 1: 0}, "minor": {5: 0, 4: 2, 3: 2, 2: 1, 1: 0},
                 "dom7": {5: 0, 4: 2, 3: 0, 2: 2, 1: 0}, "m7": {5: 0, 4: 2, 3: 0, 2: 1, 1: 0},
                 "maj7": {5: 0, 4: 2, 3: 1, 2: 2, 1: 0}, "sus4": {5: 0, 4: 2, 3: 2, 2: 3, 1: 0},
                 "sus2": {5: 0, 4: 2, 3: 2, 2: 0, 1: 0}, "power": {5: 0, 4: 2, 3: 2}}),
    "D": (4, 0, {"major": {4: 0, 3: 2, 2: 3, 1: 2}, "minor": {4: 0, 3: 2, 2: 3, 1: 1},
                 "dom7": {4: 0, 3: 2, 2: 1, 1: 2}, "m7": {4: 0, 3: 2, 2: 1, 1: 1},
                 "maj7": {4: 0, 3: 2, 2: 2, 1: 2}, "sus4": {4: 0, 3: 2, 2: 3, 1: 3},
                 "sus2": {4: 0, 3: 2, 2: 3, 1: 0}, "power": {4: 0, 3: 2, 2: 3}}),
    "C": (5, 3, {"major": {5: 3, 4: 2, 3: 0, 2: 1, 1: 0}, "dom7": {5: 3, 4: 2, 3: 3, 2: 1, 1: 0},
                 "maj7": {5: 3, 4: 2, 3: 0, 2: 0, 1: 0}}),
    "G": (6, 3, {"major": {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3}, "dom7": {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 1}}),
}
# Chord qualities (Harte labels as in chords.json, MusicXML <kind>) onto the
# shapes above; the rest by their third.
HARTE = {"maj": "major", "maj6": "major", "aug": "major", "min": "minor", "min6": "minor", "dim": "minor",
         "dim7": "minor", "minmaj7": "minor", "7": "dom7", "min7": "m7", "hdim7": "m7", "maj7": "maj7",
         "sus4": "sus4", "sus2": "sus2", "5": "power"}
KINDS = {"major": "major", "major-sixth": "major", "augmented": "major", "minor": "minor", "minor-sixth": "minor",
         "diminished": "minor", "diminished-seventh": "minor", "major-minor": "minor", "dominant": "dom7",
         "dominant-ninth": "dom7", "minor-seventh": "m7", "half-diminished": "m7", "major-seventh": "maj7",
         "suspended-fourth": "sus4", "suspended-second": "sus2", "power": "power"}
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def places(pitch, tuning=None):
    return [(s + 1, pitch - o) for s, o in enumerate(tuning or TUNING) if 0 <= pitch - o <= MAX_FRET]


def reach(low):
    """Frets past the lowest fretted one the hand stretches to without strain there.
    The guitarist put it as three frets in the first positions and five past the
    12th; read as the stretch (1 to 4 low on the neck) rather than the frets covered
    (1 to 3) — the purchased TABs agree with the stretch (same string 0.417 against
    0.384) and the TAB keeps that agreement."""
    return 3 if low <= 5 else 4 if low <= 11 else 5


def chord_from_label(label):
    """(root pitch class, shape quality) from a Harte label ("C:min7"), None for no chord."""
    if not label or ":" not in label:
        return None
    root, quality = label.split(":", 1)
    pc = (STEPS[root[0]] + root[1:].count("#") - root[1:].count("b")) % 12
    quality = quality.split("/")[0].split("(")[0]
    return pc, HARTE.get(quality, "minor" if quality.startswith("min") else "major")


def chord_from_harmony(el):
    """The same, from a MusicXML <harmony>."""
    step = el.findtext("root/root-step")
    kind = el.findtext("kind") or ""
    if not step or kind == "none":
        return None
    pc = (STEPS[step] + int(float(el.findtext("root/root-alter", "0")))) % 12
    return pc, KINDS.get(kind, "minor" if "minor" in kind or "diminished" in kind else "major")


def shapes(chord):
    """The chord's standard shapes as sets of (string, fret)."""
    if chord is None:
        return []
    pc, quality = chord
    out = []
    for root_string, root_offset, qualities in FORMS.values():
        form = qualities.get(quality)
        if form is None:
            continue
        base = (pc - TUNING[root_string - 1] - root_offset) % 12
        while base + max(form.values()) <= MAX_FRET:
            out.append(frozenset((s, base + f) for s, f in form.items()))
            base += 12
    return out


def ways(pitches, chord=None, tuning=None):
    """[(cost, hand position, {pitch: (string, fret)})], cheapest first."""
    tuning = tuning or TUNING
    pitches = sorted(set(pitches))
    if len(pitches) > len(tuning):
        pitches = pitches[:1] + pitches[-(len(tuning) - 1):]   # the bass note and the top ones
    playable = [p for p in pitches if places(p, tuning)]
    forms = shapes(chord) if len(playable) >= SHAPE_MIN else []
    out = []
    for combo in itertools.product(*[places(p, tuning) for p in playable]):
        strings = [s for s, _ in combo]
        if len(strings) != len(set(strings)):
            continue
        fretted = [f for _, f in combo if f > 0]
        low = min(fretted) if fretted else None
        cost = FRET_COST * sum(fretted)
        if fretted:
            cost += SPAN_COST * max(0, max(fretted) - low - reach(low))
        misses = len(combo) - max(len(form & set(combo)) for form in forms) if forms else None
        if forms:
            cost += SHAPE_COST * misses
        # A string skipped inside a standard shape is the shape's own: the hand
        # holds it, so a pick may strike it.
        if len(strings) >= GAP_MIN and misses != 0:
            cost += GAP_COST * (max(strings) - min(strings) + 1 - len(strings))
        out.append((cost, low, dict(zip(playable, combo))))
    out.sort(key=lambda w: w[0])
    return out[:WAYS] or [(0.0, None, {})]


def finger(groups, chords=None, tuning=None, slides=frozenset()):
    """One {pitch: (string, fret)} per onset group, cheapest over the whole part.
    `chords`, one per group: the song's chord there (chord_from_label), or None
    (the chord shapes are guitar shapes: pass none for another tuning).
    `slides`: indices of groups that slide or are tied into the next — the two go on one string."""
    layers = [ways(g, c, tuning) for g, c in zip(groups, chords or [None] * len(groups))]
    if not layers:
        return []
    strings = lambda way: {s for s, _ in way[2].values()}
    paths = [(c, [i]) for i, (c, _, _) in enumerate(layers[0])]
    for li in range(1, len(layers)):
        nxt = []
        for j, (cost, pos, _) in enumerate(layers[li]):
            def total(entry):
                before = layers[li - 1][entry[1][-1]]
                prev = before[1]
                apart = li - 1 in slides and not strings(before) & strings(layers[li][j])
                return (entry[0] + (MOVE_COST * abs(pos - prev) if pos is not None and prev is not None else 0.0)
                        + (SLIDE_COST if apart else 0.0))
            best = min(paths, key=total)
            nxt.append((total(best) + cost, best[1] + [j]))
        paths = nxt
    _, path = min(paths, key=lambda p: p[0])
    return [layers[i][j][2] for i, j in enumerate(path)]


def midi(note, shift=0):
    p = note.find("pitch")
    return 12 * (int(p.findtext("octave")) + 1) + STEPS[p.findtext("step")] + int(float(p.findtext("alter", "0"))) + shift


NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]


def with_tab_staff(notated, name, abbreviation):
    """A score of the notated part over a TAB staff of the same bars — one part,
    two staves, braced — without on the TAB what the staff above already says."""
    import copy

    from music21 import clef, layout, stream

    tabbed = copy.deepcopy(notated)
    for el in list(tabbed.recurse().getElementsByClass(("ChordSymbol", "RehearsalMark", "MetronomeMark", "TextExpression"))):
        el.activeSite.remove(el)
    for c in list(tabbed.recurse().getElementsByClass(clef.Clef)):
        c.activeSite.replace(c, clef.TabClef())
    staves = [stream.PartStaff(), stream.PartStaff()]
    for staff_, src in zip(staves, (notated, tabbed)):
        for el in list(src):
            staff_.append(el) if isinstance(el, stream.Measure) else staff_.insert(el.offset, el)
    score = stream.Score()
    for staff_ in staves:
        score.insert(0, staff_)
    score.insert(0, layout.StaffGroup(staves, name=name, abbreviation=abbreviation, symbol="brace"))
    return score


def add_tab(path, staff="2", tuning=None):
    """Fill in the TAB staff of a written part (GUITAR by default). Returns counts."""
    tuning = tuning or GUITAR
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    part = root.find("part")
    shift = 0                                         # written to sounding, from the part's <transpose>
    t = part.find("measure/attributes/transpose")
    if t is not None:
        shift = int(t.findtext("chromatic", "0")) + 12 * int(t.findtext("octave-change", "0"))
    groups, chords = [], []                           # [(measure, [note, ...])] on the TAB staff; the chord at each
    slides = set()                                    # indices of groups starting a slide or a tie (marked on any staff)
    carry = None                                      # the chord symbol sounding as a measure begins
    for measure in part.findall("measure"):
        cursor = last = 0
        symbols, onsets = [], []                      # [(position, chord)], [(position, [note, ...])]
        sliding = set()                               # positions where a note starts a slide or a tie
        for el in measure:
            if el.tag == "harmony":
                symbols.append((cursor + int(el.findtext("offset", "0")), chord_from_harmony(el)))
            elif el.tag in ("backup", "forward"):
                cursor += (-1 if el.tag == "backup" else 1) * int(el.findtext("duration", "0"))
            elif el.tag == "note":
                if el.find("chord") is None:
                    last = cursor
                    if el.find("grace") is None:
                        cursor += int(el.findtext("duration", "0"))
                if any(m.get("type") == "start" for tag in ("slide", "glissando", "tie") for m in el.iter(tag)):
                    sliding.add(last)
                if el.findtext("staff") != staff or el.find("pitch") is None:
                    continue
                if el.find("chord") is not None and onsets and onsets[-1][0] == last:
                    onsets[-1][1].append(el)
                else:
                    onsets.append((last, [el]))
        symbols.sort(key=lambda s: s[0])
        for at, notes in onsets:
            if at in sliding:
                slides.add(len(groups))
            groups.append((measure, notes))
            chords.append(next((c for pos, c in reversed(symbols) if pos <= at), carry))
        if symbols:
            carry = symbols[-1][1]
    fingering = finger([[midi(n, shift) for n in notes] for _, notes in groups],
                       chords if tuning == GUITAR else None, tuning, slides)
    placed = left_off = 0
    for (measure, notes), places_ in zip(groups, fingering):
        kept = []
        for n in notes:
            sf = places_.get(midi(n, shift))
            if sf is None:
                continue
            notations = n.find("notations")
            if notations is None:
                notations = ET.SubElement(n, "notations")
            tech = ET.SubElement(notations, "technical")
            ET.SubElement(tech, "string").text = str(sf[0])
            ET.SubElement(tech, "fret").text = str(sf[1])
            kept.append(n)
            placed += 1
        left_off += len(notes) - len(kept)
        head = notes[0]
        if not kept:
            # Nothing playable here. The note stays, hidden — removing it, or
            # turning it into a rest, broke beams and ties that verovio then
            # crashed on.
            head.set("print-object", "no")
            for n in notes[1:]:
                n.set("print-object", "no")
            continue
        if head not in kept:
            # The head carries the beam, stem and tie; give it a playable
            # note's pitch and place, and let that chord member go.
            donor = kept[0]
            for tag in ("pitch", "notations"):
                old, new = head.find(tag), donor.find(tag)
                if old is not None:
                    head.remove(old)
                if new is not None:
                    head.insert(list(head).index(head.find("duration")) if tag == "pitch" else len(head), new)
            measure.remove(donor)
            notes = [n for n in notes if n is not donor]
            kept = [head] + kept[1:]
        for n in notes[1:]:
            if n not in kept:
                measure.remove(n)
    attributes = part.find("measure/attributes")
    if attributes is not None:
        details = ET.Element("staff-details", number=staff)
        ET.SubElement(details, "staff-lines").text = str(len(tuning))
        for line, open_string in enumerate(reversed(tuning), 1):      # MusicXML line 1 is the lowest string
            st = ET.SubElement(details, "staff-tuning", line=str(line))
            ET.SubElement(st, "tuning-step").text = NAMES[open_string % 12]
            ET.SubElement(st, "tuning-octave").text = str(open_string // 12 - 1)
        attributes.append(details)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return {"notes": placed, "left_off": left_off, "onsets": len(groups)}
