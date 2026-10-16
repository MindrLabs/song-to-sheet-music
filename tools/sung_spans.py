#!/usr/bin/env python3
"""Where the singing is: the stretches a vocal transcription covers, for the
lyrics transcriber to listen to.

Usage: sung_spans.py <vocals.notes.json> [language]

Prints {"spans": [{"start_time", "end_time"}, ...], "language": ...}: the sung
notes (tools/transcribe_vocals.py) joined across rests shorter than PHRASE_GAP
and padded by PAD on each side, in seconds, for the speech-to-text component's
`span`; and the language code as given, or null when it is empty or blank (a
workflow input left out renders as a blank, which Whisper rejects).
Whisper over the whole stem invents text where only the band plays, and its
own speech detector drops rock vocals; over these stretches it heard 0.74 of
the lyrics printed in 7 songs' published parts (median) against 0.61 for the
whole stem and 0.62–0.76 with the speech detector.

With no sung notes the span is one tenth of a second at the start, so the
transcriber returns nothing at once instead of hearing the whole stem.
"""
import json
import sys
from pathlib import Path

PHRASE_GAP = 1.0   # seconds of rest that end a stretch
PAD = 0.3


def sung_spans(notes):
    spans = []
    for n in sorted(notes, key=lambda n: n["start_time"]):
        if spans and n["start_time"] - spans[-1][1] < PHRASE_GAP:
            spans[-1][1] = max(spans[-1][1], n["end_time"])
        else:
            spans.append([n["start_time"], n["end_time"]])
    return [{"start_time": round(max(0.0, a - PAD), 3), "end_time": round(b + PAD, 3)} for a, b in spans]


def main():
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__)
    notes = json.loads(Path(sys.argv[1].strip()).read_text())
    language = sys.argv[2].strip() if len(sys.argv) == 3 else ""
    print(json.dumps({"spans": sung_spans(notes) or [{"start_time": 0.0, "end_time": 0.1}],
                      "language": language or None}))


if __name__ == "__main__":
    main()
