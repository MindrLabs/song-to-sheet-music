"""Lyrics onto a sung line: Whisper's words, as syllables, onto the melody's notes.

words.json is a list of segments, each with "words": [{"text", "start_time",
"end_time"}] — faster-whisper's word timestamps, as the lyrics-transcriber component (speech-to-text)
saves them. A Korean word becomes one syllable per Hangul character; any other
word is split at its vowel groups (a rule of thumb, no dictionary). Each
syllable gets a time, its word's span shared out evenly.

Syllables and notes are then paired in order, nearest in time: a dynamic
programme over both sequences where a pair costs the gap between the
syllable's time and the note's onset, a note left without a syllable costs
SKIP_NOTE (a melisma, or a note the transcriber split) and a syllable left
without a note SKIP_SYLLABLE. Pairs more than MAX_GAP apart are not allowed. A
syllable left over joins its nearest paired neighbour's note when that note is
within MAX_GAP, so no sung text is lost from the page.
"""
import re

# Whisper puts a word's start where its first consonant begins; the
# transcribed note starts at the pitched vowel. Against 7 songs' purchased
# parts, moving every syllable this much later took the lyrics placed on the
# part's own notes from F 0.33 to 0.49 (median; 0.1-0.2 s all within 0.03,
# 0.4 s back at 0.32), and on all 27 songs with parts from 0.31 to 0.41.
# Estimating it per song did worse.
DELAY = 0.15
SKIP_NOTE = 0.35        # seconds
SKIP_SYLLABLE = 0.5
MAX_GAP = 1.0
MAX_PER_NOTE = 3        # syllables on one note, a left-over joined in
# A "word" this long made mostly of one syllable is Whisper looping, not sung
# text (one song: "아아아…" 100 times over a single note).
LOOP_LENGTH, LOOP_SHARE = 9, 0.8
HANGUL = re.compile(r"[가-힣]")
DIGRAPHS = {"th", "sh", "ch", "ph", "wh", "gh", "ck", "qu"}


def english_syllables(word):
    """A word split at its vowel groups: one consonant before a vowel goes with
    it (a digraph such as th counts as one), a cluster splits after its first
    letter. A final silent e stays with the syllable before."""
    w = word.lower()
    groups = [m.span() for m in re.finditer(r"[aeiouy]+", w)]
    if len(groups) > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups[-1][0] == len(w) - 1:
        groups = groups[:-1]
    if len(groups) <= 1:
        return [word]
    cuts = []
    for (_, end), (start, _) in zip(groups, groups[1:]):
        consonants = start - end
        if consonants >= 2 and w[start - 2:start] in DIGRAPHS:
            consonants -= 1                       # "th" before a vowel goes with it
        cuts.append(end if consonants <= 1 else end + 1)
    pieces, last = [], 0
    for c in cuts:
        pieces.append(word[last:c]); last = c
    pieces.append(word[last:])
    return [p for p in pieces if p]


def looped(text):
    letters = [c for c in text.lower() if c.isalpha()]
    if len(letters) < LOOP_LENGTH:
        return False
    return max(letters.count(c) for c in set(letters)) / len(letters) >= LOOP_SHARE


def syllables(segments):
    """[(time, text, syllabic, word)] in order. syllabic as MusicXML's: single,
    begin, middle, end; word numbers the words, so syllables that share a note
    can be told apart as one word or several."""
    out = []
    words = (w for seg in segments for w in seg.get("words") or [])
    for word, w in enumerate(words):
        text = re.sub(r"[^\w'’]", "", w["text"]).replace("’", "'")
        if not text or looped(text):
            continue
        if HANGUL.search(text):
            parts = [c for c in text if HANGUL.match(c) or c.isalnum()]
        else:
            parts = english_syllables(text)
        n = len(parts)
        span = max(0.0, w["end_time"] - w["start_time"])
        for i, p in enumerate(parts):
            syl = "single" if n == 1 or HANGUL.match(p) else ("begin" if i == 0 else "end" if i == n - 1 else "middle")
            out.append((w["start_time"] + span * i / n + DELAY, p, syl, word))
    return out


def attach(note_times, sylls):
    """{note index: [(text, syllabic, word), ...]} — sylls paired with notes in order."""
    n, m = len(note_times), len(sylls)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < n and c + SKIP_NOTE < cost[i + 1][j]:
                cost[i + 1][j], back[i + 1][j] = c + SKIP_NOTE, (i, j, "note")
            if j < m and c + SKIP_SYLLABLE < cost[i][j + 1]:
                cost[i][j + 1], back[i][j + 1] = c + SKIP_SYLLABLE, (i, j, "syl")
            if i < n and j < m:
                gap = abs(note_times[i] - sylls[j][0])
                if gap <= MAX_GAP and c + gap < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = c + gap, (i, j, "pair")
    pairs, left = {}, []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, how = back[i][j]
        if how == "pair":
            pairs[pi] = [pj]
        elif how == "syl":
            left.append(pj)
        i, j = pi, pj
    for j in left:
        t = sylls[j][0]
        near = min(pairs, key=lambda k: abs(note_times[k] - t), default=None)
        if near is not None and abs(note_times[near] - t) <= MAX_GAP and len(pairs[near]) < MAX_PER_NOTE:
            pairs[near].append(j)
    return {k: [sylls[j][1:] for j in sorted(js)] for k, js in pairs.items()}


def lyric_text(parts):
    """One note's lyric: its syllables run together within a word and spaced
    between words ("the power", not "thepower"), with the syllabic of the
    first and last (a joined Korean syllable stays 'single')."""
    text = parts[0][0]
    for (_, _, before), (t, _, word) in zip(parts, parts[1:]):
        text += t if word == before else " " + t
    first, last = parts[0][1], parts[-1][1]
    if first == last or len(parts) == 1:
        return text, first
    starts = first in ("single", "begin")
    ends = last in ("single", "end")
    return text, "single" if starts and ends else "begin" if starts else "end" if ends else "middle"
