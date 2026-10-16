#!/usr/bin/env python3
"""Engrave a MusicXML score to SVG and PNG.

Usage: render_score.py <score.musicxml> <out_prefix>

Writes <out_prefix>-p1.svg / -p1.png per page, <out_prefix>.pdf with every
page, and prints a JSON summary. A part its engraver skipped — no score, a
<score>.skipped.txt beside where it would be — draws nothing.

Uses verovio rather than MuseScore or LilyPond: it installs from pip with no
external binary, and its abi3 wheel works on any Python 3.10+, so a score
renders the same on macOS and Linux. MuseScore's command line needs the whole
app installed and aborts on exit (4.7; earlier versions also crashed outright
on music21's MusicXML).
"""
import json
import re
import sys
from pathlib import Path

# Roughly A4 at 100 dpi; `scale` trades staff size against how much fits.
PAGE = {"pageWidth": 2100, "pageHeight": 2970, "scale": 45}
# A part with lyrics gets wider spacing: verovio sizes a syllable with its own
# text font, which has no Hangul, so Korean lyrics under runs of 16ths
# overlapped at the default 0.25. 0.45 cleared them (one song's two pages
# became three); a smaller lyric size was not needed.
LYRIC_SPACING = {"spacingLinear": 0.45, "lyricWordSpace": 2.5}
# verovio keeps neighbouring syllables apart by their measured width plus
# lyricWordSpace; at the default 1.2 English words under 16ths ran into each
# other ("thatthis", 50 of 378 neighbours in one song closer than a
# thin space). 2.5 cleared them without adding a page.
#
# It measures with its own Times metrics, which have no Hangul: a syllable of
# two or three Korean characters on one note was laid out narrow and drawn
# over its neighbour. For the layout each Hangul character stands in as
# STAND_IN, Times' widest letter (0.94 em; Hangul is 0.87 in Apple SD Gothic
# Neo, 1.0 in Noto Sans CJK), and the drawing gets the Korean back.
HANGUL = re.compile("[\uac00-\ud7a3]")
STAND_IN = "W"


SPACE = 120   # SMuFL units (1000 per em) between a note and its dot in a tempo mark


def outline_smufl_text(svg: str) -> str:
    """Draw music-font characters in text as shapes, so any renderer shows them.

    verovio writes the note in a tempo mark ("♩ = 96") as a character of its
    music font, Leipzig, and ships that font inside the SVG as a WOFF2
    @font-face. Browsers use it; cairosvg does not, and draws an empty box.
    The glyph outlines verovio itself uses are in its data folder, so each
    such character is replaced by its outline at the same place, and the text
    after it is moved over by the glyph's advance. Only characters that start
    their <text> are handled — the position of one mid-line would need the
    text font's metrics. That covers tempo marks, the one case music21 writes;
    a dotted note ("♩. = 58" in 6/8) is note, space, dot, so a space between
    music characters becomes a gap of SPACE units.
    """
    import os
    import re
    import xml.etree.ElementTree as ET

    import verovio

    ns = {"svg": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", ns["svg"])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    data = os.path.join(os.path.dirname(verovio.__file__), "data")
    root = ET.fromstring(svg)
    parents = {child: parent for parent in root.iter() for child in parent}

    def glyph(font, code):
        path = os.path.join(data, font, f"{code:04X}.xml")
        metrics = os.path.join(data, f"{font}.xml")
        if not (os.path.exists(path) and os.path.exists(metrics)):
            return None
        d = re.search(r' d="([^"]+)"', open(path).read())
        adv = re.search(rf'c="{code:04X}"[^>]*h-a-x="([\d.]+)"', open(metrics).read())
        return (d.group(1), float(adv.group(1))) if d and adv else None

    for text in list(root.iter(f"{{{ns['svg']}}}text")):
        runs = [el for el in text.iter() if el is not text and (el.text or "").strip()]
        if not runs or not runs[0].get("font-family"):
            continue
        first = runs[0]
        font = first.get("font-family")
        codes = [ord(c) for c in first.text.strip()]
        if not codes or not all(0xE000 <= c <= 0xF8FF or c == 0x20 for c in codes):
            continue
        shapes = [(None, SPACE) if c == 0x20 else glyph(font, c) for c in codes]
        if any(s is None for s in shapes):
            continue
        x, y = float(text.get("x", 0)), float(text.get("y", 0))
        size = float(first.get("font-size", "0").rstrip("px") or 0)
        scale = size / 1000.0                       # SMuFL fonts: 1000 units per em
        group = ET.Element(f"{{{ns['svg']}}}g", {"transform": f"translate({x},{y}) scale({scale})"})
        advance = 0.0
        for d, adv in shapes:
            if d:
                ET.SubElement(group, f"{{{ns['svg']}}}path",
                              {"transform": f"translate({advance},0) scale(1,-1)", "d": d})
            advance += adv
        parent = parents[text]
        parent.insert(list(parent).index(text), group)
        first.text = ""
        if len(runs) > 1:
            runs[1].set("x", f"{x + advance * scale:.1f}")
    return ET.tostring(root, encoding="unicode")


# SMuFL accidentals verovio puts inside chord symbols (its figured-bass set),
# and the plain Unicode characters that read the same.
SMUFL_ACCIDENTALS = {0xEA63: "𝄫", 0xEA64: "♭", 0xEA65: "♮", 0xEA66: "♯", 0xEA67: "𝄪",
                     0xE260: "♭", 0xE261: "♮", 0xE262: "♯"}


def unicode_accidentals(svg: str) -> str:
    """Write the flats and sharps inside chord symbols as ordinary text.

    A chord symbol "E♭m" comes out of verovio as three runs — "E", a Leipzig
    character, "m" — and cairosvg cannot draw the Leipzig one (see
    outline_smufl_text). Mid-line it cannot be swapped for an outline either:
    where it sits depends on the text font's metrics. So it becomes the plain
    ♭ / ♯ character at the text's size, in a font that has it.
    """
    import xml.etree.ElementTree as ET

    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    font = None
    for text in root.iter(f"{{{svg_ns}}}text"):
        runs = [el for el in text.iter(f"{{{svg_ns}}}tspan") if (el.text or "").strip()]
        for i, run in enumerate(runs):
            chars = run.text.strip()
            if not run.get("font-family") or not all(ord(c) in SMUFL_ACCIDENTALS for c in chars):
                continue
            if i == 0:
                continue                      # starts its line: outline_smufl_text's case
            font = font or font_for("♭") or "serif"
            run.text = "".join(SMUFL_ACCIDENTALS[ord(c)] for c in chars)
            run.set("font-family", font)
            if runs[i - 1].get("font-size"):
                run.set("font-size", runs[i - 1].get("font-size"))
    return ET.tostring(root, encoding="unicode")


# Advance widths of a Times-like face, in ems, on the generous side: enough to
# tell whether two chord symbols on one line run into each other. The sharp
# and flat come from whatever font has them (unicode_accidentals) — on a Mac
# Apple SD Gothic Neo, whose sharp is about 0.85 em; 0.45 left "G♯m7F♯" touching.
NARROW, WIDE = set("ijlft()/|' "), set("MWmw")
ACCIDENTALS = set("♭♯♮𝄫𝄪")


def text_width(text, size):
    em = 0.0
    for c in text:
        em += (0.3 if c in NARROW else 0.85 if c in WIDE else 0.9 if c in ACCIDENTALS
               else 0.72 if c.isupper() else 0.52 if c.isdigit() else 0.5)
    return em * size


def spread_chord_symbols(svg: str, gap=0.5, max_shift=1.5, lift=1.15) -> str:
    """Move a chord symbol when the one before it on the line runs into it.

    verovio sets a chord symbol at its beat and leaves the spacing to the
    notes: two chords a beat apart over 8th-note slashes overlap ("G♯m7F♯").
    It has no option against that, so a symbol that would touch the one
    before is pushed right, `gap` ems clear — a little after its beat, as
    hand-engraved parts do. Where that would take it more than `max_shift`
    ems from its beat (a run of chords a beat apart in a crowded line pushed
    one as far as 7 ems), it goes up a line instead (`lift` ems), where the
    same rule keeps the raised ones apart.
    """
    import xml.etree.ElementTree as ET

    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    rows = {}
    for g in root.iter(f"{{{svg_ns}}}g"):
        if "harm" not in (g.get("class") or "").split():
            continue
        text = g.find(f"{{{svg_ns}}}text")
        if text is None or text.get("x") is None:
            continue
        runs = [t for t in text.iter(f"{{{svg_ns}}}tspan") if (t.text or "").strip()]
        sizes = [float(t.get("font-size").rstrip("px")) for t in text.iter(f"{{{svg_ns}}}tspan") if t.get("font-size")]
        if not runs or not sizes:
            continue
        label = "".join(t.text.strip() for t in runs)
        x, y = float(text.get("x")), float(text.get("y"))
        rows.setdefault(round(y / 50), []).append([x, text_width(label, max(sizes)), max(sizes), g])
    for items in rows.values():
        ends = [None, None]                       # right edge of the last symbol on each line
        for x, width, size, g in sorted(items, key=lambda i: i[0]):
            need = [0.0 if e is None or x >= e + gap * size else e + gap * size - x for e in ends]
            lane = 0 if need[0] <= max_shift * size or need[0] <= need[1] else 1
            dx, dy = need[lane], (-lift * size if lane else 0.0)
            if dx or dy:
                g.set("transform", f"translate({dx:.0f} {dy:.0f}) " + (g.get("transform") or ""))
            ends[lane] = x + dx + width
    return ET.tostring(root, encoding="unicode")


# Tried first when a heading needs a font the text font lacks; any font that
# fontconfig reports as having the characters is used otherwise.
PREFERRED_FONTS = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
                   "Malgun Gothic", "Noto Sans CJK JP", "Noto Sans CJK SC"]


def font_for(char):
    """A font family on this machine that has `char`, or None."""
    import shutil
    import subprocess
    import sys

    if shutil.which("fc-list"):
        out = subprocess.run(["fc-list", f":charset={ord(char):x}", "family"],
                             capture_output=True, text=True).stdout
        families = {name.strip() for line in out.splitlines() for name in line.split(",") if name.strip()}
        for name in PREFERRED_FONTS:
            if name in families:
                return name
        if families:
            return sorted(families)[0]
    if sys.platform == "darwin":
        return "Apple SD Gothic Neo"      # ships with macOS: Hangul and Latin
    return None


def fit_text_fonts(svg: str) -> str:
    """Give text with non-Latin characters a font that has them.

    verovio sets headings in Times, which has no Hangul. A browser falls back
    to another font per character; cairosvg uses only the first family it is
    given, so a Korean song title came out as boxes. Such a <text> gets a
    font from this machine that has its characters put first.
    """
    import sys
    import xml.etree.ElementTree as ET

    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    for text in root.iter(f"{{{svg_ns}}}text"):
        content = "".join(text.itertext())
        foreign = [c for c in content if ord(c) > 0x24F and not 0xE000 <= ord(c) <= 0xF8FF
                   and c not in "—–‘’“”…♩♭♮♯𝄫𝄪"]
        if not foreign:
            continue
        font = font_for(foreign[0])
        if font:
            text.set("font-family", f"{font}, Times, serif")
        else:
            print(f"warning: no font here has {foreign[0]!r}; it will show as a box in the PNG "
                  "(install one, e.g. fonts-noto-cjk)", file=sys.stderr)
    return ET.tostring(root, encoding="unicode")


HEADING_WIDTH = 18500   # SVG units: the page is 21000 wide, the heading centred at 10000


def fit_heading(svg: str) -> str:
    """Shrink a page heading that would run off the page until it fits.

    verovio sets the heading on one centred line at a fixed size and neither
    wraps nor shrinks it; a long Korean title (72 characters) ran off
    both edges. Its width is estimated: Latin letters from verovio's own Times
    metrics (a little wider when the line is drawn in a Hangul font, see
    fit_text_fonts), every other character a full em.
    """
    import os

    import verovio

    metrics = os.path.join(os.path.dirname(verovio.__file__), "data", "text", "Times.xml")
    advance = {int(c, 16): float(a) / 2048 for c, a in
               re.findall(r'<g c="([0-9A-F]{2})"[^>]*h-a-x="([\d.]+)"', open(metrics).read())}

    def shrink(m):
        size, text = float(m.group(2)), m.group(4)
        latin = 1.15 if HANGUL.search(text) else 1.0
        width = size * sum(advance[ord(c)] * latin if ord(c) in advance else 1.0 for c in text)
        if width <= HEADING_WIDTH:
            return m.group(0)
        return f"{m.group(1)}{size * HEADING_WIDTH / width:.0f}{m.group(3)}{text}"

    def heading(m):
        return re.sub(r'(<tspan font-size=")([\d.]+)(px">)([^<]+)', shrink, m.group(0))

    return re.sub(r'<g [^>]*class="pgHead[^"]*"[^>]*>.*?</g>', heading, svg, flags=re.S)


def with_white_page(svg: str) -> str:
    """Paint a white page behind verovio's SVG.

    verovio draws black notation on no background at all. Opened in a dark
    editor theme — VS Code's image preview, for one — that is black on near
    black and the score all but disappears. Paint the page, don't rely on
    whatever the viewer puts behind it.
    """
    end = svg.find(">", svg.find("<svg")) + 1
    return svg[:end] + '<rect width="100%" height="100%" fill="white"/>' + svg[end:]


def write_pdf(svgs, path) -> None:
    """Every page in one PDF, A4 as verovio laid it out.

    cairosvg writes one PDF per SVG; here its PDF surface draws each page onto
    one shared cairo surface instead, page after page, so no PDF library is
    needed. verovio's page is PAGE["pageWidth"] tenths of a millimetre wide
    but its SVG is sized in px at another scale; output_width (px at 96 dpi)
    puts it back at that width.
    """
    import cairocffi
    from cairosvg.parser import Tree
    from cairosvg.surface import PDFSurface

    shared = []

    class Page(PDFSurface):
        def _create_surface(self, width, height):
            if not shared:
                shared.append(cairocffi.PDFSurface(str(path), width, height))
            else:
                shared[0].set_size(width, height)
            return shared[0], width, height

    width_px = PAGE["pageWidth"] / 10 / 25.4 * 96
    for svg in svgs:
        Page(Tree(url=str(svg)), None, 96, output_width=width_px)
        shared[0].show_page()
    if shared:
        shared[0].finish()


def stand_in_hangul(xml):
    """MusicXML with each Hangul character in a lyric as STAND_IN, and every
    lyric's own text, in order."""
    import html

    originals = []

    def swap(m):
        originals.append(html.unescape(m.group(2)))
        return m.group(1) + HANGUL.sub(STAND_IN, m.group(2)) + m.group(3)

    return re.sub(r"(<lyric\b[^>]*>(?:(?!</lyric>).)*?<text[^>]*>)(.*?)(</text>)", swap, xml, flags=re.S), originals


def restore_hangul(svg, originals, start):
    """Put the lyrics' own text back over their stand-ins, from originals[start]
    on, each matched to the next one it stands in for. Returns the SVG and
    where the next page starts."""
    import html

    i = start

    def put(m):
        nonlocal i
        shown = html.unescape(m.group(2))
        for j in range(i, len(originals)):
            if HANGUL.sub(STAND_IN, originals[j]).strip() == shown.strip():
                i = j + 1
                return m.group(1) + html.escape(originals[j], quote=False) + m.group(3)
        return m.group(0)

    pattern = r'(class="syl">\s*<text[^>]*>\s*<tspan[^>]*>\s*<tspan[^>]*>)([^<]*)(</tspan>)'
    return re.sub(pattern, put, svg), i


def remove_pages(prefix, pages) -> None:
    """Drop pages beyond `pages` left by an earlier, longer render — next to
    the new pages they read as part of this score — and, when there are no
    pages at all, its PDF."""
    for old in prefix.parent.glob(f"{prefix.name}-p*.*"):
        number = old.stem[len(prefix.name) + 2:]
        if old.suffix in (".svg", ".png") and number.isdigit() and int(number) > pages:
            old.unlink()
    if not pages:
        prefix.with_name(f"{prefix.name}.pdf").unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.exists():
        note = src.with_suffix(".skipped.txt")
        if not note.exists():
            raise SystemExit(f"score not found: {src}")
        remove_pages(prefix, 0)
        print(json.dumps({"source": str(src), "pages": 0, "files": [], "first_page": None, "skipped": note.read_text().strip()}))
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)

    import verovio

    tk = verovio.toolkit()
    # Each bar's number on its SVG group (data-n), so a page can follow along:
    # find the bar that is playing and mark it.
    xml = src.read_text(errors="ignore")
    sung = "<lyric" in xml
    tk.setOptions({**PAGE, **(LYRIC_SPACING if sung else {}), "svgAdditionalAttribute": ["measure@n"]})
    korean = sung and bool(HANGUL.search(xml))
    if korean:
        xml, originals = stand_in_hangul(xml)
    if not (tk.loadData(xml) if korean else tk.loadFile(str(src))):
        raise SystemExit(f"verovio could not read {src}")
    restored = 0

    pages = tk.getPageCount()
    written, svgs = [], []
    for page in range(1, pages + 1):
        svg = prefix.with_name(f"{prefix.name}-p{page}.svg")
        drawn = tk.renderToSVG(page)
        if korean:
            drawn, restored = restore_hangul(drawn, originals, restored)
        svg.write_text(with_white_page(fit_heading(fit_text_fonts(spread_chord_symbols(unicode_accidentals(
            outline_smufl_text(drawn)))))))
        written.append(str(svg))
        svgs.append(svg)

        # PNG is what makes the score reviewable without an editor. cairosvg is
        # optional: a missing system cairo should not fail the whole render.
        try:
            import cairosvg

            # verovio sizes its SVG in px, so cairosvg's dpi changes nothing;
            # scale is what makes the page (945 px wide at 1) readable.
            png = svg.with_suffix(".png")
            cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2,
                             background_color="white")
            written.append(str(png))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: PNG skipped for page {page}: {exc}", file=sys.stderr)

    # The part to print or pass on, one file. Optional like the PNGs.
    pdf = prefix.with_name(f"{prefix.name}.pdf")
    try:
        write_pdf(svgs, pdf)
        written.append(str(pdf))
    except Exception as exc:  # noqa: BLE001
        print(f"warning: PDF skipped: {exc}", file=sys.stderr)

    remove_pages(prefix, pages)

    print(json.dumps({
        "source": str(src),
        "pages": pages,
        "files": written,
        # The first page as PNG, for a UI to show the part at a glance (None without cairo).
        "first_page": next((f for f in written if f.endswith(".png")), None),
        "bytes": sum(Path(f).stat().st_size for f in written),
    }))


if __name__ == "__main__":
    main()
