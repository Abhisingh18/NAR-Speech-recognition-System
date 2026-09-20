"""
Render report.md -> report.pdf with full fidelity (all content preserved) and
embed diagram.png as Figure 1. No content is added beyond the figure + caption.
"""
import os, re
from fpdf import FPDF
from fpdf.fonts import FontFace
from fontTools.ttLib import TTFont

ROOT = os.path.dirname(os.path.abspath(__file__))
MD   = os.path.join(ROOT, "report.md")
IMG  = os.path.join(ROOT, "diagram.png")
OUT  = os.path.join(ROOT, "report.pdf")

# ---------- fonts ----------
F_REG  = "/Library/Fonts/Arial Unicode.ttf"                      # full coverage
F_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_ITAL = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
F_BI   = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
F_MONO = "/System/Library/Fonts/Supplemental/Courier New.ttf"
F_MONOB= "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

TRANSLIT = {'ℝ': 'R', '∈': ' in ', '⌈': 'ceil(', '⌉': ')', '∑': 'sum', 'Σ': 'Sum'}

def missing_set(path):
    try:
        cmap = TTFont(path).getBestCmap()
        return cmap
    except Exception:
        return None

CMAPS = {p: missing_set(p) for p in {F_REG, F_BOLD, F_ITAL, F_BI, F_MONO, F_MONOB} if os.path.exists(p)}

def fit(text, path):
    """Transliterate any glyph the chosen font lacks (regular font covers all)."""
    cmap = CMAPS.get(path)
    if cmap is None:
        return text
    out = []
    for ch in text:
        if ord(ch) < 128 or ord(ch) in cmap:
            out.append(ch)
        else:
            out.append(TRANSLIT.get(ch, '?'))
    return "".join(out)

# ---------- colors ----------
INK   = (26, 26, 26)
BLUE  = (17, 48, 79)
LINK  = (26, 79, 138)
CODE  = (140, 42, 48)
RULE  = (180, 180, 180)
CODEBG= (244, 244, 246)
THEAD = (224, 232, 240)

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-12)
        self.set_font("Body", "", 7.5)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"filler_asr — Technical Report     ·     page {self.page_no()}",
                  align="C")

pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(True, margin=16)
pdf.set_margins(18, 16, 18)

pdf.add_font("Body", "",  F_REG)
pdf.add_font("Body", "B", F_BOLD if os.path.exists(F_BOLD) else F_REG)
pdf.add_font("Body", "I", F_ITAL if os.path.exists(F_ITAL) else F_REG)
pdf.add_font("Body", "BI", F_BI if os.path.exists(F_BI) else (F_BOLD if os.path.exists(F_BOLD) else F_REG))
pdf.add_font("Mono", "",  F_MONO)
pdf.add_font("Mono", "B", F_MONOB if os.path.exists(F_MONOB) else F_MONO)

FONTPATH = {("Body",""):F_REG, ("Body","B"):F_BOLD, ("Body","I"):F_ITAL,
            ("Body","BI"):F_BI, ("Mono",""):F_MONO, ("Mono","B"):F_MONOB}

LEFT = pdf.l_margin
EPW  = pdf.epw

# ---------- inline parsing ----------
INLINE = re.compile(
    r'(?P<code>`[^`]+`)'
    r'|(?P<bold>\*\*[^*]+\*\*)'
    r'|(?P<link>\[[^\]]+\]\([^)]*\))'
    r'|(?P<ital>\*[^*]+\*)')

def parse_inline(text, base=''):
    """-> list of (text, kind), kind in {'', 'B', 'I', 'code', 'link'}.
    Recursive so code/links nested inside bold/italic are handled."""
    runs, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], base))
        if m.group('code'):
            runs.append((m.group('code')[1:-1], 'code'))          # code wins over base
        elif m.group('bold'):
            runs.extend(parse_inline(m.group('bold')[2:-2], 'B'))
        elif m.group('link'):
            lt = re.match(r'\[([^\]]+)\]', m.group('link')).group(1)
            runs.append((lt, 'link'))
        elif m.group('ital'):
            inner = 'I' if base == '' else base
            runs.extend(parse_inline(m.group('ital')[1:-1], inner))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], base))
    return runs

def plain(text):
    return "".join(t for t, _ in parse_inline(text))

def _path(st):
    return {'code': F_MONO, 'B': F_BOLD, 'I': F_ITAL}.get(st, F_REG)

def _setfont(st, size):
    if st == 'code': pdf.set_font("Mono", "", size-0.7)
    elif st == 'B':  pdf.set_font("Body", "B", size)
    elif st == 'I':  pdf.set_font("Body", "I", size)
    else:            pdf.set_font("Body", "", size)

def _segwidth(s, st, size):
    _setfont(st, size)
    return pdf.get_string_width(fit(s, _path(st)))

def flow(runs, h=5.2, size=10.0, color=INK, x_left=None):
    """Word-wrapped layout of styled runs (wraps only at spaces; handles page breaks)."""
    x_left = pdf.l_margin if x_left is None else x_left
    x_right = pdf.w - pdf.r_margin
    # flatten to (char, style), then group into word tokens / spaces / newlines
    chars = [(ch, k) for t, k in runs for ch in t]
    tokens, word = [], []
    def flush():
        if not word:
            return
        segs, buf, cs = [], word[0][0], word[0][1]
        for ch, st in word[1:]:
            if st == cs: buf += ch
            else: segs.append((buf, cs)); buf, cs = ch, st
        segs.append((buf, cs)); tokens.append(segs); word.clear()
    for ch, k in chars:
        if ch == '\n': flush(); tokens.append('NL')
        elif ch == ' ': flush(); tokens.append('SP')
        else: word.append((ch, k))
    flush()
    auto = pdf.auto_page_break
    pdf.set_auto_page_break(False)
    x, y = x_left, pdf.get_y()
    sp_w = _segwidth(' ', '', size)
    def newline():
        nonlocal x, y
        y += h
        if y + h > pdf.h - pdf.b_margin:
            pdf.set_auto_page_break(auto, pdf.b_margin)
            pdf.add_page()
            pdf.set_auto_page_break(False)
            y = pdf.t_margin
        x = x_left
    for tok in tokens:
        if tok == 'SP':
            if x > x_left: x += sp_w
            continue
        if tok == 'NL':
            newline(); continue
        w = sum(_segwidth(s, st, size) for s, st in tok)
        if x + w > x_right + 0.05 and x > x_left:
            newline()
        for s, st in tok:
            _setfont(st, size)
            pdf.set_text_color(*(CODE if st == 'code' else LINK if st == 'link' else color))
            sw = pdf.get_string_width(fit(s, _path(st)))
            pdf.set_xy(x, y)
            pdf.cell(sw + 0.3, h, fit(s, _path(st)))
            x += sw
    pdf.set_xy(x_left, y + h)
    pdf.set_text_color(*INK)
    pdf.set_auto_page_break(auto, pdf.b_margin)

def write_runs(text, h=5.2, size=10.0, color=INK, x_left=None):
    flow(parse_inline(text), h=h, size=size, color=color, x_left=x_left)

# ---------- block renderers ----------
def heading(level, text):
    sizes = {1: 18, 2: 13.5, 3: 11.5, 4: 10.5}
    sz = sizes.get(level, 10.5)
    pdf.ln(3.5 if level <= 2 else 2.0)
    if pdf.get_y() > pdf.h - 40 and level <= 2:
        pdf.add_page()
    hcolor = BLUE if level <= 2 else INK
    runs = [(t, 'B' if k in ('', 'I', 'link') else k) for t, k in parse_inline(text)]
    flow(runs, h=sz*0.52, size=sz, color=hcolor)
    pdf.ln(1.0)
    if level <= 2:
        y = pdf.get_y() + 0.4
        pdf.set_draw_color(*RULE); pdf.set_line_width(0.3 if level == 2 else 0.5)
        pdf.line(LEFT, y, LEFT+EPW, y)
        pdf.ln(2.6)
    else:
        pdf.ln(1.0)
    pdf.set_text_color(*INK)

def hr():
    pdf.ln(1.5)
    y = pdf.get_y()
    pdf.set_draw_color(*RULE); pdf.set_line_width(0.2)
    pdf.line(LEFT, y, LEFT+EPW, y)
    pdf.ln(2.5)

def paragraph(text):
    write_runs(text, h=5.2, size=10.0)
    pdf.ln(6.4)

def code_block(lines):
    pdf.ln(1.0)
    size = 7.6
    pdf.set_font("Mono", "", size)
    lh = 4.0
    txt = "\n".join(fit(l, F_MONO) for l in lines)
    pdf.set_fill_color(*CODEBG); pdf.set_text_color(40, 40, 40)
    pdf.set_draw_color(225, 225, 228); pdf.set_line_width(0.2)
    x0, y0 = LEFT, pdf.get_y()
    pdf.multi_cell(EPW, lh, txt, border=0, fill=True, new_x="LMARGIN", new_y="NEXT",
                   padding=(1.6, 2.2, 1.6, 2.2))
    pdf.set_text_color(*INK)
    pdf.ln(2.2)

def render_list(items):
    # items: list of (level, ordinal_or_None, text)
    for level, ordinal, text in items:
        indent = 3 + level*5
        bullet = (f"{ordinal}." if ordinal is not None else "•")
        bw = 6.0
        if pdf.get_y() + 6 > pdf.h - pdf.b_margin:
            pdf.add_page()
        y = pdf.get_y()
        pdf.set_font("Body", "B" if ordinal is not None else "", 10)
        pdf.set_text_color(*INK)
        pdf.set_xy(LEFT + indent, y)
        pdf.cell(bw, 5.2, bullet)
        flow(parse_inline(text), h=5.2, size=10.0, x_left=LEFT + indent + bw)
        pdf.ln(1.1)
    pdf.ln(2.4)

def render_table(rows):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    rows = [r + [""]*(ncol-len(r)) for r in rows]
    # column weights: cap long columns, but never below the longest single word
    weights = []
    for c in range(ncol):
        cells = [plain(r[c]) for r in rows]
        content_max = max(len(s) for s in cells)
        longest_word = max((len(w) for s in cells for w in s.split()), default=3)
        weights.append(max(6, longest_word + 1, min(content_max, 24)))
    pdf.ln(1.0)
    pdf.set_font("Body", "", 8.2)
    pdf.set_draw_color(170, 175, 182)
    head_style = FontFace(emphasis="BOLD", fill_color=THEAD, color=INK)
    with pdf.table(col_widths=tuple(weights), text_align="LEFT",
                   first_row_as_headings=True, headings_style=head_style,
                   line_height=4.4, width=EPW, padding=1.3,
                   borders_layout="ALL") as table:
        for r in rows:
            row = table.row()
            for cell in r:
                row.cell(fit(plain(cell), F_REG))
    pdf.ln(3.2)

def figure_page():
    pdf.add_page()
    pdf.set_font("Body", "B", 12)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "Figure 1 — Architecture & Training Schematic", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Body", "I", 8.5); pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 5, "(every shape and number is taken directly from the source code; see Sections 5-7)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    pdf.set_text_color(*INK)
    # fit image within remaining space
    avail_w = EPW
    avail_h = pdf.h - pdf.get_y() - 16
    iw, ih = 2196, 3078
    scale = min(avail_w/iw, avail_h/ih) * 72  # px->mm approx via dpi handled by w/h
    w = avail_w
    h = w * ih / iw
    if h > avail_h:
        h = avail_h; w = h * iw / ih
    x = LEFT + (EPW - w)/2
    pdf.image(IMG, x=x, y=pdf.get_y(), w=w, h=h)

# ---------- block parser ----------
def is_table_sep(line):
    s = line.strip()
    return bool(re.fullmatch(r'\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?', s))

def split_row(line):
    s = line.strip()
    if s.startswith('|'): s = s[1:]
    if s.endswith('|'): s = s[:-1]
    return [c.strip() for c in s.split('|')]

with open(MD, encoding='utf-8') as f:
    lines = f.read().split('\n')

pdf.add_page()
i, n = 0, len(lines)
title_done = False
figure_inserted = False

while i < n:
    line = lines[i]
    stripped = line.strip()

    # fenced code
    if stripped.startswith("```"):
        i += 1; buf = []
        while i < n and not lines[i].strip().startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        code_block(buf); continue

    # heading
    m = re.match(r'^(#{1,6})\s+(.*)$', line)
    if m:
        level = len(m.group(1)); text = m.group(2).strip()
        # insert the diagram right before Section 1
        if (not figure_inserted) and re.match(r'^1\.\s', text):
            figure_page(); figure_inserted = True
            pdf.add_page()
        heading(level, text)
        i += 1; continue

    # horizontal rule
    if re.fullmatch(r'(-{3,}|\*{3,}|_{3,})', stripped):
        hr(); i += 1; continue

    # table
    if '|' in line and i+1 < n and is_table_sep(lines[i+1]):
        header = split_row(line)
        i += 2
        body = []
        while i < n and '|' in lines[i] and lines[i].strip():
            body.append(split_row(lines[i])); i += 1
        render_table([header] + body); continue

    # list block
    if re.match(r'^(\s*)([-*]|\d+\.)\s+', line):
        items = []
        while i < n and re.match(r'^(\s*)([-*]|\d+\.)\s+', lines[i]):
            mm = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', lines[i])
            indent = len(mm.group(1)); level = indent // 2
            mark = mm.group(2)
            ordinal = mark[:-1] if mark[0].isdigit() else None
            items.append((level, ordinal, mm.group(3)))
            i += 1
        render_list(items); continue

    # blank
    if stripped == '':
        i += 1; continue

    # paragraph (gather consecutive plain lines)
    buf = [line]; i += 1
    while i < n:
        nx = lines[i]; ns = nx.strip()
        if ns == '' or re.match(r'^#{1,6}\s', nx) or nx.strip().startswith("```") \
           or re.fullmatch(r'(-{3,}|\*{3,}|_{3,})', ns) \
           or re.match(r'^(\s*)([-*]|\d+\.)\s+', nx) \
           or ('|' in nx and i+1 < n and is_table_sep(lines[i+1])):
            break
        buf.append(nx); i += 1
    paragraph(" ".join(b.strip() for b in buf))

pdf.output(OUT)
print("wrote", OUT, "pages:", pdf.page_no())
