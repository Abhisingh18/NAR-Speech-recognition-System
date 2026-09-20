#!/usr/bin/env python3
"""Render the comparative-analysis markdown to a print-quality PDF (WeasyPrint).

Academic single-column layout: serif body, shaded/striped tables that avoid
mid-row page breaks, monospace decode examples, numbered pages + running title.
"""
import re, sys
import markdown
from weasyprint import HTML, CSS

SRC = "/speech/tomson/filler_asr/train_sa_pe_lm_head/comparative_analysis_sa8_baseline_100ep_mask.md"
OUT = "/speech/tomson/filler_asr/train_sa_pe_lm_head/comparative_analysis_sa8_baseline_100ep_mask.pdf"

raw = open(SRC).read()

# Split off the H1 title block (everything up to the first '---' hr) into a title page.
lines = raw.splitlines()
hr = next(i for i, l in enumerate(lines) if l.strip() == "---")
title_md = "\n".join(lines[:hr])
body_md = "\n".join(lines[hr:])

md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
                       extension_configs={"codehilite": {"guess_lang": False, "noclasses": True,
                                                          "pygments_style": "friendly"}})
title_html = md.convert(title_md)
md.reset()
body_html = md.convert(body_md)

# Tag the executive-summary numeric callouts nothing special; rely on CSS.
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<section class="titlepage">{title_html}
<div class="titlemeta">Comparative Analysis Report &middot; Filler-ASR / SA8+PE Head &middot; 2026-07-12</div>
</section>
<section class="content">{body_html}</section>
</body></html>"""

CSS_TEXT = r"""
@page {
  size: A4; margin: 20mm 18mm 20mm 18mm;
  @top-center { content: "Frame-Synchronous Filler ASR — Baseline vs. 100ep vs. Masking";
    font-family: 'DejaVu Serif', serif; font-size: 7.5pt; color: #888; }
  @bottom-center { content: counter(page) " / " counter(pages);
    font-family: 'DejaVu Serif', serif; font-size: 8pt; color: #555; }
}
@page :first { @top-center { content: none; } }

html { font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 9.3pt; line-height: 1.42; color: #1a1a1a; }
body { hyphens: auto; text-align: justify; }

/* ---- title page ---- */
.titlepage { min-height: 60vh; padding-top: 8vh; border-bottom: 2.5pt solid #2c3e50; page-break-after: always; }
.titlepage h1 { font-size: 21pt; line-height: 1.25; color: #1a2733; border: none; margin: 0 0 18pt 0; text-align: left; }
.titlepage p { font-size: 10pt; color: #333; text-align: left; }
.titlepage strong { color: #14324a; }
.titlemeta { margin-top: 26pt; font-size: 8.5pt; letter-spacing: .3px; color: #7a7a7a; text-transform: uppercase; }

/* ---- headings ---- */
h1, h2, h3, h4 { font-family: 'DejaVu Serif', serif; color: #14324a; page-break-after: avoid; }
.content h2 { font-size: 13.5pt; margin: 20pt 0 7pt; padding-bottom: 3pt; border-bottom: 1pt solid #b9c6d1; }
.content h3 { font-size: 11pt; margin: 13pt 0 4pt; color: #24475f; }
.content h4 { font-size: 9.8pt; margin: 10pt 0 3pt; color: #2c5169; }
p { margin: 0 0 6pt; orphans: 2; widows: 2; }

/* ---- tables ---- */
table { border-collapse: collapse; width: 100%; margin: 8pt 0 11pt; font-size: 8.2pt;
  page-break-inside: avoid; }
thead { background: #2c3e50; color: #fff; }
th, td { border: 0.5pt solid #c4cdd4; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { font-family: 'DejaVu Sans', sans-serif; font-size: 7.6pt; font-weight: bold; }
td { font-variant-numeric: tabular-nums; }
tbody tr:nth-child(even) { background: #eef2f5; }
tbody tr:hover { background: #e2ebf0; }
table strong { color: #0b5d1e; }   /* bolded 'best' cells read as highlighted */

/* ---- code / decode examples ---- */
pre { background: #f6f8fa; border: 0.5pt solid #d5dde3; border-left: 2.5pt solid #4a6b82;
  border-radius: 2pt; padding: 6pt 8pt; font-family: 'DejaVu Sans Mono', monospace;
  font-size: 7.4pt; line-height: 1.32; white-space: pre-wrap; word-break: break-word;
  page-break-inside: avoid; margin: 6pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #eef1f3;
  padding: 0.5pt 2pt; border-radius: 2pt; }
pre code { background: none; padding: 0; font-size: inherit; }

/* ---- blockquotes = decode-example callouts ---- */
blockquote { margin: 6pt 0 6pt 0; padding: 4pt 10pt; background: #fafaf5;
  border-left: 2.5pt solid #b7a24a; font-size: 8.6pt; page-break-inside: avoid; }
blockquote p { margin: 2pt 0; }
blockquote code { background: #efece0; }

/* ---- lists ---- */
ul, ol { margin: 4pt 0 8pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; }

hr { border: none; border-top: 0.75pt solid #cfd8de; margin: 12pt 0; }
a { color: #14324a; text-decoration: none; }

/* keep a heading with its first table/paragraph */
h2 + p, h3 + p, h2 + table, h3 + table { page-break-before: avoid; }
"""

HTML(string=html, base_url="/").write_pdf(OUT, stylesheets=[CSS(string=CSS_TEXT)])
print("wrote", OUT)
