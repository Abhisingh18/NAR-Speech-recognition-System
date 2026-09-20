#!/usr/bin/env python3
"""Render training_traces_reference.md -> print-quality PDF (WeasyPrint), repo house style.
Run with the `base` conda env (has weasyprint + markdown)."""
import markdown
from weasyprint import HTML, CSS

SRC = "/speech/tomson/filler_asr/train_sa_pe_lm_head/explainer_assets/training_traces_reference.md"
OUT = "/speech/tomson/filler_asr/train_sa_pe_lm_head/explainer_assets/training_traces_reference.pdf"

raw = open(SRC).read()
lines = raw.splitlines()
hr = next(i for i, l in enumerate(lines) if l.strip() == "---")
title_md = "\n".join(lines[:hr])
body_md = "\n".join(lines[hr:])

md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
                       extension_configs={"codehilite": {"guess_lang": False, "noclasses": True,
                                                          "pygments_style": "friendly"}})
title_html = md.convert(title_md); md.reset()
body_html = md.convert(body_md)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<section class="titlepage">{title_html}
<div class="titlemeta">Fundamental Reference &middot; Filler-ASR SA Head &middot; Training &amp; Validation Traces</div>
</section>
<section class="content">{body_html}</section>
</body></html>"""

CSS_TEXT = r"""
@page {
  size: A4; margin: 20mm 18mm 20mm 18mm;
  @top-center { content: "Reading Your Training Traces — Filler-ASR SA Head";
    font-family: 'DejaVu Serif', serif; font-size: 7.5pt; color: #888; }
  @bottom-center { content: counter(page) " / " counter(pages);
    font-family: 'DejaVu Serif', serif; font-size: 8pt; color: #555; }
}
@page :first { @top-center { content: none; } }
html { font-family: 'DejaVu Serif', 'Liberation Serif', serif; font-size: 9.3pt; line-height: 1.42; color: #1a1a1a; }
body { hyphens: auto; text-align: justify; }
.titlepage { min-height: 55vh; padding-top: 8vh; border-bottom: 2.5pt solid #2c3e50; page-break-after: always; }
.titlepage h1 { font-size: 24pt; line-height: 1.2; color: #1a2733; border: none; margin: 0 0 6pt 0; text-align: left; }
.titlepage h3 { font-size: 12pt; color: #24475f; font-weight: normal; margin: 0 0 18pt 0; text-align: left; }
.titlepage p { font-size: 10pt; color: #333; text-align: left; }
.titlepage strong { color: #14324a; }
.titlemeta { margin-top: 26pt; font-size: 8.5pt; letter-spacing: .3px; color: #7a7a7a; text-transform: uppercase; }
h1, h2, h3, h4 { font-family: 'DejaVu Serif', serif; color: #14324a; page-break-after: avoid; }
.content h2 { font-size: 13.5pt; margin: 20pt 0 7pt; padding-bottom: 3pt; border-bottom: 1pt solid #b9c6d1; }
.content h3 { font-size: 11pt; margin: 13pt 0 4pt; color: #24475f; }
.content h4 { font-size: 9.8pt; margin: 10pt 0 3pt; color: #2c5169; }
p { margin: 0 0 6pt; orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 11pt; font-size: 8.0pt; page-break-inside: avoid; }
thead { background: #2c3e50; color: #fff; }
th, td { border: 0.5pt solid #c4cdd4; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { font-family: 'DejaVu Sans', sans-serif; font-size: 7.4pt; font-weight: bold; }
td { font-variant-numeric: tabular-nums; }
tbody tr:nth-child(even) { background: #eef2f5; }
table strong { color: #0b5d1e; }
pre { background: #f6f8fa; border: 0.5pt solid #d5dde3; border-left: 2.5pt solid #4a6b82; border-radius: 2pt;
  padding: 6pt 8pt; font-family: 'DejaVu Sans Mono', monospace; font-size: 7.0pt; line-height: 1.3;
  white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; margin: 6pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #eef1f3; padding: 0.5pt 2pt; border-radius: 2pt; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; background: #fafaf5; border-left: 2.5pt solid #b7a24a;
  font-size: 8.8pt; page-break-inside: avoid; }
blockquote p { margin: 2pt 0; }
ul, ol { margin: 4pt 0 8pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; }
hr { border: none; border-top: 0.75pt solid #cfd8de; margin: 12pt 0; }
h2 + p, h3 + p, h2 + table, h3 + table { page-break-before: avoid; }
"""

HTML(string=html, base_url="/").write_pdf(OUT, stylesheets=[CSS(string=CSS_TEXT)])
print("wrote", OUT)
