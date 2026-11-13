"""HTML to PDF with headless Chrome/Edge/Chromium (trimmed from a larger helper: invoices, certificates and
Markdown were removed; only what tools/report.py needs is left).

    import hk_pdf; hk_pdf.from_html("<h1>नमस्ते</h1>", "report.pdf")

Needs a Chromium-based browser installed (HK_CHROME, Playwright's Chromium, or one on PATH). Stdlib only.
"""
from __future__ import annotations

import html as _html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Union


class PdfError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------
# The browser
# ---------------------------------------------------------------------------------------------------------

def find_browser() -> str:
    env = os.environ.get("HK_CHROME")
    if env and Path(env).exists():
        return env
    home = Path.home()
    cands = sorted(home.glob(".cache/ms-playwright/chromium-*/chrome-linux*/chrome"), reverse=True)
    cands += sorted(home.glob("Library/Caches/ms-playwright/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium"), reverse=True)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge", "chrome"):
        p = shutil.which(name)
        if p:
            cands.append(Path(p))
    cands += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
              Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
              Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
              Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"]
    for c in cands:
        if Path(c).exists():
            return str(c)
    raise PdfError("no Chromium-based browser found: install Google Chrome/Chromium, run `python -m playwright install "
                   "chromium`, or set HK_CHROME=/path/to/chrome")


BASE_CSS = """
@page { size: %(size)s; margin: %(margin)s; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 11pt/1.5 "Noto Sans", "Noto Sans Devanagari", "Noto Sans Kannada", "Noto Sans Tamil",
       "Noto Sans Telugu", "Noto Sans Bengali", "Noto Sans Gujarati", "Noto Sans Gurmukhi", "Noto Sans Malayalam",
       "Noto Sans Oriya", system-ui, sans-serif; color: #1c1b19; }
h1, h2, h3 { line-height: 1.2; break-after: avoid; page-break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 6pt; letter-spacing: -0.01em; } h2 { font-size: 14pt; margin: 18pt 0 6pt; }
table { border-collapse: collapse; width: 100%%; font-size: 10pt; } tr { break-inside: avoid; page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #d9d6cf; padding: 5pt 6pt; text-align: left; vertical-align: top; }
th { color: #5f5c55; font-weight: 600; } td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
img { max-width: 100%%; } p { margin: 0 0 6pt; } .muted { color: #6b6860; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 9.5pt; background: #f1efe9; padding: 0 2pt; }
pre { background: #f1efe9; padding: 8pt; white-space: pre-wrap; font-size: 9pt; }
"""


def _wrap(body: str, title: str, css: str, size: str, margin: str, lang: str) -> str:
    if re.search(r"<html[\s>]", body, re.I):
        doc = body
        if not re.search(r"<meta[^>]+charset", doc, re.I):
            doc = re.sub(r"(<head[^>]*>)", r'\1<meta charset="utf-8">', doc, count=1, flags=re.I) if re.search(
                r"<head[\s>]", doc, re.I) else doc.replace("<html", '<html><head><meta charset="utf-8"></head>', 1)
        return doc
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{_html.escape(title)}</title>'
            f"<style>{BASE_CSS % {'size': size, 'margin': margin}}{css}</style></head><body>{body}</body></html>")


def from_html(html: str, out: Union[str, Path], *, title: str = "document", css: str = "", size: str = "A4",
              margin: str = "16mm 14mm", landscape: bool = False, base_dir: Optional[Union[str, Path]] = None,
              lang: str = "en", page_numbers: bool = False, timeout: float = 60) -> Path:
    """Render HTML (a fragment or a full document) to a PDF file with headless Chrome. Returns the path."""
    out = Path(out).resolve()
    if landscape and "landscape" not in size:
        size = f"{size} landscape"
    if page_numbers:
        # Chrome's own header/footer would add the URL and date too; a fixed footer with counters is cleaner
        css += ('@page { @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: 8pt; '
                'color: #888; } }')
    doc = _wrap(html, title, css, size, margin, lang)
    d = Path(base_dir).resolve() if base_dir else Path(tempfile.mkdtemp(prefix="hk-pdf-"))
    src = d / f".hk-pdf-{uuid.uuid4().hex[:8]}.html"
    src.write_text(doc, encoding="utf-8")
    exe = find_browser()
    args = [exe, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer", "--hide-scrollbars",
            "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
            f"--user-data-dir={tempfile.mkdtemp(prefix='hk-pdf-profile-')}", f"--print-to-pdf={out}", src.as_uri()]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise PdfError(f"the browser did not finish in {timeout:.0f}s (an image or font that never loads?)") from None
    finally:
        try:
            src.unlink()
        except OSError:
            pass
    if not out.exists() or out.stat().st_size < 200 or out.read_bytes()[:4] != b"%PDF":
        raise PdfError(f"no PDF produced ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return out


# ---------------------------------------------------------------------------------------------------------
# Markdown (small, safe)
# ---------------------------------------------------------------------------------------------------------

