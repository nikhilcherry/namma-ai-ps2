"""hk.pdf - PDFs that look designed (reports, GST invoices, certificates) from HTML, in every Indian script.

    from hk import pdf
    pdf.from_html("<h1>नमस्ते</h1><p>Report body…</p>", "report.pdf")        # headless Chrome does the layout
    pdf.from_markdown("# Weekly report\\n\\n**42** reports resolved", "weekly.pdf", title="Ward 12")
    pdf.invoice({"seller": {...}, "buyer": {...}, "items": [...], "number": "KF/2026/0142"}, "invoice.pdf")   # GST maths done
    pdf.certificates([{"name": "Kavya Hegde", "team": "Chaukanna"}, ...], "certs.pdf",
                     event="Hack4SDG 2026", line="for building {team}", verify_url="https://x.dev/c/{id}")
    pdf.report("AQI - Bengaluru", [("Summary", "Readings were **12% higher** than last week."),
                                   ("Stations", [{"station": "Hebbal", "aqi": 182}, ...])], "aqi.pdf")

    python -m hk.pdf page.html out.pdf          python -m hk.pdf notes.md out.pdf

What it handles so you don't:
    * Devanagari, Kannada, Tamil, Bengali... are shaped correctly (conjuncts, matras): the browser does the text
      layout, which no pure-Python PDF writer gets right; files are written as UTF-8 with a charset, because a
      data: URL without one turns हिन्दी into "à¤¹" mojibake
    * finds a browser by itself: HK_CHROME, Playwright's Chromium, google-chrome/chromium/edge on PATH or in the
      usual macOS/Windows locations; prints with backgrounds, no date/URL header and footer, A4 by default
    * page rules that print well: margins via @page, tables that do not split rows, headings that stay with
      their text, page numbers ("Page 2 of 5") in the footer when asked
    * GST invoices: line totals, taxable value, CGST + SGST within a state or IGST across states (from the two
      GSTIN state codes), rounding to the rupee, amount in words (Indian English, lakh/crore) - via hk.india
    * certificates in bulk: one page per person in a single PDF (or one file each), each with a QR code that
      verifies it (hk.qr), names auto-shrunk to fit

Limits:
    * needs a Chromium-based browser installed (any recent Chrome/Edge/Chromium); not wkhtmltopdf/weasyprint
    * images must be absolute URLs, data: URIs or files next to the HTML you pass with base_dir=

Single file, stdlib only, Python 3.9+. Uses hk.qr and hk.india when next to this file.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from . import qr as _qr
except ImportError:  # copied flat
    try:
        import qr as _qr  # type: ignore
    except ImportError:
        _qr = None  # type: ignore
try:
    from . import india as _india
except ImportError:
    try:
        import india as _india  # type: ignore
    except ImportError:
        _india = None  # type: ignore

__all__ = ["from_html", "from_markdown", "invoice", "certificates", "report", "find_browser", "PdfError", "amount_words"]


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

def _md(text: str) -> str:
    t = _html.escape(text)
    out, lst, table = [], [], []

    def flush():
        nonlocal lst, table
        if lst:
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in lst) + "</ul>")
            lst = []
        if table:
            head, *rows = [r for r in table if not re.fullmatch(r"\|?\s*:?-{2,}.*", r)]
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]  # noqa: E731
            numeric = re.compile(r"[₹$-]?[\d,.]+%?")  # outside the f-string: a backslash there is a SyntaxError < 3.12

            def td(c: str) -> str:
                return f'<td{" class=num" if numeric.fullmatch(c) else ""}>{c}</td>'
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells(head)) + "</tr>" +
                       "".join("<tr>" + "".join(td(c) for c in cells(r)) + "</tr>" for r in rows) + "</table>")
            table = []
    for line in t.split("\n"):
        if re.match(r"^\s*\|.*\|\s*$", line):
            table.append(line)
            continue
        m = re.match(r"^\s*[-*]\s+(.*)", line)
        if m:
            lst.append(m.group(1))
            continue
        flush()
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        if h:
            out.append(f"<h{len(h.group(1))}>{h.group(2)}</h{len(h.group(1))}>")
        elif line.strip():
            out.append(f"<p>{line}</p>")
    flush()
    s = "\n".join(out)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)


def from_markdown(md: str, out: Union[str, Path], *, title: str = "", **kw) -> Path:
    head = f"<h1>{_html.escape(title)}</h1>" if title and not md.lstrip().startswith("# ") else ""
    return from_html(head + _md(md), out, title=title or "document", **kw)


# ---------------------------------------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------------------------------------

def _qr_svg(text: str, size_mm: float = 24) -> str:
    if _qr is None:
        return ""
    svg = _qr.svg(text, "M", border=2)
    return re.sub(r"<svg ", f'<svg style="width:{size_mm}mm;height:{size_mm}mm" ', svg, count=1)


def _inr(x: float) -> str:
    neg = x < 0
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        whole = head + "," + tail
    return ("-" if neg else "") + "₹" + whole + "." + frac


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + _words(n % 100))
    for size, word in ((10 ** 7, "crore"), (10 ** 5, "lakh"), (1000, "thousand")):
        if n >= size:
            head, rest = divmod(n, size)
            return _words(head) + " " + word + ("" if rest == 0 else " " + _words(rest))
    return str(n)


def amount_words(x: float) -> str:
    """11800.5 -> 'Rupees eleven thousand eight hundred and paise fifty only' (Indian cheque style)."""
    rupees, paise = int(x), int(round((x - int(x)) * 100))
    s = "Rupees " + _words(rupees)
    if paise:
        s += " and paise " + _words(paise)
    return s[0].upper() + s[1:] + " only"


def invoice(data: Dict[str, Any], out: Union[str, Path], **kw) -> Path:
    """GST invoice. data: number, date, seller{name,address,gstin}, buyer{...}, items[{description, hsn, qty, rate,
    gst_rate (%)}], notes, upi (VPA for a pay-by-UPI QR). Intra-state -> CGST+SGST, inter-state -> IGST."""
    s, b = data.get("seller", {}), data.get("buyer", {})
    sg, bg = (s.get("gstin") or "")[:2], (b.get("gstin") or "")[:2]
    inter = bool(sg and bg and sg != bg)
    rows, taxable, tax = [], 0.0, {}
    for i, it in enumerate(data.get("items", []), 1):
        qty, rate = float(it.get("qty", 1)), float(it.get("rate", 0))
        amt = round(qty * rate, 2)
        gr = float(it.get("gst_rate", data.get("gst_rate", 18)))
        taxable += amt
        tax[gr] = tax.get(gr, 0.0) + amt * gr / 100
        rows.append(f"<tr><td>{i}</td><td>{_html.escape(str(it.get('description', '')))}</td>"
                    f"<td>{_html.escape(str(it.get('hsn', '')))}</td><td class=num>{qty:g}</td>"
                    f"<td class=num>{_inr(rate)}</td><td class=num>{gr:g}%</td><td class=num>{_inr(amt)}</td></tr>")
    tax_rows, total_tax = [], 0.0
    for gr, t in sorted(tax.items()):
        t = round(t, 2)
        total_tax += t
        if inter:
            tax_rows.append(f"<tr><td>IGST @ {gr:g}%</td><td class=num>{_inr(t)}</td></tr>")
        else:
            half = round(t / 2, 2)
            tax_rows.append(f"<tr><td>CGST @ {gr / 2:g}%</td><td class=num>{_inr(half)}</td></tr>"
                            f"<tr><td>SGST @ {gr / 2:g}%</td><td class=num>{_inr(t - half)}</td></tr>")
    gross = taxable + total_tax
    total = float(round(gross))
    roundoff = round(total - gross, 2)
    state = (_india.STATES.get(bg or sg, "") if _india else "")
    pay = ""
    if data.get("upi"):
        from urllib.parse import urlencode, quote
        uri = "upi://pay?" + urlencode({"pa": data["upi"], "pn": s.get("name", ""), "am": f"{total:.2f}", "cu": "INR",
                                        "tn": f"Invoice {data.get('number', '')}"}, quote_via=quote)  # strict UPI apps need %20
        pay = f'<div class="pay">{_qr_svg(uri, 26)}<div>Scan to pay {_inr(total)}<br><span class="muted">{_html.escape(data["upi"])}</span></div></div>'
    party = lambda p, label: (f'<div class="party"><div class="muted">{label}</div><b>{_html.escape(p.get("name", ""))}</b><br>'  # noqa: E731
                              f'{_html.escape(p.get("address", ""))}<br>{"GSTIN: " + _html.escape(p["gstin"]) if p.get("gstin") else ""}</div>')
    css = """.top{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1.5pt solid #1c1b19;padding-bottom:8pt}
.parties{display:flex;gap:16pt;margin:12pt 0}.party{flex:1}.sum{width:50%;margin-left:auto;margin-top:8pt}
.sum td{border:0;padding:3pt 6pt}.sum .grand td{border-top:1pt solid #1c1b19;font-weight:700;font-size:12pt}
.pay{display:flex;gap:10pt;align-items:center;margin-top:14pt}.words{margin-top:10pt}"""
    html = f"""<div class="top"><div><h1>Tax Invoice</h1><div class="muted">No. {_html.escape(str(data.get('number', '')))}
 · {_html.escape(str(data.get('date', '')))}{' · Place of supply: ' + _html.escape(state) if state else ''}</div></div>
 <div style="text-align:right"><b>{_html.escape(s.get('name', ''))}</b></div></div>
<div class="parties">{party(s, "From")}{party(b, "Bill to")}</div>
<table><tr><th>#</th><th>Item</th><th>HSN/SAC</th><th class=num>Qty</th><th class=num>Rate</th><th class=num>GST</th><th class=num>Amount</th></tr>{''.join(rows)}</table>
<table class="sum"><tr><td>Taxable value</td><td class=num>{_inr(taxable)}</td></tr>{''.join(tax_rows)}
{f'<tr><td>Round off</td><td class=num>{_inr(roundoff)}</td></tr>' if roundoff else ''}<tr class="grand"><td>Total</td><td class=num>{_inr(total)}</td></tr></table>
<p class="words"><b>{amount_words(total)}</b></p>{pay}
{f'<p class="muted" style="margin-top:14pt">{_html.escape(data["notes"])}</p>' if data.get("notes") else ''}"""
    return from_html(html, out, title=f"Invoice {data.get('number', '')}", css=css, **kw)


def certificates(people: Sequence[Dict[str, Any]], out: Union[str, Path], *, event: str, line: str = "",
                 title: str = "Certificate of Participation", date: str = "", signatures: Sequence[str] = (),
                 verify_url: str = "", separate: bool = False, accent: str = "#1e5c4a", **kw) -> Union[Path, List[Path]]:
    """One landscape page per person. `line` and `verify_url` may use {name}, {team}, {id}... from each person."""
    def page(p: Dict[str, Any], i: int) -> str:
        p = {"id": p.get("id", i + 1), **p}
        name = str(p.get("name", ""))
        fs = 40 if len(name) <= 22 else max(24, int(40 * 22 / len(name)))  # long names shrink instead of wrapping
        sub = _html.escape(line.format(**p)) if line else ""
        qr = _qr_svg(verify_url.format(**p), 22) if verify_url else ""
        sigs = "".join(f'<div class="sig"><div class="line"></div>{_html.escape(s)}</div>' for s in signatures)
        return (f'<section class="cert"><div class="frame"><div class="kicker">{_html.escape(event)}</div>'
                f'<h1>{_html.escape(title)}</h1><div class="muted">presented to</div>'
                f'<div class="name" style="font-size:{fs}pt">{_html.escape(name)}</div><p class="sub">{sub}</p>'
                f'<div class="foot"><div class="sigs">{sigs}</div><div class="meta">{_html.escape(date)}'
                f'{f"<div class=qr>{qr}<span>verify</span></div>" if qr else ""}</div></div></div></section>')
    css = f""".cert{{height:180mm;break-after:page;page-break-after:always;display:flex}}.cert:last-child{{break-after:auto}}
.frame{{flex:1;border:2.5pt solid {accent};outline:0.75pt solid {accent};outline-offset:-6pt;padding:18mm 22mm;display:flex;flex-direction:column;align-items:center;text-align:center}}
.kicker{{letter-spacing:.18em;text-transform:uppercase;font-size:10pt;color:{accent};font-weight:600}}
.cert h1{{font-size:30pt;margin:8pt 0 10pt}}.name{{font-weight:700;margin:6pt 0 4pt;color:{accent};letter-spacing:-.01em}}
.sub{{font-size:13pt;max-width:180mm}}.foot{{margin-top:auto;width:100%;display:flex;justify-content:space-between;align-items:flex-end}}
.sigs{{display:flex;gap:18mm}}.sig{{font-size:10pt;color:#555}}.sig .line{{width:50mm;border-top:0.75pt solid #333;margin-bottom:3pt}}
.meta{{display:flex;gap:10pt;align-items:flex-end;font-size:10pt;color:#555}}.qr{{display:flex;flex-direction:column;align-items:center;font-size:7pt}}"""
    kw.setdefault("size", "A4 landscape")
    kw.setdefault("margin", "12mm")
    if separate:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        safe = lambda v: re.sub(r"[^\w-]+", "_", str(v))  # noqa: E731 - kept out of the f-string for Python < 3.12
        return [from_html(page(p, i), d / f"{safe(p.get('name', i))}.pdf", css=css, title=title, **kw)
                for i, p in enumerate(people)]
    return from_html("".join(page(p, i) for i, p in enumerate(people)), out, css=css, title=title, **kw)


def report(title: str, sections: Sequence[Tuple[str, Any]], out: Union[str, Path], *, subtitle: str = "", **kw) -> Path:
    """Sections of (heading, content): content is markdown text, a list of dicts (table), or raw HTML via ('', html)."""
    parts = [f"<h1>{_html.escape(title)}</h1>" + (f'<p class="muted">{_html.escape(subtitle)}</p>' if subtitle else "")]
    for head, content in sections:
        if head:
            parts.append(f"<h2>{_html.escape(head)}</h2>")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            keys = list(dict.fromkeys(k for r in content for k in r))
            num = {k for k in keys if all(isinstance(r.get(k), (int, float)) for r in content if r.get(k) is not None)}
            parts.append("<table><tr>" + "".join(f'<th{" class=num" if k in num else ""}>{_html.escape(str(k))}</th>' for k in keys)
                         + "</tr>" + "".join("<tr>" + "".join(
                             f'<td{" class=num" if k in num else ""}>{_html.escape(str(r.get(k, "")))}</td>' for k in keys) + "</tr>"
                             for r in content) + "</table>")
        elif isinstance(content, str) and head == "" and content.lstrip().startswith("<"):
            parts.append(content)
        else:
            parts.append(_md(str(content)))
    kw.setdefault("page_numbers", True)
    return from_html("".join(parts), out, title=title, **kw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m hk.pdf")
    ap.add_argument("src", help=".html or .md file")
    ap.add_argument("out")
    ap.add_argument("--landscape", action="store_true")
    ap.add_argument("--size", default="A4")
    a = ap.parse_args(argv)
    p = Path(a.src)
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix.lower() in (".md", ".markdown"):
            r = from_markdown(text, a.out, size=a.size, landscape=a.landscape)
        else:
            r = from_html(text, a.out, base_dir=p.parent, size=a.size, landscape=a.landscape, title=p.stem)
    except PdfError as e:
        print(f"hk.pdf: {e}", file=sys.stderr)
        return 1
    print(f"{r}  ({r.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
