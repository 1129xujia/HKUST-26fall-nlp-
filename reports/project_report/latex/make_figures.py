"""Rebuild vector figures from the saved report aggregates; no model calls."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK = colors.HexColor("#20252b")
GRAY = colors.HexColor("#65727c")
LIGHT = colors.HexColor("#f2f4f6")
BLUE = colors.HexColor("#3b596f")
# Embed the host's Times fonts when available. Prebuilt vector figures are
# included, so compiling the paper does not require ReportLab or these fonts.
FONT_ROOT = Path("C:/Windows/Fonts")
if (FONT_ROOT / "times.ttf").exists():
    pdfmetrics.registerFont(TTFont("Times-Roman", str(FONT_ROOT / "times.ttf")))
    pdfmetrics.registerFont(TTFont("Times-Bold", str(FONT_ROOT / "timesbd.ttf")))


def text(c, x, y, value, size=8, font="Times-Roman", align="left"):
    c.setFillColor(INK)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def arrow(c, points, color=GRAY):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(.7)
    path = c.beginPath()
    path.moveTo(*points[0])
    for p in points[1:]:
        path.lineTo(*p)
    c.drawPath(path)
    x0, y0 = points[-2]
    x, y = points[-1]
    dx, dy = x-x0, y-y0
    norm = (dx*dx+dy*dy)**.5
    ux, uy = dx/norm, dy/norm
    head = c.beginPath()
    head.moveTo(x, y)
    head.lineTo(x-4*ux+1.8*uy, y-4*uy-1.8*ux)
    head.lineTo(x-4*ux-1.8*uy, y-4*uy+1.8*ux)
    head.close()
    c.drawPath(head, stroke=0, fill=1)


def box(c, x, y, w, h, title, lines):
    c.setFillColor(LIGHT)
    c.setStrokeColor(GRAY)
    c.setLineWidth(.65)
    c.roundRect(x, y, w, h, 3, fill=1, stroke=1)
    text(c, x+w/2, y+h-12, title, 9, "Times-Bold", "center")
    for j, line in enumerate(lines):
        text(c, x+w/2, y+h-24-j*10, line, 8, align="center")


def architecture():
    c = canvas.Canvas(str(FIG / "architecture.pdf"), pagesize=(468, 145), initialFontName="Times-Roman")
    c.setTitle("Customer-service RAG application workflow")
    text(c, 234, 135, "Shared request ledger: planning, execution, evaluation, and bounded retries", 9, "Times-Bold", "center")
    box(c, 0, 69, 78, 52, "Context + plan", ["Intent + domains", "Last four turns"])
    box(c, 93, 69, 102, 52, "Native retrieval", ["Fine-tuned BGE", "Dense Top-k only"])
    box(c, 210, 69, 77, 52, "Generator", ["Evidence + query", "Cited JSON answer"])
    box(c, 302, 69, 77, 52, "Evaluator", ["Four quality scores", "Structured diagnosis"])
    box(c, 394, 69, 74, 52, "Release", ["Answer / clarify", "or handoff"])
    for a, b in [(78,93), (195,210), (287,302), (379,394)]:
        arrow(c, [(a,95), (b,95)])
    box(c, 211, 6, 168, 38, "Optimizer: supported corrective actions", ["Rewrite / adjust recall / regenerate"])
    arrow(c, [(340.5,69), (340.5,44)])
    text(c, 345, 53, "revise", 7.8)
    arrow(c, [(211,25), (144,25), (144,69)])
    text(c, 145, 48, "retrieve", 7.8)
    arrow(c, [(248,44), (248,69)])
    text(c, 252, 53, "regenerate", 7.8)
    c.save()


def axis(c, x0, y0, w, h, ticks, maxv, formatter):
    c.setLineWidth(.4)
    for tick in ticks:
        yy = y0+h*tick/maxv
        c.setStrokeColor(colors.HexColor("#d9dfe3"))
        c.line(x0, yy, x0+w, yy)
        text(c, x0-5, yy-2.5, formatter(tick), 7.5, align="right")
    c.setStrokeColor(GRAY)
    c.line(x0, y0, x0, y0+h)
    c.line(x0, y0, x0+w, y0)


def tradeoff():
    c = canvas.Canvas(str(FIG / "coverage_cost.pdf"), pagesize=(240, 213), initialFontName="Times-Roman")
    c.setTitle("Cross-domain point coverage and model token use")
    x0, w, h = 28, 205, 61
    centers = [60, 130, 200]
    text(c, 0, 202, "(a) Required-point coverage (%)", 9, "Times-Bold")
    # Legend: outlined initial, dark final; all styles remain distinct in grayscale.
    c.setStrokeColor(BLUE)
    c.setFillColor(colors.white)
    c.rect(91, 187, 9, 6, fill=1, stroke=1)
    text(c, 104, 187, "Initial", 8)
    c.setFillColor(BLUE)
    c.rect(151, 187, 9, 6, fill=1, stroke=0)
    text(c, 164, 187, "Final delivered", 8)
    y0 = 117
    axis(c, x0, y0, w, h, [0,20,40,60,80], 80, lambda v:str(v))
    initial = [48.0833333333, 56.75, 60.6666666667]
    final = [12.5, 45.0, 43.25]
    bw = 17
    for xx, a, b, label in zip(centers, initial, final, "ABC"):
        for offset, val, solid in [(-19,a,False),(2,b,True)]:
            c.setStrokeColor(BLUE)
            c.setFillColor(BLUE if solid else colors.white)
            c.setLineWidth(.8)
            c.rect(xx+offset, y0, bw, h*val/80, fill=1, stroke=1)
            text(c, xx+offset+bw/2, y0+h*val/80+3, f"{val + 1e-9:.1f}", 7, align="center")
        text(c, xx, y0-11, label, 8.5, "Times-Bold", "center")
    text(c, 0, 87, "(b) Final workflow tokens (thousands)", 9, "Times-Bold")
    y0 = 16
    axis(c, x0, y0, w, h, [0,4000,8000,12000,16000], 16000, lambda v: str(v//1000))
    values = [7312.55, 6865.8, 14755.9]
    for xx, val, label in zip(centers, values, "ABC"):
        c.setFillColor(BLUE)
        c.rect(xx-12, y0, 24, h*val/16000, stroke=0, fill=1)
        text(c, xx, y0+h*val/16000+3, f"{val/1000:.2f}", 7.5, align="center")
        text(c, xx, y0-11, label, 8.5, "Times-Bold", "center")
    c.save()


if __name__ == "__main__":
    architecture()
    tradeoff()
