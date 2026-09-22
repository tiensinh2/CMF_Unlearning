"""
Build CMF Unlearning Analysis presentation — v2
Full visual deck with diagrams, charts, and figures.
Run: python _build_slides_v2.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree
import copy

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG_DARK   = RGBColor(0x0F, 0x1B, 0x2D)   # deep navy
C_BG_MID    = RGBColor(0x16, 0x27, 0x3E)   # mid navy
C_BG_LIGHT  = RGBColor(0xF0, 0xF4, 0xF8)   # slide bg
C_ACCENT    = RGBColor(0x2E, 0x86, 0xC1)   # blue
C_ACCENT2   = RGBColor(0x1A, 0xBC, 0x9C)   # teal
C_WARN      = RGBColor(0xE7, 0x4C, 0x3C)   # red
C_WARN_SOFT = RGBColor(0xF5, 0xB7, 0xB1)   # pink
C_OK        = RGBColor(0x27, 0xAE, 0x60)   # green
C_OK_SOFT   = RGBColor(0xA9, 0xDF, 0xBA)   # light green
C_GOLD      = RGBColor(0xF3, 0x9C, 0x12)   # amber
C_GOLD_SOFT = RGBColor(0xFA, 0xD7, 0xA0)   # light amber
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK     = RGBColor(0x00, 0x00, 0x00)
C_GRAY      = RGBColor(0x95, 0xA5, 0xA6)
C_GRAY_D    = RGBColor(0x2C, 0x3E, 0x50)   # dark text
C_GRAY_L    = RGBColor(0xEC, 0xF0, 0xF1)   # rule lines
C_PURPLE    = RGBColor(0x8E, 0x44, 0xAD)
C_PURPLE_S  = RGBColor(0xD7, 0xBD, 0xE2)

W, H = Inches(13.33), Inches(7.5)   # 16:9 widescreen

# ── Helpers ───────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs, bg=C_BG_LIGHT):
    layout = prs.slide_layouts[6]   # completely blank
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide

def txb(slide, left, top, width, height,
        text="", size=18, bold=False, italic=False,
        color=C_GRAY_D, align=PP_ALIGN.LEFT,
        bg=None, wrap=True, line_spacing=None):
    tf_box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tf_box.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic= italic
    run.font.color.rgb = color
    if bg:
        tf_box.fill.solid()
        tf_box.fill.fore_color.rgb = bg
    return tf_box

def multiline_txb(slide, left, top, width, height,
                  lines, size=14, bold=False, color=C_GRAY_D,
                  align=PP_ALIGN.LEFT, bg=None, line_spacing=1.2):
    """lines = list of (text, bold, color) tuples or plain strings"""
    tb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            txt, bld, clr = item, bold, color
        else:
            txt = item[0]
            bld = item[1] if len(item) > 1 else bold
            clr = item[2] if len(item) > 2 else color
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        run = p.add_run()
        run.text = txt
        run.font.size  = Pt(size)
        run.font.bold  = bld
        run.font.color.rgb = clr
    if bg:
        tb.fill.solid()
        tb.fill.fore_color.rgb = bg
    return tb

def rect(slide, left, top, width, height, fill=C_ACCENT,
         line_color=None, line_width=None, radius=False):
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape

def circle(slide, left, top, diam, fill=C_ACCENT, line_color=None):
    shape = slide.shapes.add_shape(
        9,   # MSO_SHAPE_TYPE.OVAL
        Inches(left), Inches(top), Inches(diam), Inches(diam))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape

def arrow_right(slide, left, top, width, height=0.04,
                fill=C_GRAY):
    """Draw a simple right-pointing arrow using connector."""
    line = slide.shapes.add_connector(
        1,  # straight
        Inches(left), Inches(top + height/2),
        Inches(left + width), Inches(top + height/2))
    line.line.color.rgb = fill
    line.line.width = Pt(2)
    return line

def section_header(slide, title, subtitle="", accent=C_ACCENT):
    rect(slide, 0, 0, 13.33, 1.5, fill=accent)
    txb(slide, 0.4, 0.15, 12.5, 0.8,
        text=title, size=32, bold=True, color=C_WHITE)
    if subtitle:
        txb(slide, 0.4, 0.95, 12.5, 0.5,
            text=subtitle, size=15, color=C_WHITE, italic=True)

def slide_number(slide, n):
    txb(slide, 12.5, 7.1, 0.8, 0.35,
        text=str(n), size=11, color=C_GRAY, align=PP_ALIGN.RIGHT)

# ── Data ─────────────────────────────────────────────────────────────────────

METHODS = [
    ("grad_ascent_descent",  "NegGrad+",     C_GOLD,    C_GOLD_SOFT,  "Neutral",       "Unbounded, no target",         "−"),
    ("random_label",         "Random Label", C_ACCENT,  RGBColor(0xAB,0xCF,0xE6), "Partial", "Random, changes each epoch",  "~"),
    ("salun",                "SalUn",        C_ACCENT2, RGBColor(0x96,0xD8,0xCC), "Partial", "Same as random_label + mask", "~"),
    ("scrub",                "SCRUB",        C_WARN,    C_WARN_SOFT,  "Very Bad",      "Fixed KL target, consistent",  "✗✗"),
    ("tarun",                "TARUN",        C_PURPLE,  C_PURPLE_S,   "Worst",         "Optimised adversarial noise",  "✗✗✗"),
]

# NCC forget: No-CMF, CMF-Static, Post-hoc (k=5 retain_only), Oracle
PERF = {
    "grad_ascent_descent": dict(no_cmf=55.65, static=57.33, posthoc=76.23, oracle=67.30),
    "random_label":        dict(no_cmf=97.28, static=87.62, posthoc=78.31, oracle=67.30),
    "salun":               dict(no_cmf=96.54, static=84.46, posthoc=66.08, oracle=67.30),
    "scrub":               dict(no_cmf=65.64, static=34.40, posthoc=25.31, oracle=67.30),
    "tarun":               dict(no_cmf=71.82, static=13.05, posthoc=11.84, oracle=67.30),
}

# Output Forget Acc (lower = better unlearning)
OUT_FORGET = {
    "static":  {"grad_ascent_descent":51.46, "random_label":87.27, "salun":83.90, "scrub":32.54, "tarun":13.15},
    "posthoc": {"grad_ascent_descent":45.96, "random_label":78.31, "salun":66.08, "scrub": 4.65, "tarun": 0.03},
}

# ── Build ─────────────────────────────────────────────────────────────────────

def build():
    prs = new_prs()

    # ── SLIDE 1: Title ────────────────────────────────────────────────────────
    s = blank_slide(prs, C_BG_DARK)
    # decorative gradient strip
    rect(s, 0, 5.8, 13.33, 1.7, fill=C_ACCENT)
    rect(s, 0, 5.8, 13.33, 0.06, fill=C_ACCENT2)
    # circuit-board decorative dots
    for i, x in enumerate([0.5, 2.1, 3.8, 5.5, 7.2, 8.9, 10.6, 12.2]):
        circle(s, x, 5.65, 0.12, fill=C_ACCENT2)
    txb(s, 0.7, 0.6, 11.9, 1.2,
        text="CMF Method Compatibility Analysis",
        size=40, bold=True, color=C_WHITE)
    txb(s, 0.7, 1.85, 11.9, 0.7,
        text="Algorithmic Mechanisms of 5 Unlearning Methods & Why They (Don't) Work with CMF",
        size=19, color=C_ACCENT2, italic=True)
    txb(s, 0.7, 2.65, 11.9, 0.55,
        text="CIFAR-10 · ResNet-18 · Whole-Class Unlearning · 10 Forget Classes · Seed 0",
        size=15, color=C_GRAY)
    # 3 key stat boxes
    for i, (val, label, clr) in enumerate([
        ("5", "Methods Analyzed", C_ACCENT),
        ("50", "Experiments", C_ACCENT2),
        ("−58.77 pts", "Max CMF Damage (TARUN)", C_WARN),
    ]):
        bx = 0.7 + i * 4.1
        rect(s, bx, 3.5, 3.7, 1.5, fill=C_BG_MID)
        txb(s, bx+0.15, 3.6, 3.4, 0.7, text=val,
            size=30, bold=True, color=clr, align=PP_ALIGN.CENTER)
        txb(s, bx+0.15, 4.2, 3.4, 0.5, text=label,
            size=12, color=C_GRAY, align=PP_ALIGN.CENTER)
    txb(s, 0.7, 5.95, 11.9, 0.5,
        text="Based on: Gao et al. (2026) – An Illusion of Unlearning? · arXiv:2604.08271v1",
        size=12, color=C_WHITE, align=PP_ALIGN.CENTER)
    slide_number(s, 1)

    # ── SLIDE 2: The Illusion — Core Concept ─────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "The Illusion of Unlearning", "Why output accuracy alone is misleading", C_ACCENT)
    # Diagram: 3 phases
    phases = [
        ("θ₀\nFull Model", "Retain: 94%\nForget: 94%", C_GRAY_D, "Trained"),
        ("θ_u\nUnlearned", "Retain: 92%\nForget: 13%\n(Output)", C_WARN, "After Unlearn"),
        ("Linear\nProbe", "Retain: 92%\nForget: 43%\n(Recovered!)", C_PURPLE, "Probe Result"),
    ]
    for i, (title, stats, clr, label) in enumerate(phases):
        bx = 1.0 + i * 3.7
        rect(s, bx, 1.7, 2.9, 2.6, fill=clr)
        txb(s, bx+0.1, 1.75, 2.7, 0.75, text=title,
            size=17, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        txb(s, bx+0.1, 2.5, 2.7, 1.4, text=stats,
            size=13, color=C_WHITE, align=PP_ALIGN.CENTER)
        txb(s, bx+0.1, 4.35, 2.7, 0.35, text=label,
            size=11, color=C_GRAY, align=PP_ALIGN.CENTER, italic=True)
        if i < 2:
            txb(s, bx + 3.0, 2.8, 0.6, 0.4, text="→",
                size=28, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)
    # Key insight box
    rect(s, 0.5, 4.8, 12.3, 1.4, fill=RGBColor(0xFF,0xF3,0xCD))
    txb(s, 0.7, 4.88, 12.0, 0.45,
        text="⚡ Core Finding: Representations persist even when output accuracy drops",
        size=15, bold=True, color=C_GRAY_D)
    multiline_txb(s, 0.7, 5.35, 12.0, 0.8,
        lines=[
            ("NCC (Nearest Class Center) is the ground-truth signal: "
             "it reads directly from the feature space — bypassing the output head entirely.", False, C_GRAY_D),
        ], size=13)
    # NCC diagram
    rect(s, 10.7, 1.65, 2.3, 2.7, fill=C_BG_DARK)
    txb(s, 10.8, 1.7, 2.1, 0.4, text="NCC Forget Acc",
        size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    labels = [("Oracle", 67.3, C_OK), ("TARUN", 13.05, C_PURPLE),
              ("SCRUB", 34.40, C_WARN), ("NegGrad+", 57.32, C_GOLD)]
    for j, (nm, val, clr) in enumerate(labels):
        y = 2.2 + j*0.5
        bar_w = val / 100 * 2.0
        rect(s, 10.8, y, bar_w, 0.28, fill=clr)
        txb(s, 10.8, y, 1.8, 0.28, text=f" {nm}",
            size=9, color=C_WHITE)
        txb(s, 12.75, y, 0.3, 0.28, text=f"{val:.0f}%",
            size=9, bold=True, color=C_WHITE)
    slide_number(s, 2)

    # ── SLIDE 3: Method Overview Table ───────────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "5 Unlearning Methods — Algorithm Overview & CMF Compatibility",
                   "Each method has a unique unlearning signal; CMF amplifies those it can track", C_GRAY_D)
    # Column headers
    headers = ["Method", "Signal Type", "Bounded?", "Consistent Direction?", "Real Data?", "CMF Compat."]
    col_ws  = [2.2, 2.6, 1.3, 2.6, 1.3, 1.8]
    col_xs  = [0.2]
    for w in col_ws[:-1]:
        col_xs.append(col_xs[-1] + w)
    for ci, (h, x, cw) in enumerate(zip(headers, col_xs, col_ws)):
        rect(s, x, 1.55, cw-0.05, 0.4, fill=C_ACCENT)
        txb(s, x+0.05, 1.57, cw-0.1, 0.37, text=h,
            size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    rows = [
        ("grad_ascent_descent\n(NegGrad+)", "−CE on true labels\n(Gradient Ascent)", "✗ (unbounded)", "✓ but saturates fast", "✓", "Neutral"),
        ("random_label",                   "CE on random wrong\nlabels (Descent!)",  "✓",             "✗ random each epoch",  "✓", "Partial ~"),
        ("salun",                          "Same as random_label\n+ saliency mask",  "✓",             "✗ + mask freezes 50%", "✓", "Partial ~"),
        ("scrub (SCRUB)",                  "−KL to fixed teacher\n(Max-step)",       "✓ (saturates)", "✓✓ consistent 2ep",   "✓", "Very Bad ✗"),
        ("tarun (UNSIR)",                  "CE on adversarial\nnoise inputs",        "✓",             "✓✓✓ 3+3 epochs",      "✗ noise", "WORST ✗✗"),
    ]
    row_colors = [C_GOLD_SOFT, RGBColor(0xD6,0xEB,0xF7), RGBColor(0xD5,0xF5,0xF0),
                  C_WARN_SOFT, C_PURPLE_S]
    compat_colors = [C_GOLD, C_ACCENT, C_ACCENT2, C_WARN, C_PURPLE]
    for ri, (row, rbg, cbg) in enumerate(zip(rows, row_colors, compat_colors)):
        y = 2.0 + ri * 0.88
        bg = rbg if ri % 2 == 0 else C_BG_LIGHT
        for ci, (cell, x, cw) in enumerate(zip(row, col_xs, col_ws)):
            cell_bg = cbg if ci == 5 else bg
            cell_clr = C_WHITE if ci == 5 else C_GRAY_D
            rect(s, x, y, cw-0.05, 0.82, fill=cell_bg)
            txb(s, x+0.05, y+0.05, cw-0.1, 0.75,
                text=cell, size=10, color=cell_clr,
                bold=(ci == 5))
    slide_number(s, 3)

    # ── SLIDE 4: grad_ascent_descent ─────────────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "grad_ascent_descent (NegGrad+)", "Unbounded ascent loss — saturates before CMF can amplify", C_GOLD)
    # Algorithm box
    rect(s, 0.3, 1.6, 5.8, 3.5, fill=C_BG_DARK)
    txb(s, 0.45, 1.65, 5.5, 0.4, text="Algorithm", size=13, bold=True, color=C_ACCENT2)
    code = [
        "loss_ascent = −0.5 × CE(z_f @ W_fixed.T, y_f)",
        "loss_retain =  1.0 × CE(z_r @ W_fixed.T, y_r)",
        "∇ = ∇_ascent + ∇_retain",
        "θ ← θ − lr · ∇  [3 epochs]",
        "",
        "W_fixed = CMFweights.detach()  # W frozen",
        "# after each epoch: W ← recompute_cmf()",
    ]
    for i, ln in enumerate(code):
        clr = C_ACCENT2 if "W_fixed" in ln or "recompute" in ln else C_WHITE
        txb(s, 0.45, 2.1 + i*0.34, 5.5, 0.33,
            text=ln, size=10, color=clr)
    # NCC bar chart
    rect(s, 6.3, 1.6, 6.7, 3.5, fill=C_BG_DARK)
    txb(s, 6.45, 1.65, 6.4, 0.4, text="NCC Forget Accuracy (%)", size=13, bold=True, color=C_WHITE)
    bars = [("No CMF", 55.65, C_GRAY), ("CMF-Static", 57.33, C_GOLD), ("Post-hoc", 76.23, C_ACCENT2),
            ("Oracle", 67.30, C_OK)]
    max_val = 100
    for bi, (nm, val, clr) in enumerate(bars):
        bx = 6.6 + bi * 1.55
        bar_h = val / max_val * 2.2
        by = 4.7 - bar_h
        rect(s, bx, by, 1.2, bar_h, fill=clr)
        txb(s, bx, 4.72, 1.2, 0.35, text=nm, size=9, color=C_WHITE, align=PP_ALIGN.CENTER)
        txb(s, bx, by - 0.35, 1.2, 0.33, text=f"{val:.1f}%",
            size=11, bold=True, color=clr, align=PP_ALIGN.CENTER)
    # Key insight
    rect(s, 0.3, 5.25, 12.7, 1.7, fill=RGBColor(0xFF,0xF8,0xE1))
    txb(s, 0.5, 5.3, 12.4, 0.45,
        text="⚠ Key Insight: CMF is NOT the main problem here", size=14, bold=True, color=C_GOLD)
    txb(s, 0.5, 5.75, 12.4, 1.1,
        text="NegGrad+'s unbounded −CE loss already over-erases (NCC=55.65 < Oracle=67.30) "
             "BEFORE CMF touches anything. Illusion-gap = −11.65 (the only negative gap in the benchmark). "
             "Post-hoc helps by replacing the rigid CMF classifier with a learned one on existing features.",
        size=12, color=C_GRAY_D)
    slide_number(s, 4)

    # ── SLIDE 5: random_label & SalUn ────────────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "random_label & SalUn", "Partial compatibility — inconsistent direction limits CMF amplification", C_ACCENT)
    # Side-by-side panels
    for pi, (name, code_lines, ncc_vals, insight) in enumerate([
        ("random_label",
         ["y_train[forget_mask] = random_choice(non_forget_classes)",
          "loss = CE(z @ W_fixed.T, y_train)  # standard descent",
          "# target re-sampled EVERY epoch → random walk"],
         [("No CMF",97.28,C_GRAY),("Static",87.62,C_ACCENT),("Post-hoc",78.31,C_ACCENT2),("Oracle",67.30,C_OK)],
         "Random walk in feature space: CMF has no consistent\ndirection to amplify → partial progress only"),
        ("SalUn",
         ["mask = top_50%(|∇CE(θ₀, forget_data)|)  # computed once",
          "p.grad *= mask   # 50% of params blocked",
          "# Same relabel as random_label underneath"],
         [("No CMF",96.54,C_GRAY),("Static",84.46,C_ACCENT),("Post-hoc",66.08,C_ACCENT2),("Oracle",67.30,C_OK)],
         "Fixed mask from θ₀ + random walk → less movement than\nrandom_label → slightly weaker NCC result"),
    ]):
        px = 0.3 + pi * 6.5
        pw = 6.3
        # Header strip
        rect(s, px, 1.55, pw, 0.38, fill=C_ACCENT if pi==0 else C_ACCENT2)
        txb(s, px+0.1, 1.57, pw-0.2, 0.34, text=name,
            size=14, bold=True, color=C_WHITE)
        # Code
        rect(s, px, 1.93, pw, 1.4, fill=C_BG_DARK)
        for ci, ln in enumerate(code_lines):
            txb(s, px+0.15, 2.0+ci*0.42, pw-0.3, 0.4,
                text=ln, size=10, color=C_WHITE)
        # Bar chart
        rect(s, px, 3.38, pw, 2.0, fill=RGBColor(0xF5,0xF5,0xF5))
        txb(s, px+0.1, 3.42, pw-0.2, 0.35,
            text="NCC Forget Acc (%)", size=11, bold=True, color=C_GRAY_D)
        for bi, (nm, val, clr) in enumerate(ncc_vals):
            bx = px + 0.2 + bi * (pw-0.5)/4
            bar_h = val/100 * 1.3
            by = 5.1 - bar_h
            rect(s, bx, by, (pw-0.5)/4-0.1, bar_h, fill=clr)
            txb(s, bx, 5.13, (pw-0.5)/4-0.1, 0.28, text=nm,
                size=8, color=C_GRAY_D, align=PP_ALIGN.CENTER)
            txb(s, bx, by-0.32, (pw-0.5)/4-0.1, 0.3,
                text=f"{val:.1f}%", size=9, bold=True, color=clr,
                align=PP_ALIGN.CENTER)
        # Insight
        rect(s, px, 5.45, pw, 1.5, fill=RGBColor(0xE8,0xF4,0xFD))
        txb(s, px+0.1, 5.5, pw-0.2, 1.35, text=insight,
            size=12, color=C_GRAY_D)
    slide_number(s, 5)

    # ── SLIDE 6: SCRUB — Feedback Loop ───────────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "SCRUB — The Positive Feedback Loop",
                   "Best unlearning signal alone, worst outcome with CMF", C_WARN)
    # Feedback loop diagram
    loop_x, loop_y = 0.3, 1.6
    boxes = [
        (loop_x+0.2, loop_y+0.1, 2.6, 0.9, "Max-step:\n−KL(student ‖ teacher)", C_WARN),
        (loop_x+0.2, loop_y+1.35, 2.6, 0.9, "Feature shifts\naway from teacher", C_GRAY_D),
        (loop_x+0.2, loop_y+2.6, 2.6, 0.9, "CMF recomputes\nclass-mean W", C_ACCENT),
        (loop_x+0.2, loop_y+3.85, 2.6, 0.9, "Next epoch KL\nnow uses drifted W", C_PURPLE),
    ]
    for bx, by, bw, bh, txt, clr in boxes:
        rect(s, bx, by, bw, bh, fill=clr)
        txb(s, bx+0.1, by+0.05, bw-0.2, bh-0.1,
            text=txt, size=12, color=C_WHITE, align=PP_ALIGN.CENTER)
    # Arrows between boxes
    for i in range(len(boxes)-1):
        _, by, _, bh, _, _ = boxes[i]
        next_by = boxes[i+1][1]
        txb(s, loop_x+1.2, by+bh+0.04, 0.6, 0.27,
            text="↓", size=18, bold=True, color=C_WARN, align=PP_ALIGN.CENTER)
    # Loop-back arrow label
    txb(s, loop_x+0.0, loop_y+2.25, 0.5, 0.35,
        text="↻", size=30, bold=True, color=C_WARN)
    txb(s, loop_x+3.1, loop_y+2.1, 0.7, 0.35,
        text="LOOP", size=11, bold=True, color=C_WARN)
    # Performance panel
    rect(s, 4.0, 1.6, 5.0, 5.25, fill=C_BG_DARK)
    txb(s, 4.15, 1.65, 4.7, 0.4,
        text="NCC Forget Accuracy (%)", size=13, bold=True, color=C_WHITE)
    scrub_bars = [("No CMF",65.64,C_GRAY),("CMF-Static",34.40,C_WARN),
                  ("Post-hoc",25.31,C_PURPLE),("Oracle",67.30,C_OK)]
    for bi, (nm, val, clr) in enumerate(scrub_bars):
        bx = 4.2 + bi * 1.15
        bar_h = val/100 * 3.8
        by = 6.65 - bar_h
        rect(s, bx, by, 1.0, bar_h, fill=clr)
        txb(s, bx, 6.68, 1.0, 0.35, text=nm,
            size=9, color=C_WHITE, align=PP_ALIGN.CENTER)
        txb(s, bx, by-0.35, 1.0, 0.33,
            text=f"{val:.1f}%", size=11, bold=True, color=clr,
            align=PP_ALIGN.CENTER)
    # Delta labels
    txb(s, 4.15, 4.8, 4.7, 0.4,
        text="Δ CMF-Static vs alone: −31.24 pts  ← strongest feedback in entire benchmark",
        size=10, bold=True, color=C_WARN)
    # SCRUB timeline
    rect(s, 9.2, 1.6, 3.9, 5.25, fill=RGBColor(0xF5,0xF5,0xF5))
    txb(s, 9.3, 1.65, 3.7, 0.4, text="SCRUB Run Timeline", size=13, bold=True, color=C_GRAY_D)
    epochs = [
        ("Epoch 1", "Max-step: KL↑ on forget", C_WARN),
        ("Epoch 1", "Min-step: KL↓ on retain", C_ACCENT),
        ("Epoch 1", "recompute_cmf() →W updates", C_PURPLE),
        ("Epoch 2", "Max-step: KL↑ on forget", C_WARN),
        ("Epoch 2", "Min-step: KL↓ on retain", C_ACCENT),
        ("Epoch 2", "recompute_cmf() →W updates", C_PURPLE),
        ("Epoch 3", "Min-step ONLY (msteps=2)", C_ACCENT),
        ("Epoch 3", "recompute_cmf() → final W", C_PURPLE),
    ]
    for ei, (ep, desc, clr) in enumerate(epochs):
        ey = 2.15 + ei * 0.58
        rect(s, 9.3, ey, 0.75, 0.47, fill=clr)
        txb(s, 9.3, ey, 0.75, 0.47, text=ep,
            size=8, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        txb(s, 10.1, ey+0.05, 2.9, 0.38, text=desc, size=10, color=C_GRAY_D)
    slide_number(s, 6)

    # ── SLIDE 7: TARUN ───────────────────────────────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "TARUN (UNSIR) — Three Compounding Failure Modes",
                   "Adversarial noise + consistent direction + 6 epoch CMF loop = worst outcome", C_PURPLE)
    # Three columns for 3 failure modes
    modes = [
        ("① Adversarial Input", C_WARN,
         "Noise images optimised\nto maximise model\nconfidence on forget-class.\n\nNot real data — CMF\ncomputes class-mean of\n'what confuses the model\nmost', not real features.",
         "Noise ≠ Real features → class-mean drifts into meaningless region"),
        ("② Consistent Direction\n×6 Epochs", C_PURPLE,
         "3 Impair epochs (noise)\n+ 3 Repair epochs (retain)\n= 6 total CMF-reconstruction\ncycles vs SCRUB's 3.\n\nEvery cycle the drifted\nclass-mean is treated as\nground truth for the next.",
         "2× more feedback loop iterations than SCRUB"),
        ("③ Unrecoverable\nRepair Phase", RGBColor(0x83,0x39,0x8B),
         "Repair uses only retain\ndata — but class-mean for\nforget-class is NEVER\nupdated with real forget\ndata during repair.\n\nRetain accuracy is saved;\nforget representation stays\nirreversibly corrupted.",
         "Repair cannot fix what Impair destroyed in feature space"),
    ]
    for ci, (title, clr, body, footer) in enumerate(modes):
        cx = 0.3 + ci * 4.35
        cw = 4.2
        rect(s, cx, 1.55, cw, 0.45, fill=clr)
        txb(s, cx+0.1, 1.57, cw-0.2, 0.42, text=title,
            size=13, bold=True, color=C_WHITE)
        txb(s, cx+0.1, 2.05, cw-0.2, 3.1,
            text=body, size=12, color=C_GRAY_D)
        rect(s, cx, 5.2, cw, 0.6, fill=clr)
        txb(s, cx+0.1, 5.25, cw-0.2, 0.55, text=footer,
            size=10, bold=True, color=C_WHITE)
    # Bottom stat strip
    rect(s, 0.3, 5.9, 12.7, 1.45, fill=C_BG_DARK)
    stats = [
        ("No-CMF NCC Forget", "71.82%", C_GRAY),
        ("CMF-Static NCC Forget", "13.05%", C_WARN),
        ("CMF-Static Output Forget", "13.15%", C_WARN),
        ("Post-hoc Output Forget", "0.03%", C_ACCENT2),
        ("Damage (Δ Static−No CMF)", "−58.77 pts", C_PURPLE),
    ]
    for si, (lbl, val, clr) in enumerate(stats):
        sx = 0.6 + si * 2.5
        txb(s, sx, 5.95, 2.3, 0.45, text=lbl,
            size=9, color=C_GRAY, align=PP_ALIGN.CENTER)
        txb(s, sx, 6.4, 2.3, 0.7, text=val,
            size=22, bold=True, color=clr, align=PP_ALIGN.CENTER)
    slide_number(s, 7)

    # ── SLIDE 8: Performance Chart — NCC Forget ───────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "NCC Forget Accuracy — All Methods Compared",
                   "Lower = better unlearning from representation perspective (Oracle = 67.3%)", C_ACCENT)
    # Chart background
    chart_l, chart_t = 0.5, 1.65
    chart_w, chart_h = 12.3, 4.8
    rect(s, chart_l, chart_t, chart_w, chart_h, fill=RGBColor(0xFA,0xFA,0xFA),
         line_color=C_GRAY_L, line_width=1)
    # Oracle reference line
    oracle_y = chart_t + chart_h * (1 - 67.3/100)
    txb(s, 0.5, oracle_y - 0.28, 1.5, 0.28,
        text="Oracle 67.3%", size=10, bold=True, color=C_OK)
    # Draw oracle dashed line as thin rect
    rect(s, chart_l, oracle_y, chart_w, 0.03, fill=C_OK)

    methods_order = ["grad_ascent_descent","random_label","salun","scrub","tarun"]
    method_labels = ["NegGrad+","Random Label","SalUn","SCRUB","TARUN"]
    series = [
        ("No CMF",    [PERF[m]["no_cmf"]  for m in methods_order], C_GRAY),
        ("CMF-Static",[PERF[m]["static"]  for m in methods_order], C_ACCENT),
        ("Post-hoc",  [PERF[m]["posthoc"] for m in methods_order], C_ACCENT2),
    ]
    n_methods = 5
    group_w = chart_w / n_methods
    bar_w = group_w / 4
    for gi, mname in enumerate(method_labels):
        gx = chart_l + gi * group_w
        for si, (sname, vals, clr) in enumerate(series):
            val = vals[gi]
            bar_h_pt = val / 100 * chart_h
            by = chart_t + chart_h - bar_h_pt
            bx = gx + 0.1 + si * (bar_w + 0.04)
            rect(s, bx, by, bar_w, bar_h_pt, fill=clr)
            txb(s, bx, by - 0.33, bar_w, 0.3,
                text=f"{val:.0f}", size=9, bold=True, color=clr,
                align=PP_ALIGN.CENTER)
        # Method label
        txb(s, gx + 0.05, chart_t + chart_h + 0.05, group_w - 0.1, 0.35,
            text=mname, size=10, bold=True, color=C_GRAY_D,
            align=PP_ALIGN.CENTER)
    # Legend
    for li, (nm, _, clr) in enumerate(series):
        lx = 1.0 + li * 2.8
        rect(s, lx, 6.75, 0.25, 0.22, fill=clr)
        txb(s, lx+0.3, 6.73, 2.4, 0.26, text=nm, size=11, color=C_GRAY_D)
    # Y-axis labels
    for pct in [0, 25, 50, 75, 100]:
        y = chart_t + chart_h * (1 - pct/100)
        txb(s, 0.1, y - 0.15, 0.38, 0.3,
            text=f"{pct}%", size=9, color=C_GRAY, align=PP_ALIGN.RIGHT)
        rect(s, chart_l, y, chart_w, 0.01, fill=C_GRAY_L)
    slide_number(s, 8)

    # ── SLIDE 9: Output Forget Accuracy — Static vs Post-hoc ─────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "Output Forget Accuracy: Static vs Post-hoc",
                   "CIFAR-10 mean across 10 classes — lower is better unlearning", C_ACCENT2)
    chart_l, chart_t = 0.8, 1.7
    chart_w, chart_h = 11.7, 4.7
    rect(s, chart_l, chart_t, chart_w, chart_h,
         fill=RGBColor(0xFA,0xFA,0xFA), line_color=C_GRAY_L, line_width=1)
    methods_o = ["grad_ascent_descent","random_label","salun","scrub","tarun"]
    labels_o  = ["NegGrad+","Random Label","SalUn","SCRUB","TARUN"]
    static_v  = [OUT_FORGET["static"][m] for m in methods_o]
    posthoc_v = [OUT_FORGET["posthoc"][m] for m in methods_o]
    group_w = chart_w / 5
    bar_w = group_w * 0.35
    gap   = group_w * 0.06
    for gi, (lbl, sv, pv) in enumerate(zip(labels_o, static_v, posthoc_v)):
        gx = chart_l + gi * group_w + group_w * 0.05
        for si, (val, clr, nm) in enumerate([
                (sv, C_ACCENT, "Static"), (pv, C_ACCENT2, "Post-hoc")]):
            bar_h_pt = val / 100 * chart_h
            by = chart_t + chart_h - bar_h_pt
            bx = gx + si * (bar_w + gap)
            rect(s, bx, by, bar_w, bar_h_pt, fill=clr)
            txb(s, bx, by - 0.35, bar_w, 0.33,
                text=f"{val:.1f}%", size=9, bold=True, color=clr,
                align=PP_ALIGN.CENTER)
        # Delta label
        delta = pv - sv
        dclr = C_OK if delta < 0 else C_WARN
        txb(s, gx, chart_t + chart_h + 0.05, group_w * 0.8, 0.32,
            text=f"Δ={delta:+.1f}%", size=10, bold=True,
            color=dclr, align=PP_ALIGN.CENTER)
        txb(s, gx, chart_t + chart_h + 0.38, group_w * 0.8, 0.32,
            text=lbl, size=11, bold=True, color=C_GRAY_D,
            align=PP_ALIGN.CENTER)
    # Legend
    for li, (nm, clr) in enumerate([("CMF-Static",C_ACCENT),("Post-hoc k=5",C_ACCENT2)]):
        lx = 2.5 + li * 3.0
        rect(s, lx, 6.75, 0.25, 0.22, fill=clr)
        txb(s, lx+0.3, 6.73, 2.6, 0.26, text=nm, size=11, color=C_GRAY_D)
    for pct in [0, 25, 50, 75, 100]:
        y = chart_t + chart_h * (1 - pct/100)
        txb(s, 0.3, y - 0.15, 0.45, 0.3,
            text=f"{pct}%", size=9, color=C_GRAY, align=PP_ALIGN.RIGHT)
        rect(s, chart_l, y, chart_w, 0.01, fill=C_GRAY_L)
    slide_number(s, 9)

    # ── SLIDE 10: Compatibility Matrix Heatmap ───────────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "Compatibility Matrix: What Makes a Method CMF-Friendly?",
                   "Three algorithmic properties determine amplification risk", C_GRAY_D)
    # Property columns
    props = ["Fixed Target?", "Consistent\nDirection?", "Real Data?", "CMF\nCompatibility"]
    prop_xs = [3.3, 5.5, 7.7, 9.9]
    prop_ws = [1.9, 1.9, 1.9, 2.3]
    for i, (ph, px, pw) in enumerate(zip(props, prop_xs, prop_ws)):
        rect(s, px, 1.55, pw-0.1, 0.45, fill=C_BG_DARK)
        txb(s, px+0.05, 1.57, pw-0.2, 0.42, text=ph,
            size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    # Method rows
    matrix_data = [
        ("NegGrad+",    C_GOLD,   ["✗ No", "✓ Yes\n(saturates fast)", "✓ Real", "Neutral\n(problem pre-exists CMF)"]),
        ("Random Label",C_ACCENT, ["✗ No\n(random each ep)", "✗ No", "✓ Real", "Partial ✓\n(random walk limits amp.)"]),
        ("SalUn",       C_ACCENT2,["✗ No\n(random each ep)", "✗ No\n+50% mask", "✓ Real", "Partial ✓\n(weaker than rand.label)"]),
        ("SCRUB",       C_WARN,   ["✓ Yes\n(teacher frozen)", "✓✓ 2 epochs", "✓ Real", "Very Bad ✗\n(−31.24 pts NCC)"]),
        ("TARUN",       C_PURPLE, ["✓ Yes\n(max confidence)", "✓✓✓ 6 epochs", "✗ Noise!", "Worst ✗✗\n(−58.77 pts NCC)"]),
    ]
    cell_colors = {
        "✗ No": RGBColor(0xFC,0xF3,0xCF), "✓ Yes": RGBColor(0xD5,0xF5,0xE3),
        "✓ Real": RGBColor(0xD5,0xF5,0xE3), "✗ Noise!": C_WARN_SOFT,
    }
    compat_bg = {
        "Neutral": C_GOLD_SOFT, "Partial ✓": RGBColor(0xD6,0xEE,0xFB),
        "Very Bad ✗": C_WARN_SOFT, "Worst ✗✗": C_PURPLE_S,
    }
    compat_txt = {
        "Neutral": C_GOLD, "Partial ✓": C_ACCENT,
        "Very Bad ✗": C_WARN, "Worst ✗✗": C_PURPLE,
    }
    for ri, (mname, mclr, cells) in enumerate(matrix_data):
        ry = 2.1 + ri * 0.95
        rect(s, 0.3, ry, 2.9, 0.88, fill=mclr)
        txb(s, 0.4, ry+0.1, 2.7, 0.7, text=mname,
            size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        for ci, (cell_txt, px, pw) in enumerate(zip(cells, prop_xs, prop_ws)):
            bg = C_BG_LIGHT
            for key, clr in cell_colors.items():
                if key in cell_txt:
                    bg = clr; break
            if ci == 3:
                for key, clr in compat_bg.items():
                    if key in cell_txt:
                        bg = clr; break
            rect(s, px, ry, pw-0.1, 0.88, fill=bg)
            fc = C_GRAY_D
            if ci == 3:
                for key, clr in compat_txt.items():
                    if key in cell_txt:
                        fc = clr; break
            txb(s, px+0.05, ry+0.04, pw-0.2, 0.82,
                text=cell_txt, size=10, color=fc, bold=(ci==3))
    # Rule box
    rect(s, 0.3, 6.95, 12.7, 0.4, fill=C_BG_DARK)
    txb(s, 0.5, 6.97, 12.4, 0.36,
        text="Rule: CMF damage ∝ (has fixed target) × (direction consistent) × (multi-epoch) × (data quality)  "
             "— the better a method unlearns alone, the more CMF can amplify and destroy it",
        size=11, bold=True, color=C_ACCENT2)
    slide_number(s, 10)

    # ── SLIDE 11: Stage 1 → 2 → 3 Pipeline Diagram ───────────────────────────
    s = blank_slide(prs, C_BG_LIGHT)
    section_header(s, "CMF Pipeline: Static → Post-hoc → Gated Stage 3",
                   "Three-stage progression and where each stage targets", C_PURPLE)
    stages = [
        ("Stage 1\nCMF-Static",
         "Run base unlearning\n(encoder + W updated)\nrecompute_cmf() each epoch",
         ["Encoder: ✓ Updates", "W (CMF): ✓ Recomputed", "Duration: 3–4 epochs"],
         C_ACCENT),
        ("Stage 2\nCMF Post-hoc",
         "Freeze encoder.\nTrain only W via gradient\non retain distribution",
         ["Encoder: ✗ Frozen", "W (CMF): ✓ Trained", "Duration: 5 epochs"],
         C_ACCENT2),
        ("Stage 3\nGated Encoder",
         "Gate: skip if already\nin oracle zone. Per-epoch\nloop with revert on over-erase",
         ["Encoder: ✓ Conditional", "W (CMF): ✗ Frozen", "Gate: NCC-based stop"],
         C_PURPLE),
    ]
    for si, (title, desc, details, clr) in enumerate(stages):
        sx = 0.4 + si * 4.3
        rect(s, sx, 1.55, 3.9, 1.0, fill=clr)
        txb(s, sx+0.1, 1.58, 3.7, 0.95, text=title,
            size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        rect(s, sx, 2.6, 3.9, 1.45, fill=RGBColor(0xF5,0xF5,0xF5))
        txb(s, sx+0.1, 2.65, 3.7, 1.35, text=desc,
            size=12, color=C_GRAY_D)
        for di, dtxt in enumerate(details):
            dy = 4.15 + di * 0.45
            bg = clr if di == 0 else (C_OK_SOFT if "✓" in dtxt else C_WARN_SOFT)
            rect(s, sx, dy, 3.9, 0.4, fill=bg)
            txb(s, sx+0.1, dy+0.04, 3.7, 0.35,
                text=dtxt, size=11, color=C_WHITE if di==0 else C_GRAY_D)
        if si < 2:
            txb(s, sx+4.0, 2.8, 0.2, 0.4, text="→",
                size=30, bold=True, color=clr)
    # Key metrics comparison
    rect(s, 0.4, 5.6, 12.5, 1.7, fill=C_BG_DARK)
    txb(s, 0.6, 5.65, 12.2, 0.42,
        text="Output Forget Accuracy (mean, CIFAR-10, k=5, retain_only)",
        size=12, bold=True, color=C_WHITE)
    compare = [("TARUN","13.15%","0.03%"),("SCRUB","32.54%","4.65%"),
               ("SalUn","83.90%","66.08%"),("Rand.Label","87.27%","78.31%"),("NegGrad+","51.46%","45.96%")]
    for ci, (m, sv, pv) in enumerate(compare):
        cx = 0.7 + ci * 2.45
        txb(s, cx, 6.12, 2.2, 0.3, text=m, size=10, bold=True, color=C_GRAY,
            align=PP_ALIGN.CENTER)
        txb(s, cx, 6.42, 2.2, 0.35, text=f"Static: {sv}", size=10, color=C_WARN,
            align=PP_ALIGN.CENTER)
        txb(s, cx, 6.78, 2.2, 0.35, text=f"Post-hoc: {pv}", size=10,
            color=C_ACCENT2, align=PP_ALIGN.CENTER)
    slide_number(s, 11)

    # ── SLIDE 12: Summary & Recommendations ──────────────────────────────────
    s = blank_slide(prs, C_BG_DARK)
    section_header(s, "Summary & Key Takeaways",
                   "What the algorithm analysis tells us about CMF compatibility", C_ACCENT2)
    # 4 takeaway blocks
    takeaways = [
        (C_WARN,   "CMF Amplification Rule",
         "CMF silently amplifies any consistent ascent signal. "
         "Methods with clear target + steady direction (SCRUB, TARUN) "
         "suffer the most. Methods with noisy/random direction (random_label, SalUn) "
         "are partially shielded."),
        (C_GOLD,   "NegGrad+ is Different",
         "The only method where CMF is NOT the culprit. "
         "Unbounded −CE already over-erases before CMF acts. "
         "Post-hoc helps here by removing the rigid CMF classifier constraint."),
        (C_ACCENT2,"Post-hoc as the Fix",
         "Freezing encoder + training W via gradient (Post-hoc) is "
         "4–16× faster and strictly stronger than CMF-Static for all methods "
         "except NegGrad+ where both Stage 1 damage and Post-hoc benefit "
         "are moderate."),
        (C_OK,     "Gated Stage 3 Goal",
         "Per-epoch gate (skip if already in oracle zone) + revert on "
         "over-erase + retain-CE safety term — prevents Stage 3 from "
         "undoing the post-hoc gains and protects against the feedback loop."),
    ]
    for ti, (clr, title, body) in enumerate(takeaways):
        tx = 0.4 + (ti % 2) * 6.4
        ty = 1.65 + (ti // 2) * 2.5
        rect(s, tx, ty, 6.1, 2.3, fill=C_BG_MID)
        rect(s, tx, ty, 0.08, 2.3, fill=clr)
        txb(s, tx+0.25, ty+0.1, 5.7, 0.45,
            text=title, size=14, bold=True, color=clr)
        txb(s, tx+0.25, ty+0.58, 5.7, 1.65,
            text=body, size=11, color=C_GRAY)
    # Bottom bar
    rect(s, 0, 6.95, 13.33, 0.55, fill=C_ACCENT)
    txb(s, 0.5, 6.98, 12.3, 0.45,
        text="Full analysis: CMF_Method_Compatibility_Analysis.md  ·  "
             "Benchmark: REPORT_CMF_STATIC_VS_POSTHOC.md  ·  "
             "Paper: arXiv:2604.08271v1",
        size=11, color=C_WHITE, align=PP_ALIGN.CENTER)
    slide_number(s, 12)

    out = "CMF_Unlearning_Analysis_v2.pptx"
    prs.save(out)
    print(f"[OK] Saved: {out}  ({prs.slides.__len__()} slides)")

if __name__ == "__main__":
    build()
