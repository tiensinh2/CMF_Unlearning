"""
Build CMF_Unlearning_Analysis.pptx
Slides:
  0 - Title
  1 - Overview / Research Context
  2 - NegGrad+ (grad_ascent_descent)
  3 - random_label
  4 - SalUn
  5 - SCRUB
  6 - TARUN
  7 - Summary Comparison Table
  8 - Key Findings / Conclusion
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import pptx.oxml.ns as ns
from lxml import etree
import copy

# ── Palette ─────────────────────────────────────────────────────────────────
DARK_BG   = RGBColor(0x0A, 0x16, 0x28)   # deep navy
PANEL_BG  = RGBColor(0x10, 0x24, 0x40)   # slightly lighter navy
ACCENT1   = RGBColor(0x3B, 0x82, 0xD4)   # blue
ACCENT2   = RGBColor(0x7C, 0x5C, 0xD8)   # purple
GREEN_OK  = RGBColor(0x22, 0xC5, 0x5E)   # good compat
AMBER_MID = RGBColor(0xF5, 0x9E, 0x0B)   # partial compat
RED_BAD   = RGBColor(0xEF, 0x44, 0x44)   # bad compat
RED_WORST = RGBColor(0xDC, 0x26, 0x26)   # worst compat
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_TXT = RGBColor(0xC9, 0xD1, 0xD9)
MUTED     = RGBColor(0x8B, 0x94, 0xA3)

# ── Slide dimensions (16:9 widescreen) ──────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]          # blank
    return prs.slides.add_slide(layout)


def bg(slide, color: RGBColor):
    """Fill slide background with a solid color."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, x, y, w, h, fill_color=None, line_color=None, line_width=Pt(1)):
    """Add a plain rectangle shape."""
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        x, y, w, h
    )
    shape.line.width = line_width if line_color else Pt(0)
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    return shape


def txt(slide, text, x, y, w, h,
        size=Pt(14), bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    """Add a text box."""
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txBox


def txt_block(slide, lines, x, y, w, h,
              size=Pt(13), color=LIGHT_TXT, leading=Pt(6)):
    """Add multiple lines as a single text box with bullet-style paragraphs."""
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = leading
        run = p.add_run()
        run.text = line
        run.font.size = size
        run.font.color.rgb = color
    return txBox


def section_badge(slide, label, color, x, y):
    """Small colored badge label."""
    b = box(slide, x, y, Inches(1.9), Inches(0.32), fill_color=color)
    tf = b.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = tf.paragraphs[0].add_run()
    run.text = label
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = WHITE


def arrow_right(slide, x, y, length, color=ACCENT1):
    """Draw a simple right-pointing arrow as connector."""
    from pptx.util import Emu
    conn = slide.shapes.add_connector(
        1,  # straight
        x, y, x + length, y
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(2)
    return conn


def circle_num(slide, num, x, y, size=Inches(0.42), color=ACCENT1):
    """Numbered circle."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    shape = slide.shapes.add_shape(
        9,  # oval
        x, y, size, size
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    tf = shape.text_frame
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = tf.paragraphs[0].add_run()
    run.text = str(num)
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = WHITE
    return shape


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 0 — TITLE
# ════════════════════════════════════════════════════════════════════════════
def slide_title(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)

    # top accent bar
    box(s, 0, 0, W, Inches(0.07), fill_color=ACCENT1)

    # decorative vertical bar
    box(s, Inches(0.55), Inches(1.2), Inches(0.08), Inches(4.5), fill_color=ACCENT1)

    # main title
    txt(s, "CMF-Compatible Machine Unlearning",
        Inches(0.85), Inches(1.3), Inches(11.5), Inches(1.2),
        size=Pt(38), bold=True, color=WHITE)

    # sub-title
    txt(s, "Algorithm Mechanisms & CMF Compatibility Analysis",
        Inches(0.85), Inches(2.55), Inches(11.0), Inches(0.65),
        size=Pt(22), bold=False, color=ACCENT1)

    # description
    txt(s, "A deep-dive into 5 unlearning methods — NegGrad+, random_label,\n"
           "SalUn, SCRUB, and TARUN — examining how each interacts with the\n"
           "Class-Mean Feature (CMF) classifier reconstruction mechanism.",
        Inches(0.85), Inches(3.3), Inches(11.0), Inches(1.4),
        size=Pt(15), color=LIGHT_TXT)

    # bottom metadata bar
    box(s, 0, Inches(6.9), W, Inches(0.6), fill_color=PANEL_BG)
    txt(s, "Machine Unlearning  ·  CMF Compatibility  ·  5 Methods Analyzed",
        Inches(0.5), Inches(6.95), Inches(10), Inches(0.4),
        size=Pt(11), color=MUTED, align=PP_ALIGN.LEFT)

    # legend pills
    for label, color, xi in [
        ("Compatible", GREEN_OK, Inches(7.5)),
        ("Partial", AMBER_MID, Inches(9.1)),
        ("Incompatible", RED_BAD, Inches(10.7)),
    ]:
        section_badge(s, label, color, xi, Inches(7.1))


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — OVERVIEW: What is CMF?
# ════════════════════════════════════════════════════════════════════════════
def slide_overview(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=ACCENT1)

    txt(s, "Research Context & CMF Mechanism",
        Inches(0.4), Inches(0.15), Inches(10), Inches(0.6),
        size=Pt(26), bold=True, color=WHITE)

    # ── LEFT: CMF explanation ──
    box(s, Inches(0.3), Inches(0.9), Inches(5.9), Inches(5.9),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt(s, "What is CMF?", Inches(0.5), Inches(1.0), Inches(5.5), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT1)
    txt_block(s, [
        "Class-Mean Feature (CMF) replaces a learned linear",
        "classifier W with a non-parametric prototype:",
        "",
        "  W_c  =  mean { φ(x) : x ∈ retain, label = c }",
        "",
        "• Classifier is FROZEN during unlearning",
        "• Reconstructed after every epoch",
        "• Reflects representation geometry honestly",
        "• No gradient flows through W",
        "",
        "This creates an epoch-level feedback loop:",
        "  representation changes → W updated → next epoch",
        "  uses new W → further representation drift",
    ], Inches(0.5), Inches(1.55), Inches(5.5), Inches(4.9),
       size=Pt(12.5), color=LIGHT_TXT, leading=Pt(3))

    # ── RIGHT: 5 methods overview ──
    box(s, Inches(6.5), Inches(0.9), Inches(6.5), Inches(5.9),
        fill_color=PANEL_BG, line_color=ACCENT2, line_width=Pt(1))
    txt(s, "5 Methods vs CMF", Inches(6.7), Inches(1.0), Inches(6.1), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT2)

    methods = [
        ("1", "NegGrad+",     "Unbounded ascent — fast saturation",   AMBER_MID, "Neutral"),
        ("2", "random_label", "Random relabeling — inconsistent dir.", GREEN_OK,  "Partial ✓"),
        ("3", "SalUn",        "random_label + static saliency mask",   GREEN_OK,  "Partial ✓"),
        ("4", "SCRUB",        "KL-divergence to frozen teacher",       RED_BAD,   "Bad ✗"),
        ("5", "TARUN",        "Adversarial noise + impair/repair",     RED_WORST, "Worst ✗✗"),
    ]
    for i, (num, name, desc, color, compat) in enumerate(methods):
        yy = Inches(1.6) + i * Inches(0.98)
        circle_num(s, num, Inches(6.7), yy, size=Inches(0.38), color=color)
        txt(s, name, Inches(7.22), yy, Inches(2.3), Inches(0.38),
            size=Pt(13), bold=True, color=WHITE)
        txt(s, desc, Inches(7.22), yy + Inches(0.28), Inches(3.4), Inches(0.35),
            size=Pt(10.5), color=MUTED)
        # compat badge
        box(s, Inches(10.9), yy + Inches(0.04), Inches(1.8), Inches(0.3),
            fill_color=color)
        b = s.shapes[-1]
        tf = b.text_frame
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run()
        r.text = compat
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = WHITE

    # bottom note
    txt(s, "Key insight: CMF amplifies any consistent unidirectional signal in the representation space.",
        Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.38),
        size=Pt(11), color=MUTED, italic=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — NegGrad+ (grad_ascent_descent)
# ════════════════════════════════════════════════════════════════════════════
def slide_neggrad(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=AMBER_MID)

    txt(s, "① NegGrad+  (grad_ascent_descent)",
        Inches(0.4), Inches(0.12), Inches(9), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)
    section_badge(s, "NEUTRAL / Pre-existing issue", AMBER_MID,
                  Inches(9.5), Inches(0.18))

    # ── Algorithm box (left) ──
    box(s, Inches(0.3), Inches(0.82), Inches(5.8), Inches(5.5),
        fill_color=PANEL_BG, line_color=AMBER_MID, line_width=Pt(1))
    txt(s, "Algorithm", Inches(0.5), Inches(0.9), Inches(5.4), Inches(0.38),
        size=Pt(15), bold=True, color=AMBER_MID)

    code_lines = [
        "W_fixed = CMFweights.detach()   # frozen",
        "",
        "# (a) Forget — MAXIMIZE loss (negative sign)",
        "loss_ascent = −0.5 × CE(z_f @ W_fixed.T, y_f)",
        "loss_ascent.backward()",
        "",
        "# (b) Retain — minimize loss",
        "loss_descent = CE(z_r @ W_fixed.T, y_r)",
        "loss_descent.backward()",
        "",
        "optimizer.step()  # cumulative gradients",
        "",
        "⚠ No lower bound on −CE  →  unbounded ascent",
        "⚠ Runs for 3 epochs only (no early stop)",
    ]
    txt_block(s, code_lines, Inches(0.5), Inches(1.35), Inches(5.4), Inches(4.7),
              size=Pt(11), color=LIGHT_TXT, leading=Pt(2))

    # ── Mechanism diagram (right top) ──
    box(s, Inches(6.3), Inches(0.82), Inches(6.7), Inches(2.5),
        fill_color=PANEL_BG, line_color=AMBER_MID, line_width=Pt(1))
    txt(s, "What happens to NCC_forget", Inches(6.5), Inches(0.9),
        Inches(6.3), Inches(0.38), size=Pt(14), bold=True, color=AMBER_MID)

    # score bar chart (visual)
    bars = [
        ("Model original",  94.0, MUTED),
        ("Oracle (ideal)",  67.3, GREEN_OK),
        ("No-CMF (alone)",  55.65, RED_BAD),
        ("CMF-Static",      57.33, AMBER_MID),
        ("Post-hoc",        76.23, GREEN_OK),
    ]
    bar_x0 = Inches(6.5)
    bar_y0 = Inches(1.42)
    bar_h  = Inches(0.22)
    bar_gap = Inches(0.28)
    max_val = 100.0
    bar_max_w = Inches(4.5)
    for i, (label, val, color) in enumerate(bars):
        yy = bar_y0 + i * (bar_h + bar_gap)
        # label
        txt(s, label, bar_x0, yy, Inches(1.85), bar_h,
            size=Pt(9.5), color=LIGHT_TXT)
        # bar
        w_bar = bar_max_w * (val / max_val)
        box(s, bar_x0 + Inches(1.9), yy, w_bar, bar_h, fill_color=color)
        # value
        txt(s, f"{val:.1f}%", bar_x0 + Inches(1.9) + w_bar + Inches(0.07),
            yy, Inches(0.7), bar_h, size=Pt(9.5), color=WHITE)

    # ── Key findings (right bottom) ──
    box(s, Inches(6.3), Inches(3.45), Inches(6.7), Inches(2.87),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt(s, "Key Findings", Inches(6.5), Inches(3.55), Inches(6.3), Inches(0.38),
        size=Pt(14), bold=True, color=ACCENT1)
    txt_block(s, [
        "▸ Problem exists WITHOUT CMF (illusion_gap = −11.65)",
        "▸ Unbounded −CE has NO saturation — ascent overshoots",
        "  Oracle (67.3%) in just 3 epochs",
        "▸ CMF arrives too late: representation already beyond",
        "  the target threshold when W is first reconstructed",
        "▸ Post-hoc succeeds here: representation is actually",
        "  good; only the rigid CMF formula was the obstacle",
        "▸ CMF is NOT the primary cause — NegGrad+ design flaw",
        "  (missing early-stop / ascent ceiling) is root cause",
    ], Inches(6.5), Inches(4.0), Inches(6.3), Inches(2.2),
       size=Pt(11.5), color=LIGHT_TXT, leading=Pt(3))


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — random_label
# ════════════════════════════════════════════════════════════════════════════
def slide_random_label(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=GREEN_OK)

    txt(s, "② random_label",
        Inches(0.4), Inches(0.12), Inches(9), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)
    section_badge(s, "PARTIAL COMPATIBILITY", GREEN_OK, Inches(9.5), Inches(0.18))

    # ── Algorithm (left) ──
    box(s, Inches(0.3), Inches(0.82), Inches(5.8), Inches(5.5),
        fill_color=PANEL_BG, line_color=GREEN_OK, line_width=Pt(1))
    txt(s, "Algorithm", Inches(0.5), Inches(0.9), Inches(5.4), Inches(0.38),
        size=Pt(15), bold=True, color=GREEN_OK)
    code_lines = [
        "# Mixed loader: forget (fake labels) + retain (true)",
        "for epoch in 1..4:",
        "  for (x, y_true) in mixed_loader:",
        "    y_train = y_true.clone()",
        "    # Replace forget labels with RANDOM wrong class",
        "    y_train[forget_mask] = random_choice(",
        "        non_forget_classes)   # re-sampled every epoch",
        "",
        "    W_fixed = CMFweights.detach()   # frozen",
        "    logits  = φ(x) @ W_fixed.T",
        "    loss    = CE(logits, y_train)   # normal DESCENT",
        "    loss.backward(); optimizer.step()",
        "",
        "★ NOT gradient ascent — it is descent on wrong labels",
        "★ Target class changes every epoch (random walk)",
    ]
    txt_block(s, code_lines, Inches(0.5), Inches(1.35), Inches(5.4), Inches(4.7),
              size=Pt(11), color=LIGHT_TXT, leading=Pt(2))

    # ── Random walk diagram (right top) ──
    box(s, Inches(6.3), Inches(0.82), Inches(6.7), Inches(2.5),
        fill_color=PANEL_BG, line_color=GREEN_OK, line_width=Pt(1))
    txt(s, "Feature-space behaviour across epochs",
        Inches(6.5), Inches(0.9), Inches(6.3), Inches(0.38),
        size=Pt(14), bold=True, color=GREEN_OK)

    # draw zigzag "random walk" visual using text art
    rw_lines = [
        "Epoch 1: forget-feature pulled → class B",
        "Epoch 2: forget-feature pulled → class D  (different!)",
        "Epoch 3: forget-feature pulled → class A  (different!)",
        "Epoch 4: forget-feature pulled → class F  (different!)",
        "",
        "Result: 'random walk' — no consistent direction",
        "CMF can amplify only average net displacement",
        "   → partial effect, not full unlearning",
    ]
    txt_block(s, rw_lines, Inches(6.5), Inches(1.35), Inches(6.3), Inches(1.9),
              size=Pt(11.5), color=LIGHT_TXT, leading=Pt(3))

    # ── Score bars ──
    bars = [
        ("Oracle (ideal)",  67.3, GREEN_OK),
        ("No-CMF output",   0.0,  RED_BAD),
        ("No-CMF NCC",      97.28, MUTED),
        ("CMF-Static",      87.62, AMBER_MID),
    ]
    bar_x0 = Inches(6.5)
    bar_y0 = Inches(3.46)
    bar_h  = Inches(0.22)
    bar_gap = Inches(0.28)
    bar_max_w = Inches(4.5)
    box(s, Inches(6.3), Inches(3.45), Inches(6.7), Inches(1.75),
        fill_color=PANEL_BG, line_color=GREEN_OK, line_width=Pt(1))
    txt(s, "NCC_forget scores", Inches(6.5), Inches(3.52), Inches(6.3), Inches(0.35),
        size=Pt(13), bold=True, color=GREEN_OK)
    for i, (label, val, color) in enumerate(bars):
        yy = bar_y0 + Inches(0.42) + i * (bar_h + bar_gap)
        txt(s, label, bar_x0, yy, Inches(1.85), bar_h, size=Pt(9.5), color=LIGHT_TXT)
        if val > 0:
            w_bar = bar_max_w * (val / 100.0)
            box(s, bar_x0 + Inches(1.9), yy, w_bar, bar_h, fill_color=color)
        txt(s, f"{val:.1f}%", bar_x0 + Inches(1.9) + bar_max_w * (val/100.0) + Inches(0.07),
            yy, Inches(0.7), bar_h, size=Pt(9.5), color=WHITE)

    # ── Key findings ──
    box(s, Inches(6.3), Inches(5.3), Inches(6.7), Inches(1.05),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt_block(s, [
        "▸ Illusion exposed: No-CMF Output=0% even though NCC=97.28%",
        "▸ CMF corrects this illusion, pulling NCC_forget from 97% → 87.62%",
        "▸ Inconsistent target direction limits how far CMF can push",
        "▸ Retain quality well-preserved (94.55%) — no deliberate retain damage",
    ], Inches(6.5), Inches(5.38), Inches(6.3), Inches(0.9),
       size=Pt(11), color=LIGHT_TXT, leading=Pt(3))


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — SalUn
# ════════════════════════════════════════════════════════════════════════════
def slide_salun(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=GREEN_OK)

    txt(s, "③ SalUn  (Saliency-guided Unlearning)",
        Inches(0.4), Inches(0.12), Inches(9), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)
    section_badge(s, "PARTIAL (weaker than random_label)", GREEN_OK, Inches(9.0), Inches(0.18))

    # ── Two-phase diagram (top) ──
    box(s, Inches(0.3), Inches(0.82), Inches(12.7), Inches(1.55),
        fill_color=PANEL_BG, line_color=GREEN_OK, line_width=Pt(1))
    txt(s, "Two-phase process", Inches(0.5), Inches(0.9), Inches(4), Inches(0.38),
        size=Pt(14), bold=True, color=GREEN_OK)

    # Phase 1 box
    box(s, Inches(0.5), Inches(1.3), Inches(5.5), Inches(0.9),
        fill_color=RGBColor(0x1a,0x3a,0x5c), line_color=ACCENT1, line_width=Pt(1))
    txt(s, "Phase 1 — Build saliency mask  (run ONCE on θ₀)",
        Inches(0.65), Inches(1.36), Inches(5.2), Inches(0.35),
        size=Pt(11.5), bold=True, color=ACCENT1)
    txt(s, "Accumulate |∇L_CE(forget, true_labels)| → keep top 50% params → binary mask",
        Inches(0.65), Inches(1.65), Inches(5.2), Inches(0.35),
        size=Pt(10.5), color=LIGHT_TXT)

    # arrow
    txt(s, "→", Inches(6.15), Inches(1.5), Inches(0.5), Inches(0.4),
        size=Pt(22), bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    # Phase 2 box
    box(s, Inches(6.7), Inches(1.3), Inches(6.1), Inches(0.9),
        fill_color=RGBColor(0x1a,0x3a,0x5c), line_color=GREEN_OK, line_width=Pt(1))
    txt(s, "Phase 2 — Train like random_label  (4 epochs, mask applied)",
        Inches(6.85), Inches(1.36), Inches(5.8), Inches(0.35),
        size=Pt(11.5), bold=True, color=GREEN_OK)
    txt(s, "p.grad *= hard_mask[p]  for each param  → 50% params frozen",
        Inches(6.85), Inches(1.65), Inches(5.8), Inches(0.35),
        size=Pt(10.5), color=LIGHT_TXT)

    # ── Left: dual limitations ──
    box(s, Inches(0.3), Inches(2.5), Inches(5.8), Inches(4.0),
        fill_color=PANEL_BG, line_color=RED_BAD, line_width=Pt(1))
    txt(s, "Two compounding limitations", Inches(0.5), Inches(2.6), Inches(5.4), Inches(0.38),
        size=Pt(14), bold=True, color=RED_BAD)
    txt_block(s, [
        "① Reduced amplitude of representation shift",
        "   Only 50% of encoder params can respond to",
        "   the relabeling signal — smaller net movement",
        "   compared to unconstrained random_label.",
        "",
        "② Mask is FROZEN at θ₀ and NEVER updated",
        "   Top-50% params by |∇| on original model",
        "   may not match the 'important' params after",
        "   a few epochs of unlearning. Mask continues",
        "   to point to the wrong locations throughout.",
        "",
        "   Combined: less freedom + wrong address",
        "   → representation drifts LESS than random_label",
        "   → CMF effect weaker: 84.46% vs 87.62%",
    ], Inches(0.5), Inches(3.05), Inches(5.5), Inches(3.2),
       size=Pt(11.5), color=LIGHT_TXT, leading=Pt(3))

    # ── Right: score comparison + key note ──
    box(s, Inches(6.3), Inches(2.5), Inches(6.7), Inches(4.0),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt(s, "Scores vs random_label", Inches(6.5), Inches(2.6),
        Inches(6.3), Inches(0.38), size=Pt(14), bold=True, color=ACCENT1)

    bars = [
        ("Oracle", 67.3, GREEN_OK),
        ("random_label (No-CMF NCC)", 97.28, MUTED),
        ("random_label CMF-Static",  87.62, GREEN_OK),
        ("SalUn (No-CMF NCC)",       97.11, MUTED),
        ("SalUn CMF-Static",         84.46, AMBER_MID),
    ]
    bar_x0 = Inches(6.5)
    bar_y0 = Inches(3.05)
    bar_h  = Inches(0.22)
    bar_gap = Inches(0.26)
    bar_max_w = Inches(3.8)
    for i, (label, val, color) in enumerate(bars):
        yy = bar_y0 + i * (bar_h + bar_gap)
        txt(s, label, bar_x0, yy, Inches(2.65), bar_h, size=Pt(9.5), color=LIGHT_TXT)
        w_bar = bar_max_w * (val / 100.0)
        box(s, bar_x0 + Inches(2.7), yy, w_bar, bar_h, fill_color=color)
        txt(s, f"{val:.1f}%", bar_x0 + Inches(2.7) + w_bar + Inches(0.05),
            yy, Inches(0.6), bar_h, size=Pt(9.5), color=WHITE)

    txt(s, "SalUn ≈ random_label in core mechanism — mask adds cost without benefit under CMF.",
        Inches(6.5), Inches(6.1), Inches(6.3), Inches(0.4),
        size=Pt(11), color=MUTED, italic=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — SCRUB
# ════════════════════════════════════════════════════════════════════════════
def slide_scrub(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=RED_BAD)

    txt(s, "④ SCRUB  (Teacher-Student KL Distillation)",
        Inches(0.4), Inches(0.12), Inches(9), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)
    section_badge(s, "HIGHLY INCOMPATIBLE ✗", RED_BAD, Inches(9.3), Inches(0.18))

    # ── Algorithm box (left) ──
    box(s, Inches(0.3), Inches(0.82), Inches(5.8), Inches(5.5),
        fill_color=PANEL_BG, line_color=RED_BAD, line_width=Pt(1))
    txt(s, "Algorithm", Inches(0.5), Inches(0.9), Inches(5.4), Inches(0.38),
        size=Pt(15), bold=True, color=RED_BAD)
    code_lines = [
        "teacher = deepcopy(Θ₀);  teacher.freeze()",
        "student = deepcopy(Θ₀)   # to be updated",
        "",
        "for epoch in 1..3:                    # 3 epochs total",
        "  if epoch ≤ 2:   # MAX-STEP (2 epochs)",
        "    # Maximize KL: push student AWAY from teacher",
        "    loss = −KL( student(x_f)/T ‖ teacher(x_f)/T )",
        "    loss.backward();  optimizer.step()",
        "",
        "  # MIN-STEP (every epoch)",
        "  # Minimize KL: keep student CLOSE to teacher on retain",
        "  loss = KL(retain) + α·CE(retain)",
        "  loss.backward();  optimizer.step()",
        "",
        "  student.recompute_cmf()   # CMF after each epoch",
        "",
        "★ KL-divergence has CLEAR fixed target (teacher)",
        "★ CONSISTENT direction across all 2 max-steps",
    ]
    txt_block(s, code_lines, Inches(0.5), Inches(1.35), Inches(5.4), Inches(4.7),
              size=Pt(10.5), color=LIGHT_TXT, leading=Pt(2))

    # ── Feedback loop diagram (right top) ──
    box(s, Inches(6.3), Inches(0.82), Inches(6.7), Inches(3.2),
        fill_color=PANEL_BG, line_color=RED_BAD, line_width=Pt(1))
    txt(s, "CMF Positive Feedback Loop", Inches(6.5), Inches(0.9),
        Inches(6.3), Inches(0.38), size=Pt(14), bold=True, color=RED_BAD)

    # Draw loop diagram with text boxes + arrows
    steps = [
        (Inches(6.6),  Inches(1.38), "MAX-STEP:\npush student away\nfrom teacher (KL↑)",      RGBColor(0x7f,0x1d,0x1d)),
        (Inches(9.8),  Inches(1.38), "Feature shifts\nconsistently in\none direction",          RGBColor(0x7f,0x1d,0x1d)),
        (Inches(9.8),  Inches(2.6),  "CMF recomputes\nclass-mean using\nshifted features",     RGBColor(0x78,0x35,0x0a)),
        (Inches(6.6),  Inches(2.6),  "Next epoch:\nKL measured from\nnew shifted W",           RGBColor(0x78,0x35,0x0a)),
    ]
    for x, y, text, fc in steps:
        b = box(s, x, y, Inches(2.8), Inches(0.75), fill_color=fc,
                line_color=RED_BAD, line_width=Pt(1))
        tf = b.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.alignment = PP_ALIGN.CENTER
        run = p0.add_run()
        run.text = text
        run.font.size = Pt(10)
        run.font.color.rgb = WHITE

    # arrows connecting boxes
    # → right
    txt(s, "→", Inches(9.5), Inches(1.6), Inches(0.4), Inches(0.3),
        size=Pt(18), bold=True, color=RED_BAD, align=PP_ALIGN.CENTER)
    # ↓ right
    txt(s, "↓", Inches(11.15), Inches(2.2), Inches(0.4), Inches(0.4),
        size=Pt(18), bold=True, color=RED_BAD, align=PP_ALIGN.CENTER)
    # ← left
    txt(s, "←", Inches(9.5), Inches(2.8), Inches(0.4), Inches(0.3),
        size=Pt(18), bold=True, color=RED_BAD, align=PP_ALIGN.CENTER)
    # ↑ left
    txt(s, "↑", Inches(7.95), Inches(2.05), Inches(0.4), Inches(0.55),
        size=Pt(18), bold=True, color=RED_BAD, align=PP_ALIGN.CENTER)

    txt(s, "⚡ RUNAWAY loop — 2 consistent max-steps are enough",
        Inches(6.5), Inches(3.45), Inches(6.3), Inches(0.32),
        size=Pt(11), bold=True, color=RED_BAD)

    # ── Score bars (right bottom) ──
    box(s, Inches(6.3), Inches(4.1), Inches(6.7), Inches(2.22),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt(s, "NCC_forget: SCRUB alone vs with CMF",
        Inches(6.5), Inches(4.18), Inches(6.3), Inches(0.38),
        size=Pt(13), bold=True, color=ACCENT1)

    bars = [
        ("Oracle (ideal)",      67.3, GREEN_OK),
        ("SCRUB alone (NCC)",   65.64, GREEN_OK),
        ("CMF-Static",          34.40, RED_BAD),
        ("Post-hoc (NCC)",      25.31, RED_WORST),
    ]
    bar_x0 = Inches(6.5)
    bar_y0 = Inches(4.65)
    bar_h  = Inches(0.22)
    bar_gap = Inches(0.26)
    bar_max_w = Inches(4.0)
    for i, (label, val, color) in enumerate(bars):
        yy = bar_y0 + i * (bar_h + bar_gap)
        txt(s, label, bar_x0, yy, Inches(2.1), bar_h, size=Pt(9.5), color=LIGHT_TXT)
        w_bar = bar_max_w * (val / 100.0)
        box(s, bar_x0 + Inches(2.15), yy, w_bar, bar_h, fill_color=color)
        txt(s, f"{val:.1f}%  (Δ{val-67.3:+.1f})",
            bar_x0 + Inches(2.15) + w_bar + Inches(0.05),
            yy, Inches(1.2), bar_h, size=Pt(9.5), color=WHITE)

    txt(s, "Paradox: SCRUB's high signal quality is exactly what makes it CMF's worst victim.",
        Inches(6.5), Inches(6.38), Inches(6.3), Inches(0.38),
        size=Pt(11), color=MUTED, italic=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — TARUN
# ════════════════════════════════════════════════════════════════════════════
def slide_tarun(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=RED_WORST)

    txt(s, "⑤ TARUN  (UNSIR — Adversarial Noise Unlearning)",
        Inches(0.4), Inches(0.12), Inches(9), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)
    section_badge(s, "WORST COMPATIBILITY ✗✗", RED_WORST, Inches(9.3), Inches(0.18))

    # ── 3-phase pipeline (top) ──
    box(s, Inches(0.3), Inches(0.82), Inches(12.7), Inches(1.65),
        fill_color=PANEL_BG, line_color=RED_WORST, line_width=Pt(1))
    txt(s, "3-phase pipeline", Inches(0.5), Inches(0.9), Inches(4), Inches(0.38),
        size=Pt(14), bold=True, color=RED_WORST)

    phases = [
        ("Noise Gen",  "Optimise synthetic\nnoise to maximise\nforget-class confidence",   RGBColor(0x4a,0x04,0x04), ACCENT2),
        ("Impair",     "Train 3 epochs on\n(noise, forget_label)\n+ retain_data",          RGBColor(0x5c,0x07,0x07), RED_BAD),
        ("Repair",     "Train 3 epochs on\nretain_data only\n(CMF recomputed each ep.)",   RGBColor(0x5c,0x1a,0x00), AMBER_MID),
    ]
    xpos = [Inches(0.5), Inches(4.7), Inches(8.9)]
    for i, (title, desc, fc, tc) in enumerate(phases):
        b = box(s, xpos[i], Inches(1.3), Inches(3.9), Inches(1.0),
                fill_color=fc, line_color=tc, line_width=Pt(1.5))
        tf = b.text_frame; tf.word_wrap = True
        p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER
        r = p0.add_run(); r.text = title
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = tc
        p1 = tf.add_paragraph(); p1.alignment = PP_ALIGN.CENTER
        r2 = p1.add_run(); r2.text = desc
        r2.font.size = Pt(10); r2.font.color.rgb = LIGHT_TXT
        if i < 2:
            txt(s, "→", xpos[i] + Inches(4.0), Inches(1.6), Inches(0.6), Inches(0.5),
                size=Pt(22), bold=True, color=RED_WORST, align=PP_ALIGN.CENTER)

    txt(s, "Total: 3+3=6 encoder update epochs (2× the nominal 3-epoch setting)",
        Inches(0.5), Inches(2.4), Inches(12.3), Inches(0.3),
        size=Pt(10.5), color=MUTED, italic=True)

    # ── Left: 3 compounding factors ──
    box(s, Inches(0.3), Inches(2.8), Inches(5.8), Inches(3.7),
        fill_color=PANEL_BG, line_color=RED_WORST, line_width=Pt(1))
    txt(s, "3 compounding incompatibility factors",
        Inches(0.5), Inches(2.9), Inches(5.4), Inches(0.38),
        size=Pt(14), bold=True, color=RED_WORST)

    factors = [
        ("①", ACCENT2,    "Adversarially optimal signal",
         "Noise designed to maximally stimulate forget-class activations\n"
         "→ strongest possible ascent signal (stronger than SCRUB)"),
        ("②", RED_BAD,    "Consistent direction × 3 epochs (Impair)",
         "6 total epochs of same-direction drift gives CMF feedback\n"
         "loop maximum time to compound — double the iterations"),
        ("③", RED_WORST,  "Semantically invalid class-mean",
         "CMF recomputes forget-class mean from noise images, not real\n"
         "data → mean anchors to an artificial, unstable point in space"),
    ]
    for i, (num, color, title, desc) in enumerate(factors):
        yy = Inches(3.38) + i * Inches(1.0)
        txt(s, num, Inches(0.5), yy, Inches(0.35), Inches(0.35),
            size=Pt(14), bold=True, color=color)
        txt(s, title, Inches(0.9), yy, Inches(5.0), Inches(0.32),
            size=Pt(12), bold=True, color=color)
        txt(s, desc, Inches(0.9), yy + Inches(0.3), Inches(5.0), Inches(0.55),
            size=Pt(10.5), color=LIGHT_TXT)

    # ── Right: scores + repair note ──
    box(s, Inches(6.3), Inches(2.8), Inches(6.7), Inches(3.7),
        fill_color=PANEL_BG, line_color=ACCENT1, line_width=Pt(1))
    txt(s, "NCC_forget scores", Inches(6.5), Inches(2.9),
        Inches(6.3), Inches(0.38), size=Pt(14), bold=True, color=ACCENT1)

    bars = [
        ("Oracle",                  67.3,  GREEN_OK),
        ("TARUN alone (NCC)",       71.82, GREEN_OK),
        ("CMF-Static (NCC)",        13.05, RED_WORST),
        ("Retain (No-CMF)",         92.42, MUTED),
        ("Retain (CMF-Static)",     92.64, MUTED),
    ]
    bar_x0 = Inches(6.5)
    bar_y0 = Inches(3.4)
    bar_h  = Inches(0.22)
    bar_gap = Inches(0.26)
    bar_max_w = Inches(3.8)
    for i, (label, val, color) in enumerate(bars):
        yy = bar_y0 + i * (bar_h + bar_gap)
        txt(s, label, bar_x0, yy, Inches(2.35), bar_h, size=Pt(9.5), color=LIGHT_TXT)
        w_bar = bar_max_w * (val / 100.0)
        box(s, bar_x0 + Inches(2.4), yy, w_bar, bar_h, fill_color=color)
        txt(s, f"{val:.1f}%", bar_x0 + Inches(2.4) + w_bar + Inches(0.07),
            yy, Inches(0.8), bar_h, size=Pt(9.5), color=WHITE)

    txt(s, "Repair phase preserves retain (92.64%) but cannot restore\n"
           "a forget-class mean that has drifted to a semantically invalid location.",
        Inches(6.5), Inches(5.65), Inches(6.3), Inches(0.7),
        size=Pt(11), color=MUTED, italic=True)

    txt(s, f"▶ Worst collapse in the benchmark: 71.82 → 13.05  (Δ = −58.77 pts)",
        Inches(6.5), Inches(6.4), Inches(6.3), Inches(0.38),
        size=Pt(12), bold=True, color=RED_WORST)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Summary Comparison Table
# ════════════════════════════════════════════════════════════════════════════
def slide_summary_table(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=ACCENT1)

    txt(s, "Summary: Algorithm Properties vs CMF Compatibility",
        Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.55),
        size=Pt(24), bold=True, color=WHITE)

    # Table via manual shapes
    headers = ["Method", "Ascent Source", "Fixed Target?",
               "Consistent Dir.?", "Real Data?", "CMF Compat.", "Score Δ"]
    col_widths = [Inches(1.6), Inches(2.2), Inches(1.4),
                  Inches(1.6), Inches(1.25), Inches(1.6), Inches(1.5)]
    col_starts = [Inches(0.25)]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    header_y = Inches(0.85)
    header_h = Inches(0.42)

    # header row
    for i, (hdr, w, x) in enumerate(zip(headers, col_widths, col_starts)):
        box(s, x, header_y, w, header_h, fill_color=ACCENT1)
        txt(s, hdr, x + Inches(0.05), header_y + Inches(0.04), w - Inches(0.1), header_h,
            size=Pt(10.5), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    rows = [
        # method, ascent_src, fixed, consistent, real_data, compat, delta
        ("NegGrad+",     "−CE (true labels)",          "No (unbounded)",  "Yes, too fast",      "Yes",   AMBER_MID, "Neutral",      "−11.65"),
        ("random_label", "CE on random fake labels",   "No (changes/ep)", "No (random walk)",   "Yes",   GREEN_OK,  "Partial ✓",    "+30.0"),
        ("SalUn",        "random_label + mask",        "No",              "No, + mask limit",   "Yes",   GREEN_OK,  "Partial ✓",    "+26.6"),
        ("SCRUB",        "KL → frozen teacher",        "Yes (teacher)",   "Yes (2 max-steps)",  "Yes",   RED_BAD,   "Very Bad ✗",   "−31.24"),
        ("TARUN",        "CE on adversarial noise",    "Yes (max-conf)",  "Yes (3 imp. ep.)",   "No",    RED_WORST, "Worst ✗✗",     "−58.77"),
    ]
    row_h   = Inches(0.94)
    row_colors = [
        RGBColor(0x18, 0x28, 0x3c),
        RGBColor(0x13, 0x23, 0x35),
    ]

    for ri, (method, ascent, fixed, consist, real, compat_color, compat_txt, delta) in enumerate(rows):
        yy = header_y + header_h + ri * row_h
        rc = row_colors[ri % 2]
        # background for whole row
        box(s, Inches(0.25), yy, sum(col_widths), row_h, fill_color=rc)

        cells = [method, ascent, fixed, consist, real, compat_txt, delta]
        for ci, (cell, cw, cx) in enumerate(zip(cells, col_widths, col_starts)):
            if ci == 5:  # compat column — colored
                box(s, cx + Inches(0.06), yy + Inches(0.27),
                    cw - Inches(0.12), Inches(0.38), fill_color=compat_color)
                txt(s, cell, cx + Inches(0.06), yy + Inches(0.28),
                    cw - Inches(0.12), Inches(0.36),
                    size=Pt(10), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
            elif ci == 6:  # delta — color by sign
                delta_val = float(delta.replace("−", "-").replace("+", ""))
                dc = GREEN_OK if delta_val > 0 else (RED_WORST if delta_val < -40 else RED_BAD)
                txt(s, cell, cx + Inches(0.05), yy + Inches(0.28),
                    cw - Inches(0.1), Inches(0.36),
                    size=Pt(11), bold=True, color=dc, align=PP_ALIGN.CENTER)
            else:
                txt(s, cell, cx + Inches(0.05), yy + Inches(0.22),
                    cw - Inches(0.1), Inches(0.5),
                    size=Pt(10), color=LIGHT_TXT, align=PP_ALIGN.LEFT)

        # row separator line
        box(s, Inches(0.25), yy + row_h - Inches(0.01),
            sum(col_widths), Inches(0.01),
            fill_color=RGBColor(0x30,0x48,0x60))

    # Legend bar at bottom
    box(s, Inches(0.25), Inches(7.0), Inches(12.85), Inches(0.38),
        fill_color=PANEL_BG)
    txt(s, "Score Δ = NCC_forget(No-CMF) − NCC_forget(CMF-Static)   "
           "Negative = CMF made unlearning worse   "
           "Positive = CMF improved unlearning",
        Inches(0.4), Inches(7.04), Inches(12.5), Inches(0.3),
        size=Pt(10), color=MUTED, italic=True)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Key Findings / Conclusion
# ════════════════════════════════════════════════════════════════════════════
def slide_conclusion(prs):
    s = blank_slide(prs)
    bg(s, DARK_BG)
    box(s, 0, 0, W, Inches(0.07), fill_color=ACCENT2)

    txt(s, "Key Findings & Conclusions",
        Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.55),
        size=Pt(26), bold=True, color=WHITE)

    # ── 3-column layout ──
    col_data = [
        # title, color, lines
        ("The CMF Amplification Law", ACCENT1, [
            "CMF does not judge whether",
            "unlearning is 'correct'.",
            "It amplifies ANY consistent,",
            "directional representation shift.",
            "",
            "High-quality signals (SCRUB,",
            "TARUN) → strongest amplification",
            "→ worst collapse.",
            "",
            "Low-quality signals (random_label)",
            "→ weak amplification → partial",
            "improvement only.",
        ]),
        ("Representation vs Classifier", AMBER_MID, [
            "Two separate failure modes:",
            "",
            "① Classifier too rigid:",
            "  NegGrad+ — representation fine,",
            "  CMF formula prevents adaptation.",
            "  → Post-hoc fixes this easily.",
            "",
            "② Representation truly damaged:",
            "  SCRUB/TARUN — features drift too",
            "  far in feature space.",
            "  → Post-hoc cannot recover.",
            "  → Must fix at encoder level.",
        ]),
        ("Design Principle", ACCENT2, [
            "Compatibility ∝ 1 / (signal quality)",
            "",
            "For safe CMF integration:",
            "• Avoid single fixed target (teacher)",
            "• Avoid consistent multi-epoch direction",
            "• Use bounded ascent objectives",
            "• Consider per-epoch early-stop",
            "",
            "Adaptive CMF variants (post-hoc,",
            "budget-shared) partially mitigate",
            "feedback for well-behaved methods.",
        ]),
    ]
    col_w = Inches(4.0)
    col_gap = Inches(0.25)
    col_x_start = Inches(0.35)
    for ci, (title, color, lines) in enumerate(col_data):
        cx = col_x_start + ci * (col_w + col_gap)
        box(s, cx, Inches(0.82), col_w, Inches(5.6),
            fill_color=PANEL_BG, line_color=color, line_width=Pt(1.5))
        # top colored strip
        box(s, cx, Inches(0.82), col_w, Inches(0.07), fill_color=color)
        txt(s, title, cx + Inches(0.15), Inches(0.95), col_w - Inches(0.3), Inches(0.42),
            size=Pt(14), bold=True, color=color)
        txt_block(s, lines, cx + Inches(0.15), Inches(1.45),
                  col_w - Inches(0.3), Inches(4.7),
                  size=Pt(12), color=LIGHT_TXT, leading=Pt(3))

    # ── Bottom rule ──
    box(s, Inches(0.35), Inches(6.5), Inches(12.6), Inches(0.02),
        fill_color=ACCENT2)
    txt(s, "\"What makes a machine unlearning method good in isolation is precisely what makes it dangerous when combined with CMF.\"",
        Inches(0.35), Inches(6.6), Inches(12.6), Inches(0.55),
        size=Pt(13), italic=True, color=LIGHT_TXT, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════════════
def main():
    prs = new_prs()

    slide_title(prs)
    slide_overview(prs)
    slide_neggrad(prs)
    slide_random_label(prs)
    slide_salun(prs)
    slide_scrub(prs)
    slide_tarun(prs)
    slide_summary_table(prs)
    slide_conclusion(prs)

    out = "CMF_Unlearning_Analysis.pptx"
    prs.save(out)
    print(f"Saved: {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
