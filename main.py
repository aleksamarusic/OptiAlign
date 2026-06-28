"""
Interactive optical beam alignment simulator.

Beam path:  Laser -> Mirror 1 -> Mirror 2 -> Lens -> Detector

Physics : LightPipes (Fresnel diffraction of a TEM00 Gaussian)
UI      : PyQt6
Plots   : matplotlib embedded in Qt

Run:
    pip install LightPipes PyQt6 matplotlib scipy
    python main.py
"""

import sys
import time
import random
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QPushButton, QCheckBox, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QSizePolicy,
)

import os
import datetime
from PIL import Image

from LightPipes import Begin, GaussBeam, Tilt, Fresnel, Lens, Intensity, mm, nm, mrad

# --------------------------------------------------------------------------
# Physics constants
# --------------------------------------------------------------------------
N = 128                  # grid samples  (keep at 128 for real-time performance)
SIZE = 8 * mm            # grid physical size
WAVELENGTH = 633 * nm    # HeNe red
W0 = 1.0 * mm            # source waist
Z1, Z2, Z3 = 0.25, 0.25, 0.25   # propagation distances (m): M1->M2, M2->Lens, Lens->Det

CENTER = (N - 1) / 2.0

# Degrees of freedom: (low, high, default).  Tilt in rad, offset in m, focal in m.
#
# Tilt range note: a tilted wavefront is a linear phase the FFT propagator must
# sample, so the *net* tilt must stay below the grid Nyquist limit
# (lambda / 2*dx = ~5.1 mrad here).  The two mirror tilts add, so each is capped
# at +/-2.5 mrad -> +/-5 mrad system tilt budget, which stays alias-free.
RANGES = {
    "m1tx": (-2.5 * mrad, 2.5 * mrad),
    "m1ty": (-2.5 * mrad, 2.5 * mrad),
    "m2tx": (-2.5 * mrad, 2.5 * mrad),
    "m2ty": (-2.5 * mrad, 2.5 * mrad),
    "lx":   (-2 * mm, 2 * mm),
    "ly":   (-2 * mm, 2 * mm),
    "f":    (0.8, 5.0),
}
DEFAULTS = {"m1tx": 0, "m1ty": 0, "m2tx": 0, "m2ty": 0, "lx": 0, "ly": 0, "f": 2.0}
DOF = list(RANGES.keys())

# Reset() uses a safe sub-range so the start state is a clearly visible,
# recoverable off-centre blob rather than a beam walked entirely off the grid.
RESET_TILT = 2.0 * mrad
RESET_OFF = 1.2 * mm
RESET_F = (1.5, 3.0)


# --------------------------------------------------------------------------
# Physics pipeline
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    mirror1: np.ndarray
    mirror2: np.ndarray
    lens: np.ndarray
    detector: np.ndarray
    cx: float
    cy: float
    peak: float
    ok: bool


def _centroid(I):
    sx = I.sum(axis=0)
    sy = I.sum(axis=1)
    tot = I.sum()
    if not np.isfinite(tot) or tot <= 0:
        return CENTER, CENTER, False
    x = np.arange(I.shape[0])
    cx = float(np.average(x, weights=sx))
    cy = float(np.average(x, weights=sy))
    return cx, cy, True


def simulate(p):
    """Run the full LightPipes pipeline; return intensity taps + detector metrics."""
    try:
        F = Begin(SIZE, WAVELENGTH, N)
        F = GaussBeam(F, W0)
        I_m1 = np.asarray(Intensity(F))

        F = Tilt(F, p["m1tx"], p["m1ty"])
        F = Fresnel(F, Z1)
        I_m2 = np.asarray(Intensity(F))

        F = Tilt(F, p["m2tx"], p["m2ty"])
        F = Fresnel(F, Z2)
        I_lens = np.asarray(Intensity(F))

        F = Lens(F, p["f"], p["lx"], p["ly"])
        F = Fresnel(F, Z3)
        I_det = np.asarray(Intensity(F))

        if not np.isfinite(I_det).all():
            raise FloatingPointError("non-finite intensity")

        cx, cy, ok = _centroid(I_det)
        peak = float(I_det.max())
        return SimResult(I_m1, I_m2, I_lens, I_det, cx, cy, peak, ok)
    except Exception:
        z = np.zeros((N, N))
        return SimResult(z, z, z, z, CENTER, CENTER, 0.0, False)


# Reference peak of the perfectly aligned system (used by the optimiser).
PEAK_REF = max(simulate(DEFAULTS).peak, 1e-9)


def align_cost(res):
    """Lower is better.  Centre the centroid (primary) and keep a bright spot."""
    d = np.hypot(res.cx - CENTER, res.cy - CENTER)            # pixels
    peak_pen = 2.5 * max(0.0, 1.0 - res.peak / PEAK_REF)      # discourage spread/aliased
    return d + peak_pen


# Self-align varies the beam-centring DOF (the two mirror tilts and the lens
# decentre).  Focal length is left to the user as the focus control: it does not
# move the centroid and only adds a rugged, useless dimension to the search.
FREE = ["m1tx", "m1ty", "m2tx", "m2ty", "lx", "ly"]


def pack(p):
    """Full param dict -> normalised [-1,1] vector over the free DOF."""
    return np.array([2 * (p[k] - RANGES[k][0]) / (RANGES[k][1] - RANGES[k][0]) - 1
                     for k in FREE])


def unpack(x, base):
    """Normalised free vector (+ fixed base params) -> full param dict."""
    out = dict(base)
    for v, k in zip(x, FREE):
        lo, hi = RANGES[k]
        v = min(1.0, max(-1.0, float(v)))
        out[k] = lo + (v + 1) / 2 * (hi - lo)
    return out


def cost_and_res(x, base):
    """Objective for the self-align optimiser (free DOF in normalised space)."""
    res = simulate(unpack(x, base))
    box = 10.0 * float(np.sum(np.clip(np.abs(x) - 1.0, 0, None)))   # stay in range
    if not res.ok:
        # Beam lost / undersampled: no centroid signal, so steer the simplex back
        # toward the aligned origin (zero tilt/offset) which is bright and centred.
        return 50.0 + 5.0 * float(np.sum(np.abs(np.clip(x, -1, 1)))) + box, res
    # Centring the centroid is degenerate: many tilt/offset combinations cancel
    # at the detector but leave the *beam path* zig-zagging off-axis.  A small
    # preference for minimal tilt/offset selects the true aligned state (DOF ~ 0),
    # so the bench schematic, the sliders, and the heatmap all converge together.
    reg = 2.0 * float(np.sum(np.square(x)))
    return align_cost(res) + box + reg, res


# --------------------------------------------------------------------------
# Optimiser worker (runs off the UI thread)
#
# Two phases: (1) solve for the aligned solution silently, then (2) play a
# smooth, eased glide from the start state to that solution.  Decoupling the two
# means the on-screen motion is a steady "beam drifts in and tightens onto the
# centre" walk, instead of the optimiser's jumpy internal search path.
# --------------------------------------------------------------------------
class AlignWorker(QThread):
    progress = pyqtSignal(object, object)   # (params dict, SimResult)
    done = pyqtSignal(object)               # final aligned params dict

    GLIDE_STEPS = 60        # frames in the walk-in animation
    GLIDE_FRAME = 0.03      # seconds per frame  (~3.5 s total glide)

    def __init__(self, start_params):
        super().__init__()
        self.base = dict(start_params)        # focal length held fixed
        # If the beam is effectively lost (no usable centroid signal: not finite,
        # or a near-zero peak from undersampling), there is no gradient to follow,
        # so recover from the aligned neutral instead of the current point.
        r0 = simulate(start_params)
        lost = (not r0.ok) or (r0.peak < 0.15 * PEAK_REF)
        # Solve from the aligned origin (zero tilt/offset).  Centring the centroid
        # is degenerate -- many tilt combinations cancel at the detector but leave
        # the beam path zig-zagging off-axis -- so searching from the origin
        # converges to the *minimal-tilt* solution, which lands the bench beam (not
        # just the heatmap) on the detector centre.
        self.x0 = np.zeros(len(FREE))
        # The glide still starts from the real misaligned sliders (or the neutral,
        # if the beam was lost and there is nothing meaningful to glide from).
        self.anim_start = unpack(self.x0, self.base) if lost else dict(start_params)

    def _solve(self):
        """Find the aligned solution silently (no animation emitted)."""
        best = {"cost": float("inf"), "x": self.x0.copy()}

        def cost(x):
            c, _ = cost_and_res(x, self.base)
            if c < best["cost"]:
                best["cost"] = c
                best["x"] = np.asarray(x).copy()
            return c

        # Explicit initial simplex whose vertices step *inward* (toward the centre
        # of range) so DOF starting near a boundary or near zero are actively
        # explored, which Nelder-Mead's default simplex does poorly.
        n = len(self.x0)
        signs = np.where(self.x0 >= 0, -1.0, 1.0)
        simplex = np.vstack([self.x0] + [self.x0 + 0.5 * signs[i] * np.eye(n)[i]
                                         for i in range(n)])
        minimize(cost, self.x0, method="Nelder-Mead",
                 options={"initial_simplex": simplex, "maxfev": 120,
                          "xatol": 0.005, "fatol": 0.01})
        return unpack(best["x"], self.base)

    def run(self):
        final = self._solve()
        keys = list(RANGES.keys())
        for i in range(self.GLIDE_STEPS + 1):
            t = i / self.GLIDE_STEPS
            e = t * t * (3 - 2 * t)            # smoothstep ease-in / ease-out
            p = {k: self.anim_start[k] + (final[k] - self.anim_start[k]) * e
                 for k in keys}
            self.progress.emit(p, simulate(p))
            time.sleep(self.GLIDE_FRAME)
        self.done.emit(final)


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------
TICKS = 1000  # int slider resolution


class LabeledSlider(QWidget):
    changed = pyqtSignal()

    def __init__(self, title, key, divisor, suffix, decimals):
        super().__init__()
        self.key = key
        self.lo, self.hi = RANGES[key]
        self.divisor = divisor
        self.suffix = suffix
        self.decimals = decimals

        self.title = QLabel(title)
        self.title.setStyleSheet("color:#ccc;font-size:10px;")
        self.value = QLabel("")
        self.value.setStyleSheet("color:#7fd;font-size:10px;")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, TICKS)
        self.slider.valueChanged.connect(self._on_change)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title)
        head.addWidget(self.value)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addWidget(self.slider)

        self.set_physical(DEFAULTS[key])

    def physical(self):
        return self.lo + self.slider.value() / TICKS * (self.hi - self.lo)

    def set_physical(self, val, emit=False):
        val = min(self.hi, max(self.lo, val))
        tick = round((val - self.lo) / (self.hi - self.lo) * TICKS)
        if not emit:
            self.slider.blockSignals(True)
        self.slider.setValue(int(tick))
        if not emit:
            self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self):
        v = self.physical() / self.divisor
        self.value.setText(f"{v:+.{self.decimals}f}{self.suffix}")

    def _on_change(self):
        self._update_label()
        self.changed.emit()


def mini_canvas():
    fig = Figure(figsize=(1.25, 1.25), facecolor="#1b1b1b")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.axis("off")
    im = ax.imshow(np.zeros((N, N)), cmap="inferno", origin="lower",
                   interpolation="nearest", vmin=0, vmax=1)
    c = FigureCanvas(fig)
    c.setFixedSize(110, 110)
    return c, im


def component_column(title, sliders):
    """A column: component label, mini-heatmap, then its sliders."""
    box = QFrame()
    box.setFrameShape(QFrame.Shape.StyledPanel)
    box.setStyleSheet("QFrame{background:#181818;border:1px solid #333;border-radius:6px;}")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(3)

    lbl = QLabel(title)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("color:#fff;font-weight:bold;font-size:11px;border:0;")
    lay.addWidget(lbl)

    canvas, im = mini_canvas()
    holder = QHBoxLayout()
    holder.addStretch()
    holder.addWidget(canvas)
    holder.addStretch()
    lay.addLayout(holder)

    for s in sliders:
        lay.addWidget(s)
    lay.addStretch()
    return box, im


# --------------------------------------------------------------------------
# Optical bench schematic (live, top view)
#
# Folded periscope path:  Laser -> M1 (up) -> M2 (right) -> Lens -> Detector.
# The beam is ray-traced in the figure plane: each mirror is modelled by its
# surface normal rotated with the tilt slider, so the beam obeys the reflection
# law and always pivots ON the drawn mirror.  The lens applies a thin-lens slope
# kick (gently converging, tied to the focal slider).
#
# Visual scaling: real mirror tilts are microscopic, so a literal scale would be
# invisible.  These exaggerations are chosen so the *beam walk* is the visible
# cue while everything stays in-frame and on the optics across the full slider
# range; the mirror rotation stays consistent with the 2x reflection law.
# This is a top view, so it shows the X-plane DOF (mirror tilt-X, lens offset-X,
# focal); the tilt-Y / offset-Y sliders act out of this plane.
# --------------------------------------------------------------------------
SCHEM_E = 10.0       # mirror tilt (rad) -> mirror rotation (rad)
SCHEM_FU = 4.0       # focal length (m) -> visual focal (figure units)
SCHEM_OFF = 120.0    # lens offset (m) -> figure-unit lens shift
_YB, _YT = 0.5, 1.5
_NBAR = np.array([-1.0, 1.0]) / np.sqrt(2.0)       # nominal '/' mirror normal
_BARDIR = np.array([1.0, 1.0]) / np.sqrt(2.0)      # nominal '/' mirror bar


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def trace_beam(p):
    """Ray-trace the folded path in figure coords; return node points + optics state."""
    pos = np.array([1.0, _YB]); d = np.array([1.0, 0.0])
    pts = [pos.copy()]
    pos = pos + d * (3.0 - pos[0]) / d[0]; pts.append(pos.copy())          # hit M1
    phi1 = p["m1tx"] * SCHEM_E
    n = _rot(_NBAR, phi1); d = d - 2 * np.dot(d, n) * n                    # reflect
    pos = pos + d * (_YT - pos[1]) / d[1]; pts.append(pos.copy())          # hit M2
    phi2 = p["m2tx"] * SCHEM_E
    n = _rot(_NBAR, phi2); d = d - 2 * np.dot(d, n) * n                    # reflect
    pos = pos + d * (6.0 - pos[0]) / d[0]; lens_hit = pos.copy(); pts.append(pos.copy())
    lcy = _YT + p["lx"] * SCHEM_OFF
    slope = d[1] / d[0] - (lens_hit[1] - lcy) / (p["f"] * SCHEM_FU)        # thin lens
    d = np.array([1.0, slope]) / np.hypot(1.0, slope)
    pos = pos + d * (8.7 - pos[0]) / d[0]; pts.append(pos.copy())          # detector
    return np.array(pts), phi1, phi2, lcy


def draw_schematic(ax, p=None):
    if p is None:
        p = DEFAULTS
    ax.clear()
    ax.set_facecolor("#101010")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.25)
    ax.axis("off")

    pts, phi1, phi2, lcy = trace_beam(p)
    beam = "#ff3333"
    ax.plot(pts[:, 0], pts[:, 1], color=beam, lw=2, zorder=1)

    # direction arrowheads on the first three legs
    for a, b in zip(pts[:3], pts[1:4]):
        v = b - a
        L = np.hypot(v[0], v[1])
        if L > 0.1:
            mid = (a + b) / 2.0
            u = v / L * 0.12
            ax.annotate("", xy=mid + u, xytext=mid - u,
                        arrowprops=dict(arrowstyle="-|>", color="#ff8888", lw=1.4))

    # laser body
    ax.add_patch(mpatches.Rectangle((0.5, _YB - 0.22), 0.5, 0.44,
                 fc="#444", ec="#888", zorder=2))

    # fold mirrors, rotated consistently with the reflection
    for cx, cy, phi in [(3.0, _YB, phi1), (3.0, _YT, phi2)]:
        e = _rot(_BARDIR, phi) * 0.25
        ax.plot([cx - e[0], cx + e[0]], [cy - e[1], cy + e[1]],
                color="#7fdfff", lw=5, solid_capstyle="round", zorder=3)

    # lens (slides vertically with offset-X) + detector
    ax.add_patch(mpatches.Ellipse((6.0, lcy), 0.2, 0.62,
                 fc="#88bbff55", ec="#88bbff", lw=2, zorder=3))
    ax.add_patch(mpatches.Rectangle((8.65, _YT - 0.34), 0.22, 0.68,
                 fc="#222", ec="#ccc", zorder=3))

    names = {"Laser": (1.0, _YB), "Mirror 1": (3.0, _YB), "Mirror 2": (3.0, _YT),
             "Lens": (6.0, _YT), "Detector": (8.7, _YT)}
    for name, (px, py) in names.items():
        below = (name == "Mirror 1")
        ax.text(px, py - 0.34 if below else py + 0.40, name,
                color="#ddd", ha="center", va="top" if below else "bottom",
                fontsize=8.5)
    ax.text(0.1, 2.2, "top view (X-plane)", color="#666", fontsize=7,
            ha="left", va="top")


def qpixmap_to_pil(pix):
    """Convert a QPixmap (window grab) to a PIL RGB image for GIF encoding."""
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    b = img.constBits()
    b.setsize(img.sizeInBytes())
    arr = np.frombuffer(b, dtype=np.uint8).reshape((h, img.bytesPerLine() // 4, 4))
    return Image.fromarray(arr[:, :w, :3].copy(), "RGB")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class Simulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Beam Alignment Simulator")
        self.setStyleSheet("background:#0e0e0e;")
        self.resize(1280, 760)

        self._pending = False
        self._worker = None
        self._record = False
        self._frames = []

        self._build_ui()
        self.update_view()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # ----- LEFT: optical bench -----
        left = QVBoxLayout()

        bench_fig = Figure(figsize=(6, 1.9), facecolor="#101010")
        bench_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.bench_ax = bench_fig.add_subplot(111)
        self._init_schematic()
        self.bench_canvas = FigureCanvas(bench_fig)
        self.bench_canvas.setFixedHeight(200)
        left.addWidget(self.bench_canvas)

        # component columns
        self.m1tx = LabeledSlider("tilt X", "m1tx", mrad, " mr", 2)
        self.m1ty = LabeledSlider("tilt Y", "m1ty", mrad, " mr", 2)
        self.m2tx = LabeledSlider("tilt X", "m2tx", mrad, " mr", 2)
        self.m2ty = LabeledSlider("tilt Y", "m2ty", mrad, " mr", 2)
        self.lx = LabeledSlider("offset X", "lx", mm, " mm", 2)
        self.ly = LabeledSlider("offset Y", "ly", mm, " mm", 2)
        self.lf = LabeledSlider("focal", "f", 1.0, " m", 2)

        self.sliders = [self.m1tx, self.m1ty, self.m2tx, self.m2ty, self.lx, self.ly, self.lf]
        for s in self.sliders:
            s.changed.connect(self.schedule_update)

        cols = QHBoxLayout()
        cols.setSpacing(6)

        laser_box = QFrame()
        laser_box.setStyleSheet("QFrame{background:#181818;border:1px solid #333;border-radius:6px;}")
        lb = QVBoxLayout(laser_box)
        ltitle = QLabel("Laser\n633 nm")
        ltitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ltitle.setStyleSheet("color:#ff6666;font-weight:bold;font-size:11px;border:0;")
        lb.addWidget(ltitle)
        lb.addStretch()
        cols.addWidget(laser_box, 1)

        box_m1, self.im_m1 = component_column("Mirror 1", [self.m1tx, self.m1ty])
        box_m2, self.im_m2 = component_column("Mirror 2", [self.m2tx, self.m2ty])
        box_ln, self.im_ln = component_column("Lens", [self.lx, self.ly, self.lf])
        box_dt, self.im_dt = component_column("Detector", [])
        for b in (box_m1, box_m2, box_ln, box_dt):
            cols.addWidget(b, 1)
        left.addLayout(cols)

        # buttons
        btns = QHBoxLayout()
        self.align_btn = QPushButton("Self-Align")
        self.reset_btn = QPushButton("Reset")
        for b in (self.align_btn, self.reset_btn):
            b.setMinimumHeight(38)
            b.setStyleSheet(
                "QPushButton{background:#2a2a2a;color:#fff;border:1px solid #555;"
                "border-radius:6px;font-size:13px;font-weight:bold;}"
                "QPushButton:hover{background:#3a3a3a;}"
                "QPushButton:disabled{color:#888;}")
        self.align_btn.clicked.connect(self.self_align)
        self.reset_btn.clicked.connect(self.reset)
        btns.addWidget(self.align_btn)
        btns.addWidget(self.reset_btn)
        left.addLayout(btns)

        # record the next alignment to a GIF (opt-in so we don't write files
        # on every run)
        opts = QHBoxLayout()
        self.save_chk = QCheckBox("Save alignment animation to GIF")
        self.save_chk.setStyleSheet("color:#bbb;font-size:11px;")
        self.status = QLabel("")
        self.status.setStyleSheet("color:#7d7;font-size:11px;")
        opts.addWidget(self.save_chk)
        opts.addStretch()
        opts.addWidget(self.status)
        left.addLayout(opts)

        root.addLayout(left, 3)

        # ----- RIGHT: detector view -----
        right = QVBoxLayout()
        title = QLabel("Detector — intensity")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color:#fff;font-size:14px;font-weight:bold;")
        right.addWidget(title)

        det_fig = Figure(figsize=(5, 5), facecolor="#0e0e0e")
        det_fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.04)
        self.det_ax = det_fig.add_subplot(111)
        self.det_ax.set_facecolor("#000")
        self.det_im = self.det_ax.imshow(np.zeros((N, N)), cmap="inferno",
                                         origin="lower", extent=[0, N, 0, N],
                                         interpolation="nearest", vmin=0, vmax=1)
        self.det_ax.set_xticks([]); self.det_ax.set_yticks([])
        # crosshair (centroid) + fixed centre marker
        self.cross_v = self.det_ax.axvline(CENTER, color="#00ff88", lw=0.8, alpha=0.9)
        self.cross_h = self.det_ax.axhline(CENTER, color="#00ff88", lw=0.8, alpha=0.9)
        self.det_ax.axvline(CENTER, color="#ffffff", lw=0.5, ls=":", alpha=0.35)
        self.det_ax.axhline(CENTER, color="#ffffff", lw=0.5, ls=":", alpha=0.35)
        self.err_text = self.det_ax.text(
            N / 2, N / 2, "", color="#ff5555", ha="center", va="center",
            fontsize=13, fontweight="bold", visible=False)
        self.det_canvas = FigureCanvas(det_fig)
        self.det_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.addWidget(self.det_canvas, 1)

        # numeric readout.  Fixed height + no word-wrap so the label can never
        # reflow and resize the detector canvas frame-to-frame during the
        # animation (which made the heatmap look like it was changing size).
        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.readout.setWordWrap(False)
        self.readout.setFixedHeight(26)
        self.readout.setStyleSheet("color:#9fe;font-size:12px;font-family:monospace;")
        right.addWidget(self.readout)

        root.addLayout(right, 2)

    # ---- parameters ------------------------------------------------------
    def current_params(self):
        return {s.key: s.physical() for s in self.sliders}

    def set_params(self, p):
        for s in self.sliders:
            s.set_physical(p[s.key], emit=False)

    # ---- update / render -------------------------------------------------
    def schedule_update(self):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.update_view)

    def update_view(self):
        self._pending = False
        res = simulate(self.current_params())
        self.render(res)

    def render(self, res):
        self.im_m1.set_data(res.mirror1); self.im_m1.set_clim(0, max(res.mirror1.max(), 1e-9))
        self.im_m2.set_data(res.mirror2); self.im_m2.set_clim(0, max(res.mirror2.max(), 1e-9))
        self.im_ln.set_data(res.lens);    self.im_ln.set_clim(0, max(res.lens.max(), 1e-9))
        self.im_dt.set_data(res.detector); self.im_dt.set_clim(0, max(res.detector.max(), 1e-9))
        for im in (self.im_m1, self.im_m2, self.im_ln, self.im_dt):
            im.figure.canvas.draw_idle()

        if res.ok:
            self.err_text.set_visible(False)
            self.det_im.set_data(res.detector)
            self.det_im.set_clim(0, max(res.peak, 1e-9))
            self.cross_v.set_xdata([res.cx, res.cx])
            self.cross_h.set_ydata([res.cy, res.cy])
            self.cross_v.set_visible(True)
            self.cross_h.set_visible(True)
            dx_mm = (res.cx - CENTER) * (SIZE / N) / mm
            dy_mm = (res.cy - CENTER) * (SIZE / N) / mm
            # Fixed-width fields so the string length is identical every frame.
            self.readout.setText(
                f"centroid  ({res.cx:6.2f}, {res.cy:6.2f}) px   "
                f"({dx_mm:+.3f}, {dy_mm:+.3f}) mm   peak {res.peak:6.3f}")
        else:
            self.det_im.set_data(np.zeros((N, N)))
            self.cross_v.set_visible(False)
            self.cross_h.set_visible(False)
            self.err_text.set_text("Propagation error — adjust parameters")
            self.err_text.set_visible(True)
            self.readout.setText("centroid  (  —   ,   —   ) px   "
                                 "(  —   ,   —   ) mm   peak   —  ")
        self.det_canvas.draw_idle()

        # live bench schematic reflecting the current alignment (top view)
        self._update_schematic(self.current_params())
        self.bench_canvas.draw_idle()

    # ---- live schematic (persistent artists; only the beam/mirrors/lens move,
    #      so each frame is a cheap set_data instead of a full redraw) ---------
    def _init_schematic(self):
        ax = self.bench_ax
        ax.clear()
        ax.set_facecolor("#101010")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2.25)
        ax.axis("off")

        # static elements
        ax.add_patch(mpatches.Rectangle((0.5, _YB - 0.22), 0.5, 0.44,
                     fc="#444", ec="#888", zorder=2))                       # laser
        ax.add_patch(mpatches.Rectangle((8.65, _YT - 0.34), 0.22, 0.68,
                     fc="#222", ec="#ccc", zorder=3))                       # detector
        for tail, tip in [((1.7, _YB), (2.0, _YB)), ((3.0, 0.9), (3.0, 1.15)),
                          ((4.2, _YT), (4.5, _YT))]:                        # flow arrows
            ax.annotate("", xy=tip, xytext=tail,
                        arrowprops=dict(arrowstyle="-|>", color="#ff8888", lw=1.4))
        names = {"Laser": (1.0, _YB), "Mirror 1": (3.0, _YB), "Mirror 2": (3.0, _YT),
                 "Lens": (6.0, _YT), "Detector": (8.7, _YT)}
        for name, (px, py) in names.items():
            below = (name == "Mirror 1")
            ax.text(px, py - 0.34 if below else py + 0.40, name,
                    color="#ddd", ha="center", va="top" if below else "bottom",
                    fontsize=8.5)
        ax.text(0.1, 2.2, "top view (X-plane)", color="#666", fontsize=7,
                ha="left", va="top")

        # dynamic artists (updated in place each frame)
        self.sch_beam, = ax.plot([], [], color="#ff3333", lw=2, zorder=1)
        self.sch_m1, = ax.plot([], [], color="#7fdfff", lw=5,
                               solid_capstyle="round", zorder=3)
        self.sch_m2, = ax.plot([], [], color="#7fdfff", lw=5,
                               solid_capstyle="round", zorder=3)
        self.sch_lens = mpatches.Ellipse((6.0, _YT), 0.2, 0.62,
                                         fc="#88bbff55", ec="#88bbff", lw=2, zorder=3)
        ax.add_patch(self.sch_lens)
        self._update_schematic(DEFAULTS)

    def _update_schematic(self, p):
        pts, phi1, phi2, lcy = trace_beam(p)
        self.sch_beam.set_data(pts[:, 0], pts[:, 1])
        e1 = _rot(_BARDIR, phi1) * 0.25
        self.sch_m1.set_data([3.0 - e1[0], 3.0 + e1[0]], [_YB - e1[1], _YB + e1[1]])
        e2 = _rot(_BARDIR, phi2) * 0.25
        self.sch_m2.set_data([3.0 - e2[0], 3.0 + e2[0]], [_YT - e2[1], _YT + e2[1]])
        self.sch_lens.set_center((6.0, lcy))

    # ---- buttons ---------------------------------------------------------
    def reset(self):
        p = {
            "m1tx": random.uniform(-RESET_TILT, RESET_TILT),
            "m1ty": random.uniform(-RESET_TILT, RESET_TILT),
            "m2tx": random.uniform(-RESET_TILT, RESET_TILT),
            "m2ty": random.uniform(-RESET_TILT, RESET_TILT),
            "lx": random.uniform(-RESET_OFF, RESET_OFF),
            "ly": random.uniform(-RESET_OFF, RESET_OFF),
            "f": random.uniform(*RESET_F),
        }
        self.set_params(p)
        self.align_btn.setText("Self-Align")
        self.update_view()

    def _set_controls_enabled(self, en):
        for s in self.sliders:
            s.setEnabled(en)
        self.reset_btn.setEnabled(en)

    def self_align(self):
        if self._worker is not None:
            return
        self.align_btn.setText("Aligning…")
        self.align_btn.setEnabled(False)
        self._set_controls_enabled(False)

        self._record = self.save_chk.isChecked()
        self._frames = []
        if self._record:
            self.save_chk.setEnabled(False)
            self.status.setText("Recording…")
        else:
            self.status.setText("")

        self._worker = AlignWorker(self.current_params())
        self._worker.progress.connect(self._on_align_progress)
        self._worker.done.connect(self._on_align_done)
        self._worker.start()

    def _on_align_progress(self, params, res):
        self.set_params(params)        # move sliders so the user watches them walk
        self.render(res)
        if self._record:
            self._capture_frame()

    def _on_align_done(self, params):
        self.set_params(params)
        self.update_view()
        self.align_btn.setText("Aligned ✓")
        self.align_btn.setEnabled(True)
        self._set_controls_enabled(True)
        self._worker = None

        if self._record and self._frames:
            try:
                path = self._save_gif()
                self.status.setText(f"Saved {os.path.basename(path)}  "
                                    f"({len(self._frames)} frames)")
                print("Saved alignment animation:", path)
            except Exception as exc:                       # never crash on save
                self.status.setText(f"Save failed: {exc}")
        self._frames = []
        self.save_chk.setEnabled(True)

    # ---- animation recording --------------------------------------------
    def _capture_frame(self):
        # Force the deferred canvas draws so the grab reflects this exact frame.
        self.det_canvas.draw()
        self.bench_canvas.draw()
        for im in (self.im_m1, self.im_m2, self.im_ln, self.im_dt):
            im.figure.canvas.draw()
        pix = self.grab()
        if pix.width() > 1000:                             # keep the GIF a sane size
            pix = pix.scaledToWidth(1000, Qt.TransformationMode.SmoothTransformation)
        self._frames.append(qpixmap_to_pil(pix))

    def _save_gif(self):
        out_dir = os.path.dirname(os.path.abspath(__file__))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"alignment_{ts}.gif")
        frames = self._frames
        durations = [40] * len(frames)
        durations[-1] = 1500                               # hold the aligned result
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        return path


def main():
    app = QApplication(sys.argv)
    win = Simulator()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
