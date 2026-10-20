#!/usr/bin/env python3
"""Generate the TornadoGCM end-to-end architecture/information-flow SVG.

The figure answers one question: how does TornadoGCM move from inputs through
a PZHA-partitioned model step into distributed training or inference, while
the SDA control plane governs precision and recovers from numerical failure?

It is a C-type information-flow diagram with A-type state-placement
annotations.  All coordinates are explicit so the layout can be checked for
partial overlaps and line/box collisions before publishing.

Usage:
    python docs/assets/diagram_src/build_architecture_svg.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

W, H = 2040, 1400

# Semantic palette (values mirror the diagram_in_arch_org visual system).
BLUE = "#1F6FB2"
BLUE_FILL = "#E3EFF9"
GREEN = "#2E8B57"
GREEN_FILL = "#E4F3EA"
RED = "#C0392B"
RED_FILL = "#FAE6E3"
AMBER = "#E08A1E"
AMBER_FILL = "#FDF1DC"
PURPLE = "#7B5EA7"
PURPLE_FILL = "#F1ECF9"
GRAY = "#4A4A4A"
GRAY_FILL = "#F2F2F2"
INK = "#1C1C1C"
SOFT = "#5C5C5C"
RULE = "#D6D6D6"
LANE_TRAIN = "#F7F7F7"
LANE_MODEL = "#F1F4F7"
LANE_INFER = "#F4F0E9"
LANE_GOV = "#F7F4FB"

FONT = "Segoe UI, Helvetica, Arial, sans-serif"
MONO = "Cascadia Mono, Consolas, monospace"
FONT_SCALE = 1.06


@dataclass
class Node:
    name: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class Label:
    x: float
    y: float
    text: str
    size: float = 14
    fill: str = INK
    anchor: str = "start"
    weight: str = "400"
    mono: bool = False
    owner: str = ""


@dataclass
class Edge:
    points: list[tuple[float, float]]
    kind: str
    label: str = ""
    label_at: tuple[float, float] | None = None
    label_size: float = 12.5
    exclude: set[str] = field(default_factory=set)
    num: str = ""
    num_at: tuple[float, float] | None = None


KIND_STYLE = {
    "data": dict(stroke=BLUE, width=2.2, dash="", marker="arrow", both=False),
    "grad": dict(stroke=RED, width=3.6, dash="", marker="arrow", both=True),
    "ctrl": dict(stroke=PURPLE, width=2.0, dash="7 5", marker="arrow", both=False),
    "fail": dict(stroke=RED, width=2.0, dash="1.5 5", marker="arrow", both=False),
    "io": dict(stroke=GRAY, width=1.8, dash="4 4", marker="arrow", both=False),
    "flow": dict(stroke=GRAY, width=1.8, dash="", marker="arrow", both=False),
    "loop": dict(stroke=SOFT, width=1.7, dash="5 5", marker="arrow", both=False),
    "io_gap": dict(stroke=GRAY, width=1.8, dash="4 4", marker="", both=False),
    "ctrl_gap": dict(stroke=PURPLE, width=2.0, dash="7 5", marker="", both=False),
}


class Scene:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.nodes: list[Node] = []
        self.labels: list[Label] = []
        self.edges: list[Edge] = []

    # ---------- primitives ----------
    def raw(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.4,
             rx=0, dash="", name="", opacity=1.0) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
            f'opacity="{opacity}"/>'
        )
        if name:
            self.nodes.append(Node(name, x, y, w, h))

    def text(self, x, y, s, size=14, fill=INK, anchor="start", weight="400",
             mono=False, owner="", halo=False) -> None:
        size = round(size * FONT_SCALE, 2)
        family = MONO if mono else FONT
        esc = (s.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
        cls = ' class="halo"' if halo else ""
        self.raw(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{cls}>'
            f'{esc}</text>'
        )
        if owner or s:
            self.labels.append(Label(x, y, s, size, fill, anchor, weight, mono, owner))

    def lines(self, points, stroke, width=1.8, dash="", marker="", both=False,
              start=False, name="") -> None:
        pts = " ".join(f"{x},{y}" for x, y in points)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ""
        if marker:
            m += f' marker-end="url(#{marker})"'
        if both:
            m += f' marker-start="url(#{marker})"'
        self.raw(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round"{d}{m}/>'
        )

    def edge(self, points, kind, label="", label_at=None, label_size=12.5,
             exclude=(), num="", num_at=None) -> None:
        st = KIND_STYLE[kind]
        self.lines(points, st["stroke"], st["width"], st["dash"], st["marker"],
                   both=st["both"])
        e = Edge([(float(x), float(y)) for x, y in points], kind, label,
                 label_at, label_size, set(exclude), num, num_at)
        self.edges.append(e)
        if label and label_at:
            self.text(label_at[0], label_at[1], label, label_size, st["stroke"],
                      anchor="middle", owner=f"edge:{id(e)}", halo=True)
        if num and num_at:
            self.badge(num_at[0], num_at[1], num)

    def badge(self, x, y, n) -> None:
        self.raw(f'<circle cx="{x}" cy="{y}" r="11" fill="#FFFFFF" '
                 f'stroke="{GRAY}" stroke-width="1.4"/>')
        self.text(x, y + 4.5, str(n), 13, INK, anchor="middle", weight="600")

    def diamond(self, x, y, size=7) -> None:
        self.raw(
            f'<path d="M{x},{y-size} L{x+size},{y} L{x},{y+size} '
            f'L{x-size},{y} Z" fill="{PURPLE}" stroke="#FFFFFF" stroke-width="1.5"/>'
        )

    def cylinder(self, x, y, w, h, title, sub, color=BLUE, fill=BLUE_FILL,
                 name="") -> None:
        ry = 10
        self.raw(
            f'<path d="M{x},{y+ry} A{w/2},{ry} 0 0 1 {x+w},{y+ry} '
            f'V{y+h-ry} A{w/2},{ry} 0 0 1 {x},{y+h-ry} Z" '
            f'fill="{fill}" stroke="{color}" stroke-width="1.8"/>'
        )
        self.raw(
            f'<path d="M{x},{y+ry} A{w/2},{ry} 0 0 0 {x+w},{y+ry}" '
            f'fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        self.text(x + w / 2, y + h / 2 - 4, title, 14, color, anchor="middle",
                  weight="600", owner=name)
        self.text(x + w / 2, y + h / 2 + 14, sub, 11.5, SOFT, anchor="middle",
                  owner=name)
        self.nodes.append(Node(name or title, x, y, w, h))

    def doc(self, x, y, w, h, title, sub, color=PURPLE, fill=PURPLE_FILL,
            name="") -> None:
        fold = 16
        self.raw(
            f'<path d="M{x},{y} H{x+w-fold} L{x+w},{y+fold} V{y+h} H{x} Z" '
            f'fill="{fill}" stroke="{color}" stroke-width="1.8"/>'
        )
        self.raw(
            f'<path d="M{x+w-fold},{y} V{y+fold} H{x+w}" fill="none" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
        self.text(x + 12, y + 27, title, 14, color, weight="600", owner=name)
        self.text(x + 12, y + 46, sub, 11.5, SOFT, owner=name)
        self.nodes.append(Node(name or title, x, y, w, h))

    def box(self, x, y, w, h, title, sub="", color=GRAY, fill=GRAY_FILL,
            name="", title_size=14.5, sub_size=11.5, rx=6, dash="") -> None:
        self.rect(x, y, w, h, fill, color, 1.8, rx, dash, name or title)
        lines = title.split("\n")
        ty = y + 22
        for i, line in enumerate(lines):
            self.text(x + 12, ty + i * 18, line, title_size, color,
                      weight="600", owner=name or title)
        if sub:
            sy = ty + len(lines) * 18 + 2
            for i, line in enumerate(sub.split("\n")):
                self.text(x + 12, sy + i * 15, line, sub_size, SOFT,
                          owner=name or title)

    def gpu_card(self, x, y, w, h, rank) -> None:
        color = GRAY
        self.rect(x, y, w, h, "#FFFFFF", color, 2.0, 4)
        # bracket
        self.rect(x - 6, y + 4, 6, h - 8, "#E9E9E9", color, 1.2, 1)
        # board
        self.rect(x + 10, y + 14, w - 20, h - 42, "#FAFAFA", color, 1.4, 3)
        # die
        self.rect(x + 22, y + 28, 34, 34, "#EAEFF4", color, 1.6, 2)
        self.text(x + 39, y + 49, "SM", 10, color, anchor="middle")
        # HBM stacks
        for i in range(4):
            hx = x + 66 + (i % 2) * 22
            hy = y + 26 + (i // 2) * 22
            self.rect(hx, hy, 16, 16, BLUE_FILL, BLUE, 1.0, 1)
        # gold finger
        self.rect(x + 14, y + h - 22, w - 28, 10, "#FFFFFF", color, 1.2, 1)
        for i in range(8):
            gx = x + 18 + i * 11
            self.raw(f'<path d="M{gx},{y+h-21} V{y+h-13}" stroke="{color}" '
                     f'stroke-width="1"/>')
        self.text(x + w / 2, y + h - 4, rank, 11, color, anchor="middle",
                  weight="600", owner=f"gpu{rank}")
        self.nodes.append(Node(f"gpu{rank}", x - 6, y, w + 6, h))

    def tensor_grid(self, x, y, w, h, name="") -> None:
        self.raw(
            f'<path d="M{x},{y+14} L{x+18},{y} H{x+w} V{y+h-14} '
            f'L{x+w-18},{y+h} H{x} Z" fill="{BLUE_FILL}" stroke="{BLUE}" '
            f'stroke-width="1.8"/>'
        )
        cols, rows = 5, 4
        cw, ch = (w - 18) / cols, (h - 14) / rows
        for c in range(cols):
            for r in range(rows):
                shade = "#FFFFFF" if (c + r) % 2 else BLUE_FILL
                self.raw(
                    f'<rect x="{x + c*cw}" y="{y + 14 + r*ch}" '
                    f'width="{cw}" height="{ch}" fill="{shade}" '
                    f'stroke="{BLUE}" stroke-opacity=".35" stroke-width=".8"/>'
                )
        self.nodes.append(Node(name or "tensor", x, y, w, h))

    def header(self) -> None:
        self.text(48, 52, "TornadoGCM — end-to-end system flow", 31, INK,
                  weight="700")
        self.text(48, 84,
                  "PyTorch NeuralGCM · PZHA precision zones · SDA control plane · distributed training · production inference",
                  17, SOFT)
        self.text(48, 112,
                  "1 batch → 2 forward K steps → 3 PZHA model step → 4 loss + all-reduce → 5 checkpoint → 6 rollout → 7 precision hot-swap → 8 failure recovery",
                  14.5, GRAY)

    def lanes(self) -> None:
        self.rect(36, 140, 1600, 310, LANE_TRAIN, "#E6E6E6", 1.0, 8)
        self.text(52, 170, "TRAINING PLANE", 18, INK, weight="700")
        self.text(52, 192, "distributed · K-step rollout · DDP / FSDP / DTensor", 13.5, SOFT)

        self.rect(36, 480, 1600, 420, LANE_MODEL, "#DDE5EC", 1.0, 8)
        self.text(52, 510, "SHARED MODEL STEP · one rank's view", 18, INK, weight="700")
        self.text(52, 534, "train + infer share this step · diamonds = zone casts", 13.5, SOFT)

        self.rect(36, 930, 1600, 290, LANE_INFER, "#EDE6DA", 1.0, 8)
        self.text(52, 960, "INFERENCE & EVALUATION PLANE", 18, INK, weight="700")
        self.text(52, 982, "chunked rollout · NaN guard / rollback · Zarr output", 13.5, SOFT)

        self.rect(1670, 140, 334, 1080, LANE_GOV, "#E6DDF2", 1.2, 8)
        self.text(1688, 170, "SDA / PZHA", 18, PURPLE, weight="700")
        self.text(1688, 192, "precision governance plane", 13, SOFT)

    def legend(self) -> None:
        x, y, w, h = 36, 1245, 1968, 120
        self.rect(x, y, w, h, "#FFFFFF", RULE, 1.0, 6)
        self.text(x + 14, y + 22, "LEGEND", 14, INK, weight="700")
        # row 1: shapes / fills
        items = [
            ("data", BLUE, BLUE_FILL, "data / state tensor / I/O"),
            ("Z3", GREEN, GREEN_FILL, "Z3 learned physics (BF16)"),
            ("Z1", AMBER, AMBER_FILL, "Z1 dynamics (FP32 / TF32)"),
            ("Z2", RED, RED_FILL, "Z2 conservation fixer (FP64)"),
            ("gov", PURPLE, PURPLE_FILL, "SDA / precision control"),
            ("infra", GRAY, GRAY_FILL, "infrastructure / orchestration"),
        ]
        cx = x + 14
        for key, color, fill, label in items:
            self.rect(cx, y + 34, 24, 16, fill, color, 1.4, 3)
            self.text(cx + 32, y + 47, label, 12, SOFT)
            cx += 315
        # row 2: line semantics
        cx = x + 14
        for kind, label in [
            ("data", "tensor / data flow"),
            ("grad", "NCCL all-reduce (Σg)"),
            ("ctrl", "policy / config control"),
            ("fail", "failure / rollback"),
            ("io", "artifact I/O (disk)"),
            ("loop", "rollout / repeat loop"),
        ]:
            st = KIND_STYLE[kind]
            self.lines([(cx, y + 76), (cx + 42, y + 76)], st["stroke"],
                       st["width"], st["dash"], st["marker"], both=st["both"])
            self.text(cx + 50, y + 81, label, 12, SOFT)
            cx += 315
        # row 3: glyphs
        self.cylinder(x + 14, y + 92, 34, 20, "", "", BLUE, BLUE_FILL, "legend-cyl")
        self.text(x + 56, y + 107, "artifact (dataset / checkpoint / forecast)", 11.5, SOFT)
        self.diamond(x + 340, y + 102, 6)
        self.text(x + 354, y + 107, "zone_cast boundary", 11.5, SOFT)
        self.badge(x + 520, y + 102, "1")
        self.text(x + 536, y + 107, "ordered flow", 11.5, SOFT)
        self.text(x + 660, y + 107,
                  "State placement: GPU HBM = model shards / activations / grads  ·  host DRAM = batch prefetch  ·  disk = ERA5 / checkpoint / forecast",
                  11.5, SOFT)

    def build(self) -> None:
        self.raw(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
                 f'width="100%" role="img" aria-labelledby="title desc">')
        self.raw('<title id="title">TornadoGCM end-to-end system flow</title>')
        self.raw('<desc id="desc">Distributed training and production inference around a shared '
                 'PZHA model step, with the SDA precision governance plane.</desc>')
        self.raw('<defs>')
        for mid, marker in [
            ("arrow", '<path d="M1,1 L11,6 L1,11 Z" fill="context-stroke"/>'),
            ("arrow-both", '<path d="M1,1 L11,6 L1,11 Z" fill="context-stroke"/>'),
        ]:
            both = ' orient="auto-start-reverse"' if mid == "arrow-both" else ""
            self.raw(
                f'<marker id="{mid}" viewBox="0 0 12 12" refX="11" refY="6" '
                f'markerWidth="8" markerHeight="8" markerUnits="strokeWidth"'
                f'{both}>{marker}</marker>'
            )
        self.raw('</defs>')
        self.raw(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
        self.header()
        self.lanes()

        # ---------------- TRAINING ----------------
        self.cylinder(56, 210, 200, 100, "ERA5 / WeatherBench", "Zarr · (T, L, lon, lat)", name="era5")
        self.text(156, 326, "disk", 10.5, SOFT, anchor="middle")
        self.box(320, 215, 150, 90, "DataLoader", "DistributedSampler\nbatches [B,T,L,…]", GRAY, GRAY_FILL, "loader")
        self.text(395, 322, "host DRAM", 10.5, SOFT, anchor="middle")
        self.rect(500, 185, 460, 240, "none", GRAY, 2.6, 8, name="gpu-node")
        self.text(516, 210, "GPU node · N ranks", 14, GRAY, weight="700")
        self.text(944, 210, "fault domain", 11, SOFT, anchor="end")
        self.gpu_card(520, 235, 125, 140, "rank 0")
        self.gpu_card(667, 235, 125, 140, "rank 1")
        self.gpu_card(814, 235, 125, 140, "rank r")
        self.edge([(645, 305), (667, 305)], "grad",
                  exclude=["gpurank 0", "gpurank 1", "gpu-node"])
        self.edge([(792, 305), (814, 305)], "grad",
                  exclude=["gpurank 1", "gpurank r", "gpu-node"])
        self.text(730, 396, "NCCL all-reduce · ring / tree · Σ gradients", 12, RED, anchor="middle")
        self.text(730, 416, "HBM: shards + activations + grads", 10.5, SOFT, anchor="middle")
        self.text(730, 230, "model shard + HBM", 11, SOFT, anchor="middle")

        self.box(1000, 220, 170, 90, "Trajectory loss", "MSE + physics\nconstraints", BLUE, BLUE_FILL, "loss")
        self.box(1200, 220, 190, 90, "Backward", "clip · FP32 EMA", GRAY, GRAY_FILL, "backward")
        self.box(1420, 220, 180, 90, "Optimizer", "Adam · EMA update", GRAY, GRAY_FILL, "optimizer")
        self.cylinder(1440, 340, 180, 70, "checkpoint .pt", "{model, optimizer, EMA}", name="ckpt")
        self.text(1530, 425, "disk", 10.5, SOFT, anchor="middle")
        self.box(1200, 345, 190, 65, "NaN / grad explosion", "detector", RED, RED_FILL, "nan", title_size=13, sub_size=11)

        self.edge([(256, 260), (320, 260)], "data", "batch", (288, 250), 11.5, ["era5", "loader"], "1", (288, 275))
        self.edge([(470, 260), (500, 260)], "data")
        self.edge([(1170, 265), (1200, 265)], "data")
        self.edge([(1390, 265), (1420, 265)], "data")
        self.edge([(1510, 310), (1530, 340)], "io", "save", (1560, 328), 11.5, ["optimizer", "ckpt"])
        self.edge([(1295, 310), (1295, 345)], "fail", "NaN?", (1322, 330), 11.5, ["backward", "nan"])
        self.edge([(1390, 377), (1440, 377)], "fail", "resume", (1415, 368), 11.5, ["nan", "ckpt"])
        self.edge([(1510, 220), (1510, 176), (730, 176), (730, 185)], "loop", "next step", (1120, 168), 12, ["optimizer", "gpu-node"], "4", (1180, 176))
        self.edge([(1295, 410), (1295, 435), (1645, 435), (1645, 514)],
                  "ctrl_gap", "escalate precision", (1500, 425), 12, ["nan"], "8", (1645, 470))
        self.edge([(1645, 526), (1645, 780), (1690, 780)], "ctrl")

        # forward / trajectory
        self.edge([(730, 425), (730, 470), (480, 470), (480, 620)], "data",
                  "forward K steps (one rank)", (610, 468), 12, ["gpu-node", "encoder"], "2", (730, 455))
        self.edge([(1330, 620), (1330, 460), (980, 460), (980, 265), (1000, 265)], "data",
                  "trajectory [B,K,…]", (1120, 468), 12, ["z2", "loss"], "4", (980, 300))

        # ---------------- MODEL STEP ----------------
        self.doc(56, 540, 200, 90, "JAX NeuralGCM", "checkpoint .pkl", BLUE, BLUE_FILL, "jax")
        self.text(156, 645, "disk", 10.5, SOFT, anchor="middle")
        self.box(300, 540, 170, 70, "load_jax_weights", "strict=False · transpose", GRAY, GRAY_FILL, "loader-weights")
        self.edge([(256, 585), (300, 585)], "io", "read", (278, 575), 11, ["jax", "loader-weights"])
        self.edge([(470, 575), (470, 515), (476, 515)], "io_gap",
                  exclude=["loader-weights"])
        self.edge([(486, 515), (840, 515), (840, 540)], "io",
                  "weights", (700, 506), 11.5, ["z3"])

        self.box(320, 620, 180, 100, "Encoder", "WeatherBench → modal\nState construction", BLUE, BLUE_FILL, "encoder")
        self.tensor_grid(540, 610, 150, 115, "state")
        self.text(615, 743, "State (L, m, l) · fp32", 12, BLUE, anchor="middle", weight="600")
        self.text(615, 760, "ζ, δ, T', log ps", 11.0, SOFT, anchor="middle")
        self.box(760, 540, 200, 110, "Z3 · Learned physics", "EPD 2365 → 384 → 192\nBF16 compute / params", GREEN, GREEN_FILL, "z3")
        self.box(760, 700, 200, 110, "Z1 · Dycore", "PrimitiveEquationsSigma\nIMEX-RK3-SIL3 ×5\nspectral filter · TF32", AMBER, AMBER_FILL, "z1")
        self.box(1020, 620, 180, 100, "ODE solver", "IMEX-RK3-SIL3\nimplicit + explicit", GRAY, GRAY_FILL, "imex")
        self.box(1240, 620, 180, 100, "Z2 · Fixer", "dry-air mass\ntotal energy · FP64", RED, RED_FILL, "z2")
        self.box(1460, 620, 150, 100, "Decoder", "modal → WB dict", BLUE, BLUE_FILL, "decoder")

        self.edge([(500, 670), (540, 670)], "data", "x(t)", (520, 660), 11.5, ["encoder", "state"], "", None)
        self.edge([(690, 655), (715, 655), (715, 595), (760, 595)], "data", "features", (735, 640), 11, ["state", "z3"])
        self.edge([(690, 685), (725, 685), (725, 755), (760, 755)], "data", "state", (742, 720), 11, ["state", "z1"])
        self.edge([(960, 595), (990, 595), (990, 650), (1020, 650)], "data", "tendency", (1005, 610), 11, ["z3", "imex"])
        self.edge([(960, 755), (1005, 755), (1005, 690), (1020, 690)], "data", "tendency", (1015, 730), 11, ["z1", "imex"])
        self.edge([(1200, 670), (1240, 670)], "data", "x(t+1)*", (1220, 660), 11.5, ["imex", "z2"])
        self.edge([(1420, 670), (1460, 670)], "data", "x(t+1)", (1440, 660), 11.5, ["z2", "decoder"])

        # zone-cast diamonds
        for x, y in [(520, 670), (740, 595), (990, 650), (1220, 670), (1440, 670)]:
            self.diamond(x, y)

        # inner loop and inference feed
        self.edge([(1535, 720), (1535, 830), (615, 830), (615, 725)], "loop",
                  "repeat inner_steps × outer_steps", (1075, 820), 12, ["decoder", "state"], "3", (960, 812))
        self.edge([(1535, 830), (1535, 860), (700, 860), (700, 980)], "data",
                  "model output → inference", (1120, 878), 11.5, ["decoder", "runner"], "6", (700, 920))

        # ---------------- INFERENCE ----------------
        self.cylinder(56, 990, 200, 90, "Initial condition", "encoded modal State", name="ic")
        self.box(320, 985, 200, 90, "MPInference", "SDAInference.run()\nouter_steps · no_grad", BLUE, BLUE_FILL, "engine")
        self.box(560, 985, 200, 90, "InferenceRunner", "chunked · NaN guard\natomic checkpoint", GRAY, GRAY_FILL, "runner")
        self.box(800, 985, 180, 90, "ZarrWriter", "numpy / async\nzarr output", GRAY, GRAY_FILL, "writer")
        self.cylinder(1020, 985, 180, 100, "forecast.zarr", "u, v, t, z + tracers", name="forecast")
        self.text(1110, 1102, "disk", 10.5, SOFT, anchor="middle")
        self.box(1240, 985, 180, 90, "EvaluationRunner", "RMSE · ACC · bias", GREEN, GREEN_FILL, "eval")
        self.box(1460, 985, 140, 90, "metrics", "Day-3/5/10", GRAY, GRAY_FILL, "metrics")
        self.box(320, 1120, 200, 60, "EnsembleRunner", "optional vmap / members", GRAY, GRAY_FILL, "ensemble", title_size=13, sub_size=10.5)
        self.box(800, 1120, 180, 60, "last checkpoint", "rollback target", RED, RED_FILL, "rollback", title_size=13, sub_size=10.5)

        self.edge([(256, 1035), (320, 1035)], "data", "x_0", (288, 1025), 11.5, ["ic", "engine"])
        self.edge([(520, 1035), (560, 1035)], "data")
        self.edge([(760, 1035), (800, 1035)], "data")
        self.edge([(980, 1035), (1020, 1035)], "io", "write", (1000, 1025), 11, ["writer", "forecast"])
        self.edge([(1200, 1035), (1240, 1035)], "data")
        self.edge([(1420, 1035), (1460, 1035)], "data")
        self.edge([(520, 1035), (520, 720), (500, 720)], "data",
                  "rollout ×N (no_grad)", (610, 890), 12, ["engine", "encoder"])
        self.edge([(660, 1075), (660, 1100), (890, 1100), (890, 1120)], "fail",
                  "NaN → rollback", (775, 1092), 11.5, ["runner", "rollback"])
        self.edge([(410, 1075), (410, 1120)], "loop", "optional", (430, 1102), 10.5, ["engine", "ensemble"])

        # checkpoint export bus (right gutter + inference header corridor)
        self.edge([(1620, 375), (1665, 375), (1665, 514)],
                  "io_gap", "export_inference_checkpoint (BF16)", (1120, 942), 12, ["ckpt"])
        self.edge([(1665, 526), (1665, 950), (540, 950), (540, 1035), (520, 1035)],
                  "io", exclude=["engine"])
        # checkpoint -> rollback tap
        self.edge([(1665, 1110), (890, 1110), (890, 1120)], "io", "last ckpt", (1240, 1102), 11, ["ckpt", "rollback"])

        # ---------------- GOVERNANCE ----------------
        self.doc(1690, 205, 290, 70, "SDA YAML", "policy · scheduler · storage", PURPLE, PURPLE_FILL, "sda-yaml")
        self.rect(1690, 300, 290, 255, "#FFFFFF", PURPLE, 1.8, 6, name="policy-stack")
        self.text(1702, 322, "PrecisionPolicy", 14, PURPLE, weight="700")
        self.text(1966, 322, "dtype / zone", 10.5, SOFT, anchor="end")
        rows = [
            (332, BLUE, BLUE_FILL, "Z0 · FP64", "bases / coords / implicit"),
            (386, AMBER, AMBER_FILL, "Z1 · FP32 / TF32", "SHT / dycore einsums"),
            (440, RED, RED_FILL, "Z2 · FP64", "mass / energy fixers"),
            (494, GREEN, GREEN_FILL, "Z3 · BF16", "EPD / towers"),
        ]
        for y, color, fill, title, sub in rows:
            self.rect(1702, y, 266, 48, fill, color, 1.4, 4)
            self.text(1714, y + 20, title, 12.5, color, weight="700")
            self.text(1714, y + 38, sub, 10.5, SOFT)

        self.box(1690, 575, 290, 60, "zone_cast", "f64_math · einsum_highest", PURPLE, PURPLE_FILL, "zone-cast", title_size=13.5, sub_size=10.5)
        self.box(1690, 655, 290, 60, "SDAProfiler", "grad · spectral · zone timing", PURPLE, PURPLE_FILL, "profiler", title_size=13.5, sub_size=10.5)
        self.box(1690, 735, 290, 80, "PrecisionScheduler", "milestone · loss_aware\nstability_aware", PURPLE, PURPLE_FILL, "scheduler", title_size=13.5, sub_size=10.5)
        self.box(1690, 835, 290, 55, "hot-swap policy", "re-apply TF32", PURPLE, PURPLE_FILL, "swap", title_size=13.5, sub_size=10.5)
        self.box(1690, 910, 290, 60, "ZoneDiscovery", "torch.export → FX → clusters", GRAY, GRAY_FILL, "discovery", title_size=13.5, sub_size=10.5)
        self.box(1690, 990, 290, 90, "Data-plane backends (optional)", "Storage exact / lossy · zstd\nSparse FP32 + BF16\nAccelerator compile / Triton", GRAY, GRAY_FILL, "backends", title_size=13.0, sub_size=10.5)
        self.rect(1690, 1100, 290, 90, "#FFFFFF", PURPLE, 1.2, 6, "6 4", name="report")
        self.text(1702, 1124, "SDAReport", 13, PURPLE, weight="600")
        self.text(1702, 1144, "logs · TensorBoard · zone audits", 11, SOFT)
        self.text(1702, 1162, "host policy applied at GPU zone boundaries", 10.5, SOFT)

        self.edge([(1835, 275), (1835, 300)], "ctrl", "load", (1850, 290), 11, ["sda-yaml", "policy-stack"])
        self.edge([(1835, 555), (1835, 575)], "ctrl")
        self.edge([(1835, 715), (1835, 735)], "ctrl")
        self.edge([(1835, 815), (1835, 835)], "ctrl")
        self.edge([(1980, 862), (1992, 862), (1992, 305), (1980, 305)], "ctrl", exclude=["swap", "policy-stack"])
        self.edge([(1835, 910), (1835, 890)], "ctrl", "discovered policy", (1852, 900), 10.5, ["discovery", "swap"])
        self.edge([(1690, 520), (1618, 520)], "ctrl", "policy → zones", (1652, 510), 11, ["policy-stack"])

        self.legend()
        self.raw("</svg>")

    # ---------- checks ----------
    def check(self) -> None:
        problems: list[str] = []
        # Node partial-overlap check.
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                ox = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
                oy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
                if ox > 1.0 and oy > 1.0:
                    a_contains = (a.x <= b.x and a.y <= b.y and
                                  a.x + a.w >= b.x + b.w and a.y + a.h >= b.y + b.h)
                    b_contains = (b.x <= a.x and b.y <= a.y and
                                  b.x + b.w >= a.x + a.w and b.y + b.h >= a.y + a.h)
                    if not (a_contains or b_contains):
                        problems.append(f"NODE OVERLAP: {a.name} <-> {b.name} ({ox:.0f}x{oy:.0f})")
        # Segment vs node check.
        def seg_rect(p, q, n: Node) -> bool:
            x1, y1 = p
            x2, y2 = q
            # Liang-Barsky against a slightly shrunken rectangle.
            pad = 2.5
            xmin, ymin = n.x + pad, n.y + pad
            xmax, ymax = n.x + n.w - pad, n.y + n.h - pad
            if xmax <= xmin or ymax <= ymin:
                return False
            dx, dy = x2 - x1, y2 - y1
            t0, t1 = 0.0, 1.0
            for pp, qq in [(-dx, x1 - xmin), (dx, xmax - x1),
                           (-dy, y1 - ymin), (dy, ymax - y1)]:
                if abs(pp) < 1e-9:
                    if qq < 0:
                        return False
                else:
                    t = qq / pp
                    if pp < 0:
                        t0 = max(t0, t)
                    else:
                        t1 = min(t1, t)
                    if t0 > t1:
                        return False
            return True
        for e in self.edges:
            for p, q in zip(e.points, e.points[1:]):
                for n in self.nodes:
                    if n.name in e.exclude:
                        continue
                    if seg_rect(p, q, n):
                        problems.append(
                            f"LINE THROUGH NODE: {e.kind} {p}->{q} hits {n.name}"
                        )
        if problems:
            print("\n".join(problems))
            print(f"\n{len(problems)} problem(s)")
        else:
            print("collision check: clean (nodes + line/box)")


def main() -> None:
    scene = Scene()
    scene.build()
    out = Path(__file__).resolve().parents[1] / "architecture.svg"
    out.write_text("\n".join(scene.parts), encoding="utf-8")
    print("wrote", out, out.stat().st_size, "bytes")
    scene.check()


if __name__ == "__main__":
    main()
