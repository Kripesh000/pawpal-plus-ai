"""Deterministic SYNTHETIC test images for demos and automated tests.

These are procedurally drawn illustrations, not photographs of real stool. They exist so the
pipeline can be exercised end to end without distributing real animal-waste photos, and so the
evaluation harness has labelled inputs. Results on them say nothing about real-world accuracy.
"""

from __future__ import annotations

import io
import math
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# kind -> (shape, colour, texture) ; the dog fecal score each shape is *meant* to depict.
SCORE_KINDS: Dict[int, str] = {1: "pellets", 2: "firm_log", 3: "log", 4: "soft_log", 5: "piles", 6: "spots", 7: "puddle"}

COLORS: Dict[str, Tuple[int, int, int]] = {
    "brown": (98, 64, 40),
    "dark_brown": (62, 42, 30),
    "black": (24, 22, 22),
    "yellow": (178, 142, 62),
    "gray": (172, 166, 152),
}
BACKGROUNDS: Dict[str, Tuple[int, int, int]] = {
    "tile": (208, 203, 194),
    "grass": (72, 112, 52),
    "concrete": (142, 142, 148),
}


def _noise(rng: np.random.Generator, size: int, cell: int, amp: float) -> np.ndarray:
    grid = rng.normal(0, 1, (size // cell + 3, size // cell + 3)).astype(np.float32)
    img = Image.fromarray(grid, mode="F").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) * amp


def _blob(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, rng: np.random.Generator, lumpy: float) -> None:
    pts = []
    k = 28
    phases = rng.uniform(0, 2 * math.pi, 4)
    for i in range(k):
        a = 2 * math.pi * i / k
        wob = 1 + lumpy * (
            0.5 * math.sin(2 * a + phases[0]) + 0.35 * math.sin(3 * a + phases[1]) + 0.25 * math.sin(5 * a + phases[2])
        )
        pts.append((cx + r * wob * math.cos(a), cy + r * wob * math.sin(a)))
    draw.polygon(pts, fill=255)


def _log(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator, length: float, width: float,
         curve: float, wobble: float, offset: Tuple[float, float] = (0, 0)) -> None:
    angle = rng.uniform(-0.6, 0.6)
    ca, sa = math.cos(angle), math.sin(angle)
    cx, cy = size / 2 + offset[0], size / 2 + offset[1]
    phase = rng.uniform(0, 6.28)
    for i in range(41):
        t = i / 40 - 0.5
        x, y = t * length, curve * length * (t * t - 0.08)
        px, py = cx + x * ca - y * sa, cy + x * sa + y * ca
        w = width * (1 + wobble * math.sin(9 * t + phase)) * (1 - 0.35 * (2 * abs(t)) ** 4)
        draw.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=255)


def render_sample(kind: str, seed: int = 0, size: int = 320, background: str = "tile",
                  color: str = "brown", extra: str = "") -> Image.Image:
    """Render a synthetic image.

    kind: one of SCORE_KINDS values, or 'empty', 'blank', 'blue_object'.
    extra: '', 'red_streaks', 'mucus', 'foreign', 'blur', 'dark', 'overexposed'.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    bg = np.array(BACKGROUNDS[background], dtype=np.float32)
    canvas = np.tile(bg, (size, size, 1))
    canvas += _noise(rng, size, 3, 5.0)[..., None] + _noise(rng, size, 40, 6.0)[..., None]
    mask_img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask_img)
    crack = Image.new("L", (size, size), 0)
    cd = ImageDraw.Draw(crack)
    texture_amp, shine_n, edge_blur = 6.0, 0, 0.8

    if kind == "blank":
        return Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8))
    if kind == "blue_object":
        d.ellipse([size * 0.3, size * 0.3, size * 0.68, size * 0.68], fill=255)
        color_arr = np.array((40, 80, 200), dtype=np.float32)
    elif kind == "empty":
        color_arr = np.array(COLORS[color], dtype=np.float32)
    else:
        color_arr = np.array(COLORS[color], dtype=np.float32)
        if kind == "pellets":
            for _ in range(rng.integers(6, 9)):
                _blob(d, size / 2 + rng.uniform(-70, 70), size / 2 + rng.uniform(-50, 50), rng.uniform(11, 16), rng, 0.18)
            texture_amp, shine_n = 18.0, 0
        elif kind == "firm_log":
            _log(d, size, rng, 200, 34, rng.uniform(-0.2, 0.2), 0.04)
            texture_amp, shine_n = 15.0, 0
            for t in np.linspace(-0.4, 0.4, 5):  # segmentation cracks
                x = size / 2 + t * 200
                cd.line([(x, size / 2 - 40), (x + rng.uniform(-6, 6), size / 2 + 40)], fill=255, width=2)
        elif kind == "log":
            _log(d, size, rng, 205, 40, rng.uniform(-0.15, 0.15), 0.03)
            texture_amp, shine_n = 7.0, 3
        elif kind == "soft_log":
            _log(d, size, rng, 170, 50, rng.uniform(-0.15, 0.15), 0.10)
            texture_amp, shine_n, edge_blur = 5.0, 8, 1.6
        elif kind == "piles":
            _blob(d, size / 2 - 45, size / 2 + 10, 46, rng, 0.14)
            _blob(d, size / 2 + 50, size / 2 - 18, 40, rng, 0.14)
            texture_amp, shine_n, edge_blur = 5.0, 10, 1.8
        elif kind == "spots":
            cxs = size / 2
            _blob(d, cxs - 20, size / 2, 52, rng, 0.45)
            for _ in range(5):
                _blob(d, cxs + rng.uniform(-100, 100), size / 2 + rng.uniform(-80, 80), rng.uniform(10, 20), rng, 0.4)
            texture_amp, shine_n, edge_blur = 9.0, 6, 2.2
            color_arr = color_arr * 1.12
        elif kind == "puddle":
            _blob(d, size / 2, size / 2, 92, rng, 0.40)
            texture_amp, shine_n, edge_blur = 2.5, 14, 3.0
            color_arr = color_arr * 1.18
        else:
            raise ValueError(f"unknown kind {kind!r}")

    mask = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(edge_blur)), dtype=np.float32) / 255.0
    inner = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32) / 255.0
    layer = np.tile(color_arr, (size, size, 1)) * (0.72 + 0.4 * inner[..., None])
    layer += (_noise(rng, size, 2, texture_amp) + _noise(rng, size, 8, texture_amp * 0.6))[..., None]
    crack_m = np.asarray(crack.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0
    layer *= (1 - 0.5 * crack_m)[..., None]

    ys, xs = np.nonzero(np.asarray(mask_img) > 200)
    if len(ys) and kind not in ("empty",):
        # specular highlights (moisture)
        shine = Image.new("L", (size, size), 0)
        sd = ImageDraw.Draw(shine)
        for _ in range(shine_n):
            i = rng.integers(0, len(ys))
            sd.ellipse([xs[i] - 5, ys[i] - 2, xs[i] + 5, ys[i] + 2], fill=170)
        shine_m = np.asarray(shine.filter(ImageFilter.GaussianBlur(1.6)), dtype=np.float32) / 255.0
        layer = layer * (1 - shine_m[..., None]) + 250 * shine_m[..., None]

        if extra == "red_streaks":
            red = Image.new("L", (size, size), 0)
            rd = ImageDraw.Draw(red)
            for i in range(4):
                j = rng.integers(0, len(ys))
                rd.line([(xs[j] - 14, ys[j] - 6 + i), (xs[j] + 14, ys[j] + 6 - i)], fill=255, width=3)
            red_m = np.asarray(red.filter(ImageFilter.GaussianBlur(0.9)), dtype=np.float32) / 255.0 * mask
            layer = layer * (1 - red_m[..., None]) + np.array([185, 20, 22], dtype=np.float32) * red_m[..., None]
        if extra == "mucus":
            pale = Image.new("L", (size, size), 0)
            pd = ImageDraw.Draw(pale)
            for _ in range(5):
                j = rng.integers(0, len(ys))
                pd.ellipse([xs[j] - 12, ys[j] - 7, xs[j] + 12, ys[j] + 7], fill=255)
            pale_m = np.asarray(pale.filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32) / 255.0 * mask
            layer = layer * (1 - 0.85 * pale_m[..., None]) + np.array([238, 234, 220], dtype=np.float32) * 0.85 * pale_m[..., None]
        if extra == "foreign":
            for hue in ((40, 90, 210), (30, 170, 60), (220, 60, 150)):
                j = rng.integers(0, len(ys))
                fm = Image.new("L", (size, size), 0)
                ImageDraw.Draw(fm).ellipse([xs[j] - 5, ys[j] - 5, xs[j] + 5, ys[j] + 5], fill=255)
                fm_a = np.asarray(fm.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255.0
                layer = layer * (1 - fm_a[..., None]) + np.array(hue, dtype=np.float32) * fm_a[..., None]

    out = canvas * (1 - mask[..., None]) + layer * mask[..., None]
    out *= rng.uniform(0.94, 1.06)
    out += rng.normal(0, 2.5, out.shape).astype(np.float32)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if extra == "blur":
        img = img.filter(ImageFilter.GaussianBlur(9))
    elif extra == "dark":
        img = Image.fromarray((np.asarray(img, dtype=np.float32) * 0.16).astype(np.uint8))
    elif extra == "overexposed":
        img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * 0.6 + 170, 0, 255).astype(np.uint8))
    return img


def to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# name -> render kwargs for the committed demo images.
DEMO_CATALOG: Dict[str, dict] = {
    "normal_firm_log": dict(kind="firm_log", seed=101),
    "normal_log": dict(kind="log", seed=102, background="concrete"),
    "very_hard_pellets": dict(kind="pellets", seed=103),
    "loose_piles": dict(kind="piles", seed=104),
    "loose_spots": dict(kind="spots", seed=105, color="yellow"),
    "watery_puddle": dict(kind="puddle", seed=106, background="concrete"),
    "red_streaks": dict(kind="firm_log", seed=107, extra="red_streaks"),
    "black_tarry": dict(kind="log", seed=108, color="black"),
    "pale_gray": dict(kind="log", seed=109, color="gray", background="concrete"),
    "mucus_coating": dict(kind="log", seed=110, extra="mucus"),
    "foreign_inclusions": dict(kind="firm_log", seed=111, extra="foreign"),
    "blurry_photo": dict(kind="log", seed=112, extra="blur"),
    "dark_photo": dict(kind="log", seed=113, extra="dark"),
    "no_stool_blue_object": dict(kind="blue_object", seed=114),
    "empty_ground": dict(kind="empty", seed=115),
}


def write_demo_images(out_dir: str) -> Dict[str, str]:
    """Write the demo catalogue as PNG files plus two invalid inputs; return name -> path."""
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, kwargs in DEMO_CATALOG.items():
        path = os.path.join(out_dir, f"{name}.png")
        render_sample(**kwargs).save(path)
        paths[name] = path
    bad = os.path.join(out_dir, "invalid_not_an_image.jpg")
    with open(bad, "wb") as handle:
        handle.write(b"this is plain text pretending to be a jpeg")
    paths["invalid_not_an_image"] = bad
    return paths


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images")
    for key, value in write_demo_images(here).items():
        print(f"{key:24s} {value}")
