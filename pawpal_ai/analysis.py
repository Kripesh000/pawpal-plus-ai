"""Visual analysis: measure image features and turn them into structured observations.

HeuristicVisionBackend is classical image processing (colour segmentation, texture and shape
statistics). It is NOT a trained computer-vision model, and its outputs are labelled as
heuristic estimates everywhere they are shown.
"""

from __future__ import annotations

import colorsys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .imaging import SubjectMask, label_components
from .models import Finding, ImageQualityReport, VisualFeatures

METHOD_NAME = "heuristic-image-features-v1"


def _hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB(0-255) -> (hue_deg, sat, val)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    delta = mx - mn
    hue = np.zeros_like(mx)
    nz = delta > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60 * ((g - b)[rm] / delta[rm])) % 360
    hue[gm] = 60 * ((b - r)[gm] / delta[gm]) + 120
    hue[bm] = 60 * ((r - g)[bm] / delta[bm]) + 240
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0.0)
    return hue, sat, mx


def _erode(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8)) * 255).filter(ImageFilter.MinFilter(size))
    return np.asarray(img) > 127


class HeuristicVisionBackend:
    """Measure colour, texture, shape and special-feature statistics inside the subject mask."""

    name = METHOD_NAME

    def extract(self, subject: SubjectMask) -> VisualFeatures:
        """Compute VisualFeatures; assumes `subject` came from a quality-approved image."""
        rgb, mask = subject.rgb, subject.mask
        comps = subject.components
        feats = VisualFeatures(coverage=round(subject.fraction, 4))
        if not comps:
            return feats
        largest = comps[0]
        feats.n_pieces = sum(1 for c in comps if c.area >= 0.1 * largest.area)
        feats.elongation = round(largest.elongation, 3)
        feats.compactness = round(largest.compactness, 3)
        feats.edge_roughness = round(largest.edge_roughness, 3)
        feats.segment_dips = largest.segment_dips

        core = _erode(mask, 5 if largest.area > 500 else 3)
        if core.sum() < 20:
            core = mask
        pix = rgb[core]
        feats.mean_rgb = [round(float(v), 1) for v in pix.mean(axis=0)]
        hue, sat, val = _hsv(pix)
        n = len(pix)

        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        blurred = np.asarray(
            Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32
        )
        feats.texture = round(float((gray - blurred)[core].std()), 2)

        # Colour class fractions (mutually exclusive, evaluated in priority order).
        dark = val < 0.16
        red = ~dark & ((hue < 14) | (hue > 340)) & (sat > 0.55) & (pix[:, 0] > 1.5 * pix[:, 1])
        pale = ~dark & ~red & (sat < 0.20) & (val > 0.55)
        yellow = ~dark & ~red & ~pale & (hue >= 38) & (hue <= 70) & (sat > 0.45) & (val > 0.5)
        green = ~dark & ~red & ~pale & ~yellow & (hue > 70) & (hue < 170) & (sat > 0.25)
        brown = ~(dark | red | pale | yellow | green)
        feats.color_fractions = {
            "brown": round(float(brown.mean()), 4),
            "dark": round(float(dark.mean()), 4),
            "red": round(float(red.mean()), 4),
            "yellow": round(float(yellow.mean()), 4),
            "green": round(float(green.mean()), 4),
            "pale": round(float(pale.mean()), 4),
        }

        # Specular highlights (moisture) and large pale glossy patches (mucus-like coating).
        median_v = float(np.median(val))
        bright = (val > min(0.95, median_v + 0.22)) & (sat < 0.4)
        full_pix_mask = np.zeros(mask.shape, dtype=bool)
        full_pix_mask[core] = bright
        feats.shine = round(float(bright.mean()), 4)
        if feats.color_fractions["pale"] < 0.5 and bright.any():
            labels, count = label_components(full_pix_mask)
            big = 0
            for idx in range(1, count + 1):
                area = int((labels == idx).sum())
                if area >= max(30, int(0.012 * core.sum())):
                    big += area
            feats.pale_glossy_fraction = round(big / n, 4)

        # Inclusions: pixels with hues that stool/blood/bile do not produce.
        odd = (sat > 0.45) & (val > 0.3) & (((hue >= 170) & (hue <= 340)) | ((hue > 85) & (hue < 160) & (sat > 0.5)))
        feats.inclusion_fraction = round(float(odd.mean()), 4)
        return feats

    @staticmethod
    def describe_color(feats: VisualFeatures) -> Tuple[str, float]:
        """Return (dominant colour label, confidence 0..1) from the colour fractions."""
        fr = feats.color_fractions
        if not fr:
            return "unknown", 0.0
        main = max(("brown", "dark", "yellow", "green", "pale"), key=lambda k: fr.get(k, 0.0))
        share = fr[main]
        r, g, b = feats.mean_rgb
        _h, _s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if main == "brown":
            label = "dark brown" if v < 0.30 else "brown"
        else:
            label = {
                "dark": "very dark / black",
                "yellow": "yellow-orange",
                "green": "greenish",
                "pale": "pale gray / whitish",
            }[main]
        return label, float(min(0.9, max(0.3, share)))


class StoolAnalyzer:
    """Turns raw features into the structured list of visual observations."""

    def __init__(self, backend: HeuristicVisionBackend | None = None) -> None:
        self.backend = backend or HeuristicVisionBackend()

    def analyze(self, subject: SubjectMask, quality: ImageQualityReport) -> Tuple[VisualFeatures, List[Finding]]:
        """Return (features, findings). Findings describe what is visible, not what it means."""
        feats = self.backend.extract(subject)
        q = quality.score
        dim = "dim_lighting_colour_unreliable" in quality.warnings
        fr = feats.color_fractions
        findings: List[Finding] = []

        color_label, color_conf = self.backend.describe_color(feats)
        color_conf *= q * (0.8 if dim else 1.0)
        findings.append(Finding("color", "observed", round(color_conf, 2), color_label))

        shape = self._shape_label(feats)
        findings.append(Finding("shape", "observed", round(0.6 * q, 2), shape))
        texture = self._texture_label(feats)
        findings.append(Finding("surface_texture", "observed", round(0.55 * q, 2), texture))

        def flag(name: str, value: float, lo: float, hi: float, present: str, absent: str, cap: float = 0.8) -> None:
            if value >= lo:
                conf = min(cap, 0.35 + (value - lo) / max(hi - lo, 1e-6) * (cap - 0.35)) * q * (0.85 if dim else 1.0)
                findings.append(Finding(name, "possible", round(conf, 2), present))
            else:
                findings.append(Finding(name, "not_detected", round(0.5 * q, 2), absent))

        flag("red_discoloration", fr.get("red", 0.0), 0.008, 0.06,
             "Reddish areas are visible. Colour alone cannot show whether this is blood; dyes in food or treats can look similar.",
             "No clearly red areas detected (this does not rule out blood, e.g. microscopic).")
        flag("dark_or_black_stool", fr.get("dark", 0.0), 0.55, 0.9,
             "Very dark / black stool is visible. Some diets and supplements darken stool; a photo cannot show whether it is digested blood.",
             "Stool is not very dark.")
        flag("pale_or_gray_stool", fr.get("pale", 0.0), 0.5, 0.85,
             "Pale gray or whitish stool is visible.", "Stool is not unusually pale.")
        flag("yellow_orange_stool", fr.get("yellow", 0.0), 0.5, 0.85,
             "Yellow-orange colouring is visible.", "No strong yellow-orange colouring detected.")
        flag("mucus_like_coating", feats.pale_glossy_fraction, 0.03, 0.15,
             "Pale glossy patches that could be mucus are visible. Reflections and food residue can look similar.",
             "No obvious pale glossy coating detected.", cap=0.65)
        flag("foreign_material", feats.inclusion_fraction, 0.003, 0.03,
             "Small areas of colour not typical of stool are visible; they could be foreign material (for example plastic or fabric) or a lighting artefact.",
             "No obviously non-stool-coloured inclusions detected.", cap=0.6)
        return feats, findings

    @staticmethod
    def _shape_label(f: VisualFeatures) -> str:
        if f.n_pieces >= 4 and f.elongation < 0.7:
            return f"{f.n_pieces} small separate pieces (pellet-like)"
        if f.elongation >= 0.72 and f.n_pieces <= 2:
            return "elongated log-like piece" + ("s" if f.n_pieces == 2 else "")
        if f.compactness < 0.8:
            return "irregular spreading shape with no defined form"
        return "rounded pile-like mass" + ("es" if f.n_pieces >= 2 else "")

    @staticmethod
    def _texture_label(f: VisualFeatures) -> str:
        if f.texture >= 11:
            return "rough, dry-looking surface"
        if f.shine >= 0.012:
            return "smooth with glossy / moist highlights"
        return "smooth matte surface"
