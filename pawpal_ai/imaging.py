"""Image loading, quality validation and subject (foreground) segmentation.

Everything here works on raw bytes, so it does not care whether the picture came from a
file upload, a phone camera, a smart litter box or a monitoring station.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from .models import ImageInput, ImageQualityReport

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_SIDE = 96
WORK_SIDE = 192  # working resolution for segmentation / features


class ImageInputError(ValueError):
    """Raised for unreadable, unsupported or unsafe input. `code` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class LoadedImage:
    """A decoded RGB image (max 512 px on the long side) plus provenance."""

    rgb: np.ndarray  # uint8, HxWx3
    original_size: Tuple[int, int]
    fmt: str
    name: str
    sha256: str
    source_type: str


class ImageProcessor:
    """Decode, validate and normalise image bytes into a LoadedImage."""

    def __init__(self, max_side: int = 512) -> None:
        self.max_side = max_side

    def load(self, image_input: ImageInput) -> LoadedImage:
        """Decode `image_input`; raise ImageInputError with a clear code on any problem."""
        data = image_input.data
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise ImageInputError("empty", "The image file is empty.")
        if len(data) > MAX_BYTES:
            raise ImageInputError("too_large", f"Image exceeds {MAX_BYTES // (1024 * 1024)} MB.")
        try:
            probe = Image.open(io.BytesIO(data))
            fmt = (probe.format or "").upper()
            width, height = probe.size
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise ImageInputError(
                "unreadable", "File is not a readable image (corrupt or not an image)."
            ) from None
        if fmt not in SUPPORTED_FORMATS:
            raise ImageInputError(
                "unsupported_format", f"Format {fmt or 'unknown'} is not supported; use {SUPPORTED_FORMATS}."
            )
        if width * height > MAX_PIXELS:
            raise ImageInputError("too_many_pixels", "Image dimensions are unreasonably large.")
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                backdrop = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(backdrop, img)
            img = img.convert("RGB")
        except (OSError, ValueError, SyntaxError):
            raise ImageInputError("unreadable", "Image data is truncated or corrupt.") from None
        if min(img.size) < MIN_SIDE:
            raise ImageInputError(
                "too_small", f"Image is {img.size[0]}x{img.size[1]}; at least {MIN_SIDE}px per side is needed."
            )
        original = img.size
        if max(img.size) > self.max_side:
            scale = self.max_side / max(img.size)
            img = img.resize((max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))), Image.LANCZOS)
        return LoadedImage(
            rgb=np.asarray(img, dtype=np.uint8),
            original_size=original,
            fmt=fmt,
            name=image_input.name,
            sha256=hashlib.sha256(bytes(data)).hexdigest(),
            source_type=image_input.source_type,
        )


# --------------------------------------------------------------------------- #
# Subject segmentation
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    """A connected foreground region."""

    area: int
    bbox: Tuple[int, int, int, int]  # y0, x0, y1, x1
    elongation: float  # 0 = round, 1 = line
    compactness: float  # area / fitted-ellipse area (1 = ellipse-like)
    touches_border: bool
    edge_roughness: float = 1.0  # outline length / outline of the fitted ellipse (1 = smooth)
    segment_dips: int = 0  # dark transverse bands along the long axis (segmentation / cracks)


@dataclass
class SubjectMask:
    """Foreground mask at working resolution plus summary measurements."""

    rgb: np.ndarray  # working-resolution RGB (uint8)
    mask: np.ndarray  # bool
    fraction: float
    components: List[Component]
    background_rgb: Tuple[float, float, float]
    background_std: float
    mean_hue_deg: float
    mean_saturation: float
    plausible_color: bool


def _resize_to(rgb: np.ndarray, side: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    scale = side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected component labelling (iterative flood fill; fine at working resolution)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    current = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] and labels[y, x] == 0:
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def _fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background-coloured gaps fully enclosed by the subject (e.g. pale glossy patches)."""
    inverse = ~mask
    labels, count = label_components(inverse)
    if count == 0:
        return mask
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    filled = mask.copy()
    limit = 0.15 * mask.size
    for idx in range(1, count + 1):
        if idx in edge:
            continue
        region = labels == idx
        if region.sum() < limit:
            filled |= region
    return filled


def _edge_roughness(comp_mask: np.ndarray, eig: np.ndarray) -> float:
    """Outline length relative to the outline of the ellipse with the same second moments."""
    pad = np.pad(comp_mask, 1)
    inner = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    perimeter = float((comp_mask & ~inner).sum())
    a, b = 2.0 * np.sqrt(eig[1]), 2.0 * np.sqrt(eig[0])
    ellipse = float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))
    return float(perimeter / ellipse) if ellipse > 0 else 1.0


def _segment_dips(ys: np.ndarray, xs: np.ndarray, axis: np.ndarray, gray: np.ndarray, area: int) -> int:
    """Count narrow dark bands across a component's long axis (log segmentation / cracks)."""
    if area < 150:
        return 0
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    t = (coords - coords.mean(axis=0)) @ axis
    bins = np.round(t - t.min()).astype(int)
    n_bins = int(bins.max()) + 1
    if n_bins < 30:
        return 0
    sums = np.bincount(bins, weights=gray[ys, xs], minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    profile = profile[3:-3]  # ignore the rounded ends
    valid = ~np.isnan(profile)
    if valid.sum() < 20:
        return 0
    profile = np.where(valid, profile, np.nanmean(profile))
    kernel = np.ones(13) / 13.0
    trend = np.convolve(np.pad(profile, 6, mode="edge"), kernel, mode="valid")
    detail = profile - trend
    thresh = max(8.0, 2.5 * float(np.std(detail)))
    dips, last = 0, -10
    for i in range(1, len(detail) - 1):
        if detail[i] < -thresh and detail[i] <= detail[i - 1] and detail[i] <= detail[i + 1] and i - last > 4:
            dips += 1
            last = i
    return dips


def segment_subject(rgb: np.ndarray) -> SubjectMask:
    """Separate a foreground subject from the background using border-colour contrast.

    Limitation (by design, documented): it assumes the subject differs in colour from the
    surface around it and does not fill the entire frame. It is not a trained detector.
    """
    work = _resize_to(rgb, WORK_SIDE)
    height, width, _ = work.shape
    pix = work.astype(np.float32)
    band = max(3, int(0.07 * min(height, width)))
    border = np.concatenate(
        [pix[:band].reshape(-1, 3), pix[-band:].reshape(-1, 3), pix[:, :band].reshape(-1, 3), pix[:, -band:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    bg_std = float(np.mean(np.std(border, axis=0)))
    dist = np.sqrt(((pix - bg) ** 2).sum(axis=2))
    threshold = max(38.0, 3.0 * bg_std)
    raw = (dist > threshold).astype(np.uint8) * 255
    cleaned = Image.fromarray(raw).filter(ImageFilter.MedianFilter(5))
    mask = _fill_interior_holes(np.asarray(cleaned) > 127)

    labels, count = label_components(mask)
    min_area = max(12, int(0.0015 * height * width))
    keep = np.zeros_like(mask)
    components: List[Component] = []
    gray_work = (0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]).astype(np.float32)
    for idx in range(1, count + 1):
        comp_mask = labels == idx
        ys, xs = np.nonzero(comp_mask)
        area = len(ys)
        if area < min_area:
            continue
        keep[ys, xs] = True
        cov = np.cov(np.vstack([ys, xs]).astype(np.float64)) if area > 2 else np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-6, None)
        eig = np.sort(vals)
        elong = float(1.0 - np.sqrt(eig[0] / eig[1]))
        ellipse_area = float(np.pi * 4.0 * np.sqrt(eig[0] * eig[1]))
        compact = float(min(area / ellipse_area, 1.5)) if ellipse_area > 0 else 0.0
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        touches = y0 == 0 or x0 == 0 or y1 == height - 1 or x1 == width - 1
        components.append(
            Component(
                area, (y0, x0, y1, x1), elong, compact, touches,
                _edge_roughness(comp_mask, eig), _segment_dips(ys, xs, vecs[:, int(np.argmax(vals))], gray_work, area),
            )
        )
    components.sort(key=lambda c: c.area, reverse=True)

    fraction = float(keep.mean())
    if keep.any():
        mean_rgb = pix[keep].mean(axis=0) / 255.0
        h, s, _v = colorsys.rgb_to_hsv(*[float(c) for c in mean_rgb])
        hue_deg = h * 360.0
    else:
        hue_deg, s = 0.0, 0.0
    # Subjects that are clearly saturated blue/cyan/purple are not stool-coloured.
    plausible = not (s > 0.25 and 165.0 <= hue_deg <= 300.0)
    return SubjectMask(
        rgb=work,
        mask=keep,
        fraction=fraction,
        components=components,
        background_rgb=(float(bg[0]), float(bg[1]), float(bg[2])),
        background_std=bg_std,
        mean_hue_deg=hue_deg,
        mean_saturation=float(s),
        plausible_color=plausible,
    )


# --------------------------------------------------------------------------- #
# Quality validation
# --------------------------------------------------------------------------- #
def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian: a standard focus measure (low = blurry)."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


class ImageQualityValidator:
    """Checks focus, lighting, resolution and whether a stool-coloured subject is present.

    The subject check is colour/contrast based: it cannot tell stool from other brown objects.
    """

    def __init__(
        self,
        min_brightness: float = 45.0,
        max_brightness: float = 225.0,
        min_sharpness: float = 14.0,
        min_contrast: float = 5.0,
        min_subject_fraction: float = 0.02,
        max_subject_fraction: float = 0.85,
        soft_min_side: int = 224,
    ) -> None:
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.min_contrast = min_contrast
        self.min_subject_fraction = min_subject_fraction
        self.max_subject_fraction = max_subject_fraction
        self.soft_min_side = soft_min_side

    def assess(self, image: LoadedImage) -> Tuple[ImageQualityReport, SubjectMask]:
        """Return the quality report and the segmentation (reused by the analyzer)."""
        rgb = image.rgb
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        dark_clip = float((gray < 12).mean())
        bright_clip = float((gray > 243).mean())
        sharp_gray = _resize_to(np.stack([gray] * 3, axis=-1).astype(np.uint8), 256)[..., 0]
        sharpness = laplacian_variance(sharp_gray)
        subject = segment_subject(rgb)

        issues: List[str] = []
        warnings: List[str] = []
        if brightness < self.min_brightness:
            issues.append("too_dark")
        elif brightness > self.max_brightness or bright_clip > 0.5:
            issues.append("overexposed")
        if contrast < self.min_contrast:
            issues.append("flat_or_blank")
        if sharpness < self.min_sharpness:
            issues.append("out_of_focus")
        if subject.fraction < self.min_subject_fraction:
            issues.append("no_subject_found")
        elif subject.fraction > self.max_subject_fraction:
            issues.append("subject_fills_frame")
        elif not subject.plausible_color:
            issues.append("subject_not_stool_coloured")

        if min(image.original_size) < self.soft_min_side:
            warnings.append("low_resolution")
        if dark_clip > 0.25 and "too_dark" not in issues:
            warnings.append("heavy_shadows")
        if subject.components and subject.components[0].touches_border:
            warnings.append("subject_cut_off_at_edge")
        if brightness < self.min_brightness + 25 and "too_dark" not in issues:
            warnings.append("dim_lighting_colour_unreliable")
        if sharpness < self.min_sharpness * 1.8 and "out_of_focus" not in issues:
            warnings.append("slightly_soft_focus")

        subject_visible = not any(
            i in issues for i in ("no_subject_found", "subject_fills_frame", "subject_not_stool_coloured", "flat_or_blank")
        )
        subject_conf = 0.0
        if subject_visible:
            subject_conf = float(np.clip(1.0 - abs(subject.fraction - 0.22) / 0.6, 0.35, 0.95))

        score = 1.0 - 0.28 * len(issues) - 0.08 * len(warnings)
        score = float(np.clip(score, 0.0, 1.0))
        usable = not issues
        recommendation = (
            "Photo is usable."
            if usable and not warnings
            else "Photo is usable but see warnings; retake in bright, even light if possible."
            if usable
            else "Please retake: " + _retake_advice(issues)
        )
        report = ImageQualityReport(
            usable=usable,
            score=round(score, 3),
            issues=issues,
            warnings=warnings,
            metrics={
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "sharpness": round(sharpness, 1),
                "subject_fraction": round(subject.fraction, 3),
                "dark_clip": round(dark_clip, 3),
                "bright_clip": round(bright_clip, 3),
            },
            subject_visible=subject_visible,
            subject_confidence=round(subject_conf, 3),
            recommendation=recommendation,
        )
        return report, subject


_ADVICE = {
    "too_dark": "use more light (avoid heavy shadow)",
    "overexposed": "reduce glare or direct flash",
    "flat_or_blank": "the image has almost no detail",
    "out_of_focus": "hold steady and tap to focus on the sample",
    "no_subject_found": "frame the sample clearly against a plain contrasting surface",
    "subject_fills_frame": "step back so the sample and some surrounding surface are both visible",
    "subject_not_stool_coloured": "no stool-coloured subject was found; check what the photo shows",
}


def _retake_advice(issues: List[str]) -> str:
    return "; ".join(_ADVICE.get(i, i) for i in issues) + "."
