"""Leaf image classifier.

This module exposes a single entry point, :func:`predict_leaf`, which the API
calls with the raw bytes of an uploaded image.

The current implementation is a deterministic colour heuristic, not a trained
model. It separates green leaf tissue from brown/necrotic tissue and uses the
share of affected tissue to choose between the labels currently described in
``treatments.json``. It exists so the API can be exercised end to end.

TODO: replace ``predict_leaf`` with inference against the trained
plant-disease model once its weights and label mapping are available. The
return contract must stay the same: a label from ``treatments.json`` plus a
confidence in the 0..1 range.
"""

import io

from PIL import Image, UnidentifiedImageError

# Images are downscaled before analysis so classification stays fast.
ANALYSIS_SIZE = (96, 96)

# Pixel thresholds, expressed on the 0-255 RGB scale.
DARK_CUTOFF = 45
GREEN_CUTOFF = 55
BROWN_MIN_RED = 60
BROWN_RED_GREEN_GAP = 15

# Minimum number of plant pixels needed for a meaningful result.
MIN_TISSUE_PIXELS = 25

# Share of affected leaf tissue above which early blight is reported.
BLIGHT_THRESHOLD = 0.15

HEALTHY_LABEL = "Tomato___healthy"
EARLY_BLIGHT_LABEL = "Tomato___Early_blight"


class LowQualityImageError(ValueError):
    """Raised when an image holds too little leaf tissue to classify."""


def _is_leaf_pixel(red, green, blue):
    if max(red, green, blue) < DARK_CUTOFF:
        return False

    return green > red and green > blue and green > GREEN_CUTOFF


def _is_lesion_pixel(red, green, blue):
    return (
        red > BROWN_MIN_RED
        and red > green >= blue
        and red - green > BROWN_RED_GREEN_GAP
    )


def _count_tissue(pixels):
    leaf_pixels = 0
    lesion_pixels = 0

    for red, green, blue in pixels:
        if _is_leaf_pixel(red, green, blue):
            leaf_pixels += 1
        elif _is_lesion_pixel(red, green, blue):
            lesion_pixels += 1

    return leaf_pixels, lesion_pixels


def _load_pixels(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            rgb_image.thumbnail(ANALYSIS_SIZE)
            return list(rgb_image.getdata())
    except (UnidentifiedImageError, OSError) as error:
        raise LowQualityImageError(
            "The uploaded file is not a readable image."
        ) from error


def predict_leaf(image_bytes):
    """Return a placeholder diagnosis for the supplied leaf image.

    Args:
        image_bytes: Raw bytes of a JPEG, PNG or WebP image.

    Returns:
        A dict with ``label`` (matching a key in ``treatments.json``),
        ``confidence`` between 0 and 1, and ``details`` describing the
        heuristic evidence.

    Raises:
        LowQualityImageError: If the image is unreadable or contains too
            little leaf tissue to analyse.
    """
    if not image_bytes:
        raise LowQualityImageError("The uploaded image is empty.")

    pixels = _load_pixels(image_bytes)
    leaf_pixels, lesion_pixels = _count_tissue(pixels)
    tissue_pixels = leaf_pixels + lesion_pixels

    if tissue_pixels < MIN_TISSUE_PIXELS:
        raise LowQualityImageError(
            "The image does not appear to contain enough leaf tissue."
        )

    blight_score = lesion_pixels / tissue_pixels

    if blight_score >= BLIGHT_THRESHOLD:
        label = EARLY_BLIGHT_LABEL
        confidence = 0.55 + blight_score
    else:
        label = HEALTHY_LABEL
        confidence = 0.55 + (BLIGHT_THRESHOLD - blight_score) * 2

    return {
        "label": label,
        "confidence": round(min(confidence, 0.95), 4),
        "details": {
            "method": "colour-heuristic-placeholder",
            "leaf_pixels": leaf_pixels,
            "lesion_pixels": lesion_pixels,
            "blight_score": round(blight_score, 4),
        },
    }
