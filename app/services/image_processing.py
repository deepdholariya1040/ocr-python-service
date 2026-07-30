"""
=============================================================================
Image preprocessing
=============================================================================
Pure-Pillow + numpy preprocessing to maximize PaddleOCR accuracy without
pulling in a heavy computer-vision dependency (no OpenCV - numpy is already
a transitive dependency of paddlepaddle, so it costs nothing extra).

Pipeline, in order:

  1. Normalize orientation via EXIF.
  2. Upscale small images / downscale huge ones to a sane OCR range.
  3. Light denoise (median filter - removes salt-and-pepper / JPEG-block
     noise from a phone photo without smearing text edges).
  4. Grayscale.
  5. Autocontrast (stretches the histogram, cheap and safe).
  6. Light sharpening (unsharp mask).
  7. Quality validation (blur + contrast heuristics) - logged, never
     blocking; a low score just tells the caller the image was poor
     quality, it does not fail the request.

Deliberately conservative on one point: this does NOT hard-binarize
(adaptive-threshold) the final image handed to the OCR engine. Adaptive
thresholding is the right move for classical engines like Tesseract, which
work directly off binary pixel maps - but PaddleOCR's detector/recognizer
are learned models trained on natural photos, and hard black/white
binarization measurably *hurts* their accuracy by discarding the grayscale
gradients they rely on. A blur/contrast quality check runs instead (see `assess_image_quality`)
purely for diagnostics - it never alters what PaddleOCR actually sees.
=============================================================================
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.config import Settings

_UNSHARP_MASK = ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
_DENOISE_FILTER = ImageFilter.MedianFilter(size=3)


def preprocess_for_ocr(image: Image.Image, settings: Settings) -> Image.Image:
    working = ImageOps.exif_transpose(image) or image

    working = _resize_within_bounds(working, settings)

    grayscale = ImageOps.grayscale(working)
    denoised = grayscale.filter(_DENOISE_FILTER)
    contrasted = ImageOps.autocontrast(denoised, cutoff=1)
    sharpened = contrasted.filter(_UNSHARP_MASK)

    return sharpened


def assess_image_quality(image: Image.Image) -> Dict[str, object]:
    """
    Cheap, non-blocking heuristics used purely for diagnostics/logging
    (surfaced in the response `meta`). Never raises, never affects the
    pipeline's control flow - a "poor" score does not stop OCR from
    running, it just helps explain a bad result after the fact.
    """
    try:
        grayscale = ImageOps.grayscale(ImageOps.exif_transpose(image) or image)
        arr = np.asarray(grayscale, dtype=np.float32)

        # Blur estimate: variance of a simple Laplacian-like edge response.
        # Low variance -> few sharp edges -> likely blurry/out-of-focus.
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.asarray(edges, dtype=np.float32)
        blur_score = float(edge_arr.var())

        # Contrast estimate: standard deviation of pixel intensities.
        contrast_score = float(arr.std())

        return {
            "blurScore": round(blur_score, 2),
            "contrastScore": round(contrast_score, 2),
            "likelyBlurry": blur_score < 80.0,
            "likelyLowContrast": contrast_score < 25.0,
        }
    except Exception:  # noqa: BLE001 - diagnostics must never break the pipeline
        return {}


def _resize_within_bounds(image: Image.Image, settings: Settings) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)

    if longest_side > settings.OCR_MAX_IMAGE_DIMENSION:
        scale = settings.OCR_MAX_IMAGE_DIMENSION / longest_side
    elif longest_side < settings.OCR_MIN_IMAGE_DIMENSION:
        scale = settings.OCR_MIN_IMAGE_DIMENSION / longest_side
    else:
        return image

    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return image.resize(new_size, resample)
