"""Diagnostic service layer.

The API layer only talks to this module, so replacing the placeholder colour
heuristic with the trained plant-disease model is a change in ``classifier.py``
(or pointing ``predict_leaf`` at real inference) rather than a change to every
endpoint.

``diagnose_image`` normalises whatever the vision layer returns into the small
contract the API depends on: a non-empty label and a confidence in the 0..1
range.
"""

from classifier import LowQualityImageError, predict_leaf

MODEL_VERSION = "colour-heuristic-placeholder-v1"

__all__ = [
    "DiagnosisUnavailableError",
    "InvalidPredictionError",
    "LowQualityImageError",
    "MODEL_VERSION",
    "diagnose_image",
]


class DiagnosisUnavailableError(RuntimeError):
    """Raised when no vision model has been integrated yet."""


class InvalidPredictionError(RuntimeError):
    """Raised when the vision layer returns output the API cannot trust."""


def diagnose_image(image_bytes):
    """Run the vision layer and return a normalised diagnosis.

    Args:
        image_bytes: Raw bytes of an already validated image upload.

    Returns:
        A dict with ``label``, ``confidence``, ``model_version`` and
        ``details``.

    Raises:
        LowQualityImageError: The image holds too little leaf tissue.
        DiagnosisUnavailableError: No model is integrated yet.
        InvalidPredictionError: The model output failed validation.
    """
    try:
        prediction = predict_leaf(image_bytes)
    except LowQualityImageError:
        raise
    except NotImplementedError as error:
        raise DiagnosisUnavailableError(
            "The computer vision model is not integrated yet."
        ) from error

    if not isinstance(prediction, dict):
        raise InvalidPredictionError("The classifier returned a malformed result.")

    label = prediction.get("label")
    if not isinstance(label, str) or not label.strip():
        raise InvalidPredictionError("The classifier returned an empty label.")

    try:
        confidence = float(prediction["confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidPredictionError(
            "The classifier returned an invalid confidence score."
        ) from error

    if not 0 <= confidence <= 1:
        raise InvalidPredictionError(
            "The classifier returned a confidence outside the 0..1 range."
        )

    return {
        "label": label.strip(),
        "confidence": confidence,
        "model_version": prediction.get("model_version", MODEL_VERSION),
        "details": prediction.get("details") or {},
    }
