"""Upload validation, shared by the Flask site and the model API."""
from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import PurePosixPath

from flask import Request
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class UploadRejected(Exception):
    def __init__(self, code: str, status: int, title: str, message: str):
        super().__init__(message)
        self.code, self.status, self.title, self.message = code, status, title, message


def too_large(limit_mb: int) -> UploadRejected:
    return UploadRejected("too_large", 413, "That file is too large", f"Please upload an image of {limit_mb} MB or less.")


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    filename: str
    width: int
    height: int
    image_format: str
    sha256: str


def clean_filename(name: str | None) -> str:
    """Remove any path or shell-significant characters, and cap the length."""
    cleaned = secure_filename(name or "")
    if not cleaned:
        return "upload"
    if len(cleaned) <= 120:
        return cleaned
    suffix = PurePosixPath(cleaned).suffix[:10]
    return cleaned[: 120 - len(suffix)] + suffix


def validate_upload(name: str | None, data: bytes, max_bytes: int, max_pixels: int) -> ValidatedImage:
    filename = clean_filename(name)
    if not name or not data:
        raise UploadRejected("missing_file", 400, "No image received", "Please choose an image before submitting.")
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            "wrong_type", 415, "Unsupported file type",
            f"VisionGuard accepts {', '.join(sorted(ALLOWED_EXTENSIONS))} images only.",
        )
    if len(data) > max_bytes:
        raise too_large(max_bytes // (1024 * 1024))
    undecodable = UploadRejected(
        "undecodable", 422, "We could not read that image",
        "The file has an image extension, but its contents are damaged or are not an image.",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                width, height, image_format = image.width, image.height, image.format
                if width * height > max_pixels:
                    raise UploadRejected(
                        "too_many_pixels", 413, "That image is too large",
                        "Please upload an image with fewer pixels.",
                    )
                image.load()
    except UploadRejected:
        raise
    except Exception as error:  # Pillow raises many types for malformed input.
        raise undecodable from error
    if image_format not in ALLOWED_FORMATS:
        raise UploadRejected(
            "wrong_type", 415, "Unsupported file type",
            f"The file's real format is {image_format}, which VisionGuard does not accept.",
        )
    return ValidatedImage(data, filename, width, height, image_format, hashlib.sha256(data).hexdigest())


class InMemoryRequest(Request):
    """Keep uploads in RAM. Werkzeug would otherwise spool files over 500 KB to a temp file."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()
