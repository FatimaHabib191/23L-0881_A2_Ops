import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.inference import InferenceError, Verdict  # noqa: E402
from webapp.settings import Settings  # noqa: E402


def make_settings(tmp_path, **overrides) -> Settings:
    values = dict(
        inference_mode="local", model_path="unused", model_version="VisionGuard:test",
        model_threshold=0.5, database_url=f"sqlite:///{tmp_path / 'test.db'}", max_upload_mb=1,
        max_pixels=4_000_000, model_api_url="", model_api_key="", model_api_timeout=5.0,
        secret_key="test", history_page_size=20,
    )
    values.update(overrides)
    return Settings(**values)


class FakePredictor:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def predict(self, image_bytes, filename="upload"):
        self.calls += 1
        if self.fail:
            raise InferenceError("offline")
        return Verdict("ai_generated", 0.91, 0.91, 0.5, "VisionGuard:test", "local", 7)

    def status(self):
        return {"model_loaded": True}


def image_bytes(fmt="JPEG", size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 40, 200)).save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def jpeg():
    return image_bytes()
