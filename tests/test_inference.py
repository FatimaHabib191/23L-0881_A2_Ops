import keras
import numpy as np
import pytest
from conftest import image_bytes

from visionguard.evaluation import choose_threshold, compute_metrics
from visionguard.models import build_cnn
from visionguard.preprocess import bytes_to_batch
from webapp.inference import LocalPredictor, ModelLoadError, verdict_from_probability


def test_preprocess_shape_range_and_rgb():
    batch = bytes_to_batch(image_bytes("PNG"), 96).numpy()
    assert batch.shape == (1, 96, 96, 3)
    assert 0 <= batch.min() and batch.max() <= 255 and batch[0, 48, 48].argmax() == 2


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_supported_formats_decode(fmt):
    assert bytes_to_batch(image_bytes(fmt), 32).shape == (1, 32, 32, 3)


def test_verdict_confidence_is_for_the_chosen_class():
    assert verdict_from_probability(0.8, 0.5) == ("ai_generated", 0.8)
    label, confidence = verdict_from_probability(0.2, 0.5)
    assert label == "real" and confidence == pytest.approx(0.8)


def test_local_predictor_loads_keras_file(tmp_path):
    path = tmp_path / "model.keras"
    build_cnn(32, 0.1).save(path)
    predictor = LocalPredictor(str(path), 0.5, "v-test")
    verdict = predictor.predict(image_bytes("JPEG"))
    assert verdict.label in {"real", "ai_generated"} and 0.5 <= verdict.confidence <= 1
    assert predictor.status()["image_size"] == 32


def test_missing_model_fails_clearly():
    with pytest.raises(ModelLoadError, match="MODEL_PATH"):
        LocalPredictor("/nope/model.keras", 0.5, "v")


def test_saved_model_keeps_its_scaling_layer(tmp_path):
    path = tmp_path / "model.keras"
    build_cnn(32, 0.1).save(path)
    assert any(isinstance(layer, keras.layers.Rescaling) for layer in keras.models.load_model(path).layers)


def test_threshold_respects_false_alarm_cap():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.7, 50), np.linspace(0.3, 1.0, 50)])
    chosen = choose_threshold(y, p, max_fpr=0.1)
    assert chosen["val_fpr"] <= 0.1
    assert compute_metrics(y, p, chosen["threshold"])["false_alarm_rate"] <= 0.1
