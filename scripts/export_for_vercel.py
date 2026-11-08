"""Task 7: convert the champion to TensorFlow Lite and compare it with the Keras model.

TensorFlow is too large for a Vercel function, so the API runs this smaller export.
"""
import _bootstrap  # noqa: F401
import argparse
import io
import shutil
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

from visionguard.config import PROJECT_ROOT, SPLIT_DIR
from visionguard.preprocess import bytes_to_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "champion" / "model.keras")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    target = PROJECT_ROOT / "vercel_api"

    model = keras.models.load_model(args.model, compile=False)
    (target / "model.tflite").write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    shutil.copy(PROJECT_ROOT / "webapp" / "validation.py", target / "validation.py")
    print(f"model.tflite: {(target / 'model.tflite').stat().st_size / 1e6:.1f} MB")

    split = SPLIT_DIR / "test.csv"
    if not split.exists():
        return
    interpreter = Interpreter(model_path=str(target / "model.tflite"))
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    size = int(model.input_shape[1])
    differences = []
    for path in pd.read_csv(split)["path"].head(args.samples):
        data = (PROJECT_ROOT / path).read_bytes()
        keras_probability = float(model(bytes_to_batch(data, size), training=False)[0][0])
        with Image.open(io.BytesIO(data)) as image:
            resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        interpreter.set_tensor(input_details["index"], np.asarray(resized, dtype=np.float32)[np.newaxis, ...])
        interpreter.invoke()
        differences.append(abs(keras_probability - float(interpreter.get_tensor(output_details["index"])[0][0])))
    print(f"Keras vs Vercel path on {len(differences)} test images: mean difference in P(AI) {np.mean(differences):.4f}, largest {np.max(differences):.4f}")


if __name__ == "__main__":
    main()
