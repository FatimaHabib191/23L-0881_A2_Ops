"""The one decode-and-resize path shared by training and serving.

Models take raw RGB values in [0, 255]. Value scaling lives inside each saved
model as layers, so a serving process cannot apply the wrong normalisation.
"""
from __future__ import annotations

import tensorflow as tf


def decode_and_resize(image_bytes, image_size: int) -> tf.Tensor:
    """Decode encoded image bytes to a float32 HxWx3 tensor in [0, 255]."""
    image = tf.io.decode_image(
        image_bytes, channels=3, dtype=tf.uint8, expand_animations=False
    )
    image.set_shape([None, None, 3])
    # Direct resize: the aspect ratio is not preserved (see report section 2.1).
    image = tf.image.resize(image, (image_size, image_size), method="bilinear", antialias=True)
    return tf.cast(image, tf.float32)


def bytes_to_batch(image_bytes: bytes, image_size: int) -> tf.Tensor:
    """Build a batch of one for inference."""
    return tf.expand_dims(decode_and_resize(tf.constant(image_bytes), image_size), 0)
