"""Model builders. Every model accepts raw RGB in [0, 255]."""
from __future__ import annotations

import keras
from keras import layers

BACKBONE_LAYER = "backbone"
# Layer-name prefixes that make up the final block of each backbone.
LAST_BLOCK = {
    "mobilenetv2": ("block_16", "Conv_1", "out_relu"),
    "efficientnetb0": ("block7a", "top_"),
}


def compile_binary(model: keras.Model, lr: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_cnn(image_size: int, dropout: float) -> keras.Model:
    inputs = keras.Input((image_size, image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="scale_0_1")(inputs)
    for index, filters in enumerate((32, 64, 128, 256), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = layers.BatchNormalization(name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="verdict")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(backbone: str, image_size: int, dropout: float, weights: str = "imagenet") -> keras.Model:
    shape = (image_size, image_size, 3)
    weights_arg = None if weights == "none" else weights
    inputs = keras.Input(shape, name="image")
    if backbone == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_minus1_1")(inputs)
        base = keras.applications.MobileNetV2(
            include_top=False, weights=weights_arg, input_shape=shape, name=BACKBONE_LAYER
        )
    elif backbone == "efficientnetb0":
        x = inputs  # Keras EfficientNet contains its own rescaling and normalisation.
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=weights_arg, input_shape=shape, name=BACKBONE_LAYER
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    base.trainable = False
    # training=False keeps BatchNorm statistics fixed, including during fine-tuning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="verdict")(x)
    return keras.Model(inputs, outputs, name=f"transfer_{backbone}")


def build_model(name: str, image_size: int, dropout: float, weights: str = "imagenet") -> keras.Model:
    if name == "cnn":
        return build_cnn(image_size, dropout)
    return build_transfer(name, image_size, dropout, weights)


def unfreeze(model: keras.Model, backbone: str, spec: str | int) -> list[str]:
    """Unfreeze the last block, or the last N layers. BatchNorm layers stay frozen."""
    base = model.get_layer(BACKBONE_LAYER)
    base.trainable = True
    if isinstance(spec, int) or str(spec).isdigit():
        count = int(spec)
        chosen = {id(item) for item in base.layers[-count:]} if count > 0 else set()
    else:
        prefixes = LAST_BLOCK[backbone]
        chosen = {id(item) for item in base.layers if item.name.startswith(prefixes)}
    for layer in base.layers:
        is_batch_norm = isinstance(layer, layers.BatchNormalization)
        layer.trainable = id(layer) in chosen and not is_batch_norm
    return trainable_layer_names(model)


def trainable_layer_names(model: keras.Model) -> list[str]:
    names: list[str] = []
    for layer in model.layers:
        if layer.name == BACKBONE_LAYER:
            names += [f"backbone/{item.name}" for item in layer.layers if item.trainable and item.trainable_weights]
        elif layer.trainable and layer.trainable_weights:
            names.append(layer.name)
    return names
