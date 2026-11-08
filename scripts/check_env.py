"""Task 1.1: print exact library versions and whether TensorFlow sees a GPU."""
import _bootstrap  # noqa: F401
import getpass
import platform
import socket
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import tensorflow as tf

print(f"user={getpass.getuser()} host={socket.gethostname()} time={datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python {platform.python_version()} on {platform.platform()}")
for package in ("tensorflow", "tensorflow-cpu", "keras", "mlflow", "dvc", "keras-tuner", "flask", "pillow", "numpy", "pandas", "scikit-learn"):
    try:
        print(f"{package:<16}{version(package)}")
    except PackageNotFoundError:
        pass
gpus = tf.config.list_physical_devices("GPU")
print(f"tf.__version__   {tf.__version__}")
print(f"GPUs visible     {len(gpus)} {gpus}")
