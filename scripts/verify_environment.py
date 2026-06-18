from __future__ import annotations

import importlib.util


def main() -> None:
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tensorflow",
    ]
    for package in packages:
        if importlib.util.find_spec(package) is None:
            raise ImportError(f"Missing package: {package}")

    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch
    import transformers
    import datasets
    import accelerate
    import tensorflow as tf

    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA available: {torch.cuda.is_available()}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"torch CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch CUDA device 0: {torch.cuda.get_device_name(0)}")
        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"torch CUDA tensor check: {y.shape} on {y.device}")

    print(f"transformers: {transformers.__version__}")
    print(f"datasets: {datasets.__version__}")
    print(f"accelerate: {accelerate.__version__}")
    print(f"tensorflow: {tf.__version__}")
    print(f"tensorflow GPU devices: {tf.config.list_physical_devices('GPU')}")
    value = tf.reduce_sum(tf.random.normal([100, 100])).numpy()
    print(f"tensorflow tensor check: {bool(value != 0)}")


if __name__ == "__main__":
    main()
