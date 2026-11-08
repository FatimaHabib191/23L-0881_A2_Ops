"""Tasks 2.2-2.4 and 3.3: train one configuration and log it to MLflow."""
import _bootstrap  # noqa: F401
import argparse

from visionguard.config import TrainConfig
from visionguard.tracking import setup_tracking
from visionguard.training import train


def main() -> None:
    defaults = TrainConfig(run_name="")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=["cnn", "mobilenetv2", "efficientnetb0"], default=defaults.model)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--fine-lr", type=float, default=defaults.fine_lr)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--augment", choices=["none", "light", "strong"], default=defaults.augment)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--fine-epochs", type=int, default=defaults.fine_epochs)
    parser.add_argument("--unfreeze", default=defaults.unfreeze, help="block, none, or a layer count")
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--logging", choices=["manual", "autolog"], default=defaults.logging)
    parser.add_argument("--weights", choices=["imagenet", "none"], default=defaults.weights)
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    args = vars(parser.parse_args())
    backend = args.pop("backend")
    print("MLflow tracking:", setup_tracking(backend=backend).split("@")[-1])
    train(TrainConfig(**args))


if __name__ == "__main__":
    main()
