"""Command-line training:  python train.py [--epochs 40] [--batch-size 128] [--force]"""
import argparse

import cifar_core as core


def main():
    ap = argparse.ArgumentParser(description="Train the CIFAR-100 CNN")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true", help="retrain even if a valid saved model exists")
    args = ap.parse_args()

    if not args.force and core.load_trained_model() is not None and core.METRICS_PATH.exists():
        print("A valid trained model already exists - skipping training (use --force to retrain).")
        return

    metrics = core.train_model(epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    print(f"\nTest accuracy : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy: {metrics['test_top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved to      : {core.MODEL_PATH}")


if __name__ == "__main__":
    main()
