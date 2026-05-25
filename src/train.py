"""
Training entry point.

Owner: Zhen Luo
Responsibility: Parse config, create dataset / model / trainer, launch training

Usage:
    python -m src.train --config configs/default.yaml
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="WalkerNet Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    # TODO: add more CLI overrides if needed
    return parser.parse_args()


def main():
    # TODO: implement
    # 1. Parse args and load config (yaml -> dict)
    # 2. Set random seed
    # 3. Create datasets (train / val)
    # 4. Create DataLoaders
    # 5. Create model (from config)
    # 6. Create Trainer and run
    pass


if __name__ == "__main__":
    main()
