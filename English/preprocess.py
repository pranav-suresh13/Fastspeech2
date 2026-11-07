import argparse
import os
import yaml
from preprocessor.preprocessor import Preprocessor


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to config.yaml (positional)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (flag)")
    args = parser.parse_args()

    cfg_file = args.config_pos if args.config_pos is not None else args.config
    print(f"Loading configuration from: {cfg_file}")
    config = load_config(cfg_file)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

