import argparse
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class Config:
    iteration_count: int


def train(config: Config):
    for i in tqdm(range(config.iteration_count), desc="Training"):
        print(i)


def main():
    argument_parser = argparse.ArgumentParser(prog="Animating Gaussian Splats")
    argument_parser.add_argument("-i", "--iteration-count", type=int, default=200_000)
    args = argument_parser.parse_args()
    config = Config(iteration_count=args.iteration_count)
    train(config=config)


if __name__ == "__main__":
    main()
