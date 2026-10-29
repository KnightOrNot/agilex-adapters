"""LeRobot adapters; importing this package never opens hardware."""

from .devices import Gello, GelloConfig, PiperX, PiperXConfig

__all__ = ["Gello", "GelloConfig", "PiperX", "PiperXConfig"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.print_help()
