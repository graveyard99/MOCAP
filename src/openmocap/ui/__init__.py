"""Desktop application. Numerical and pipeline modules never import this package."""


def main() -> int:
    from .app import launch

    return launch()
