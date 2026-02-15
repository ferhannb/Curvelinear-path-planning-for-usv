import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rota_optimalds.scenario import run_default
else:
    from .scenario import run_default


def main():
    run_default(plot=True)


if __name__ == "__main__":
    main()
