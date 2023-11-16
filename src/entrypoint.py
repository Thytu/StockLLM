import dvc.api
import importlib


def main(stage_to_execute: str):

    params = dvc.api.params_show()

    pkg = importlib.import_module(params["stages"][stage_to_execute]["module"])

    pkg.main(
        **params["stages"][stage_to_execute]["params"]
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python entrypoint.py module-to-import")

    main(stage_to_execute=sys.argv[1])
