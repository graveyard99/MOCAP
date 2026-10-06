"""Explicit conversion of a trusted model to a safe numerical NPZ asset.

Run: python -m openmocap.body_models.convert source.pkl model.npz --trusted-pickle
This does not obtain a license or grant permission to redistribute the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import load_body_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model", choices=["smpl", "smplh", "smplx"], default="smpl")
    parser.add_argument(
        "--num-betas",
        type=int,
        default=10,
        help="Persistent shape directions to retain (default 10)",
    )
    parser.add_argument(
        "--trusted-pickle",
        action="store_true",
        help="Explicitly allow executing your trusted pickle during conversion",
    )
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve() or args.destination.exists():
        parser.error("Choose a new destination; conversion never overwrites an asset")
    body = load_body_model(
        args.source,
        model_name=args.model,
        trust_pickle=args.trusted_pickle,
        num_betas=args.num_betas,
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as output:
        np.savez_compressed(
            output,
            v_template=body.vertices,
            f=body.faces,
            shapedirs=body.shapedirs,
            posedirs=body.posedirs,
            J_regressor=body.joint_regressor,
            weights=body.weights,
            parents=body.parents,
            joint_names=np.array(body.names),
        )
    load_body_model(args.destination, model_name=args.model, num_betas=args.num_betas)
    print(f"Validated numerical {args.model} asset: {args.destination}")


if __name__ == "__main__":
    main()
