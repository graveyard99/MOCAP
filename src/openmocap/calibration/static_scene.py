"""Optional COLMAP integration keeps performer pixels out of scene tracking."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from numpy.typing import ArrayLike


def static_scene_mask(performer_mask: ArrayLike, *, padding_pixels: int = 12) -> np.ndarray:
    """COLMAP mask: 255 permits static features; 0 rejects actor and border."""
    mask = (np.asarray(performer_mask) > 0).astype(np.uint8)
    if mask.ndim != 2 or padding_pixels < 0:
        raise ValueError("Performer mask must be 2D; padding must be nonnegative")
    if padding_pixels:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * padding_pixels + 1, 2 * padding_pixels + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return ((1 - mask) * 255).astype(np.uint8)


def run_colmap(
    image_directory: str | Path,
    work_directory: str | Path,
    *,
    mask_directory: str | Path | None = None,
    executable: str = "colmap",
    sequential: bool = True,
    use_gpu: bool = False,
) -> Path:
    """Execute official installed COLMAP and write a text reconstruction.

    Masks must use COLMAP image-name + '.png' convention. Output has ambiguous
    scale and MUST be aligned/imported with a measured metric reference.
    Commands use argument arrays, never downloaded code or shell interpolation.
    """
    binary = shutil.which(executable)
    if binary is None:
        raise FileNotFoundError(
            "COLMAP is unavailable; install an official local binary or import a solved camera track"
        )
    images, work = Path(image_directory).resolve(), Path(work_directory).resolve()
    if not images.is_dir():
        raise ValueError("COLMAP image directory does not exist")
    work.mkdir(parents=True, exist_ok=True)
    sparse, text = work / "sparse", work / "text"
    sparse.mkdir(exist_ok=True)
    text.mkdir(exist_ok=True)
    database = work / "database.db"
    extractor = [
        binary,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--SiftExtraction.use_gpu",
        str(int(use_gpu)),
    ]
    if mask_directory:
        extractor.extend(["--ImageReader.mask_path", str(Path(mask_directory).resolve())])
    commands = [
        extractor,
        [
            binary,
            "sequential_matcher" if sequential else "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            str(int(use_gpu)),
        ],
        [
            binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    models = sorted(path for path in sparse.iterdir() if path.is_dir())
    if not models:
        raise RuntimeError("COLMAP did not produce a registered static-scene reconstruction")
    subprocess.run(
        [
            binary,
            "model_converter",
            "--input_path",
            str(models[0]),
            "--output_path",
            str(text),
            "--output_type",
            "TXT",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return text
