import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
from pydantic import BaseModel
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Compress an image using ImageMagick.")
    parser.add_argument("input", type=Path, help="Path to the source image file")
    parser.add_argument(
        "output", type=Path, help="Path where the compressed image will be saved"
    )
    return parser.parse_args()


def get_image_side_length():
    image_width = 100
    image_height = 100
    assert image_width == image_height
    return image_width


class Frame(BaseModel):
    file_path: Path
    rotation: float
    time: float
    transform_matrix: List[List[float]]

    @property
    def translation(self) -> Tuple:
        return tuple(np.array(self.transform_matrix)[:3, 3])

    @property
    def quaternion(self) -> Tuple:
        return tuple(
            Rotation.from_matrix(np.array(self.transform_matrix)[:3, :3]).as_quat()
        )


class Transforms(BaseModel):
    camera_angle_x: float
    frames: List[Frame]


def create_input_path_files(images_path: Path, path: Path, bry):
    transforms: Transforms = Transforms.model_validate_json(
        (images_path / f"transforms_{bry}.json").read_text()
    )

    image_side_length = get_image_side_length()
    focal_length = (image_side_length / 2) / math.tan(transforms.camera_angle_x / 2)
    principle_point = image_side_length / 2
    camera_id = 1
    with (path / "cameras.txt").open("x") as f:
        csv.writer(f, delimiter=" ").writerow(
            [
                camera_id,
                "SIMPLE_PINHOLE",
                image_side_length,
                image_side_length,
                focal_length,
                principle_point,
                principle_point,
            ]
        )

    with (path / "images.txt").open("x") as f:
        writer = csv.writer(f, delimiter=" ")
        for index, frame in enumerate(transforms.frames):
            writer.writerow(
                [
                    index,
                    *frame.quaternion,
                    *frame.translation,
                    camera_id,
                    f"{frame.file_path.name}.png",
                ]
            )
            writer.writerow([])

    (path / "points3D.txt").touch(exist_ok=False)


def bryan(images: Path, output_path: Path):
    bry = "train"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    input_path = output_path / "sparse"
    input_path.mkdir(exist_ok=False)
    create_input_path_files(images, input_path, bry)

    cmd = [
        "colmap",
        "image_undistorter",
        "--image_path",
        str(images / bry),
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path / "dense"),
        "--output_type",
        "COLMAP",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compression failed: {e.stderr.strip()}") from e

    return result.stdout


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.output.exists():
        print(f"Warning: Overwriting existing file {args.output}")

    output = bryan(args.input, args.output)
    print(f"Compression complete. Details:\n{output}")


if __name__ == "__main__":
    main()
