"""
OpenScribe is free and open-source software for the design and simulation of 3D
nanoprinted materials. Please contribute to the development of this software by
reporting bugs, suggesting new features, and submitting pull requests.

3D PRINTED MATERIALS WITH NANOVOXELATED ELASTIC MODULI
Repo: https://github.com/peterlionelnewman/openscribe
Paper doi: ----------------
Paper url: ----------------
Peter L. H. Newman, 2024, (peterlionelnewman @github)

As well, we'd love if you'd cite us if you use our script in any way :-)

Creative Commons Attribution-NonCommercial 4.0 International Public License
github.com/peterlionelnewman/openscribe/LICENSE


--------------------------------------------------------------------------------

This file contains utility functions that are used in the OpenScribe software.

"""


# std
import re

# 3rd
import cv2
import numpy as np

# local


def rotate(image, angle, p=False):
    if angle == 0:
        return image

    # determine padding
    angle_rad = np.abs(np.radians(angle))
    padding = int(np.ceil(max(image.shape) * (angle_rad % (np.pi / 2)))) + 1

    # pad
    if p:
        image = np.pad(image, padding, mode="constant", constant_values=0)
    else:
        image = np.pad(image, padding, mode="edge")

    # rotation
    (height, width) = image.shape[:2]
    (center_x, center_y) = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    rot_mat[0, 2] += (new_width / 2) - center_x
    rot_mat[1, 2] += (new_height / 2) - center_y
    image = cv2.warpAffine(
        image, rot_mat, (new_width, new_height), flags=cv2.INTER_AREA
    )

    image = image[padding:-padding, padding:-padding]  # remove padding

    return image


def custom_round_to_int(arr):
    # since numpy does a half round to even, which is introducing errors
    return np.where(arr - np.floor(arr) >= 0.5, np.ceil(arr), np.floor(arr)).astype(int)


def parse_printer_position_from_file(file_path='path/to/gwl/output/directory'):
    with open(file_path, 'r') as f:
        code = f.read()

    # Initial positions
    x_pos, y_pos = 0.0, 0.0

    # Extract initial X, Y, and Z offsets
    x_offset_match = re.search(r'XOffset (\d+\.\d+)', code)
    if x_offset_match:
        x_pos = float(x_offset_match.group(1))

    y_offset_match = re.search(r'YOffset (\d+\.\d+)', code)
    if y_offset_match:
        y_pos = float(y_offset_match.group(1))

    # Iterate over each MoveStage command to update positions
    for match in re.finditer(r'MoveStageX ([+-]?\d+\.\d+)', code):
        x_pos += float(match.group(1))

    for match in re.finditer(r'MoveStageY ([+-]?\d+\.\d+)', code):
        y_pos += float(match.group(1))

    return x_pos, y_pos