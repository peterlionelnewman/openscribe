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

This script processes 3D STL files and converts them to a .GWL file to be
printed with Nanoscribe 3DNP. Various functions and printing parameters can be
set in the script including:
    - unit cell intersections with the STL object,
    - laser illumination functions,
    - print strategies,
    - hatch spacing,
    - laser power,

All units are in um / microns unless specified elsewhere.

Coordinate system is ZYX except for within the gwl file where it is XYZ.

The part dimensions and that you designed / imported (probably mm) using your
CAD software will be assumed to be um / micron once imported.

"""

# std
import datetime
import json
import io
# noinspection PyPackageRequirements
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import shutil
import sys

# 3rd
import attr
import cv2
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from stl import mesh
from tqdm import tqdm

# user
import src.unit_cell as unit_cell
import src.utils as utils
import src.plotter as plot


@attr.s()
class PrintProp:
    # print parameters for rendering input

    # ['linear', 'gaussian', None]
    laser_power: float = attr.ib(default=100.0)
    illumination_function: str = attr.ib(default="gaussian")
    illumination_function_minus: float = attr.ib(default=5.0)
    # 'z mapping power' i.e. different powers at differet depths [[z_min, z_max]; [power_min, power_max]]
    depth_power_map: list = attr.ib(default=[None, None])
    # the fit type of the power vs depth map; linear (slinear) or full 3d order spline (cubic)
    depth_power_map_fittype: str = attr.ib(default="slinear")
    # [y, x]; (µm) sub-field that galvo prints (after fov), list = attr.ib(default=fov[1:] is good if no galvoSub field
    gov: float = attr.ib(default=101.0)
    # (µm) over-sampling of galvo_subdivision_yx
    gov_overlap: float = attr.ib(default=4.0)
    # [z, y, x]; size of a given field of print (before sub fov with galvo)
    fov: list = attr.ib(default=[280, 300, 300])
    # [z, y, x]; overlap between fov
    fov_overlap: list = attr.ib(default=[5, 8, 8])
    # [z, y, x]
    resolution: list = attr.ib(default=[0.4, 0.25, 0.25])
    # These used to be implemented but were removed. launch an issue to request these features?
    # 'Fast', 'Fast-Wait', 'Long-Dist', 'Long-Dist-Wait', 'Bogo-Long', 'Bogo-Wait', 'Stage', 'Simple', 'Fast-no-rotate'
    print_strategy: str = attr.ib(default="fast")
    # None or Checkerboard
    tiling_strategy: str = attr.ib(default="checkerboard")
    # (s); between every wait_count layers (in addition to Nanoscribe's 'piezo_settling_time'
    wait_time: int = attr.ib(default=None)
    # every 'wait_count' lines, printer waits 'wait_time'
    wait_count: int = attr.ib(default=None)
    # or can be path
    unit_cell: str = attr.ib(default="None")
    unit_cell_inverse: bool = attr.ib(default=False)
    # (scalar factor relates to volumetric ratio of solid : space)
    unit_cell_fill: float = attr.ib(default=1.0)
    # length of gyroid unit cell (xyz) symmetry
    unit_cell_period: float = attr.ib(default=18.0)
    # (scalar); interpolate factor for stl object
    interp_factor: float = attr.ib(default=1.0)
    # the change in angle of approach for rastering each slice - works if > 0
    rotation_degrees: float = attr.ib(default=-1)

    # these parameters don't require rendering reprocessing of the stl object
    galvo_acceleration: float = attr.ib(default=10.0)
    # (ms); time between layers, Nanoscribe's 'piezo_settling_time' parameter
    piezo_settling_time: float = attr.ib(default=0.01)
    # (µm/s); speed of galvo
    scan_speed: float = attr.ib(default=100_000.0)
    # (scalar); power scaling factor for laser max is ~1.22
    power_scale: float = attr.ib(default=1.0)
    # for sweeps -> without offset by default
    position_offset: list = attr.ib(default=[0, 0, 0])
    # (µm/s); speed of stage
    stage_velocity: float = attr.ib(default=1000.0)

    # script state, saving and other
    file_id: list = attr.ib(default=[])
    file: list = attr.ib(default=[])


@attr.s()
class Stl:

    filename: Path = attr.ib(default=None)
    output_directory: Path = attr.ib(default="")

    triangles: np.array = attr.ib(default=None)
    # [z, y, x]; (µm) object bounds / size of the stl object
    bounds: np.ndarray = attr.ib(default=None)

    is_unit_cell: bool = attr.ib(default=False)

    # (scalar); rescale factor for stl object
    rescale: float = attr.ib(default=1.0)
    # (axis order); flips stl object
    rotate: list = attr.ib(default=[0, 1, 2])

    def import_stl(self):
        # Load the STL file
        try:
            return mesh.Mesh.from_file(str(self.filename)).vectors
        except:
            return mesh.Mesh.from_file(str(self.filename)[1:]).vectors

    def init_stl_class(self):

        s = f"\033[32;1;4m \nLoading {self.filename.stem}.STL,\033[0m"
        if self.rescale != 1.0:
            s += f" & rescale factor {''.join(str(self.rescale))}\033[0m"
        print(s)

        # import STL file
        triangles = self.import_stl()

        # round to 3 decimal places == nm
        triangles = np.round(triangles, 3)

        # positive z direction
        triangles[:, :, 0] = triangles[:, :, 0] - triangles[:, :, 0].min()

        if self.rotate != [0, 1, 2]:
            for idx, val in enumerate(self.rotate):
                if val < 0:  # flip the axis for negative values
                    triangles[:, :, idx] = np.abs(
                        (triangles[:, :, idx] - triangles[:, :, idx].max())
                    )
                    self.rotate[idx] = abs(val)
            triangles = triangles[:, :, self.rotate]

        # xyz -> zyx (T, 3 points, xyz) -> (T, 3 points, 3 zyx) ~rescaling here!
        triangles = np.flip(triangles, axis=2)

        # rescale / stretch
        if isinstance(self.rescale, float):
            triangles = triangles * self.rescale
        elif isinstance(self.rescale, list) and len(self.rescale) == 3:
            triangles[:, :, 0] = triangles[:, :, 0] * self.rescale[0]
            triangles[:, :, 1] = triangles[:, :, 1] * self.rescale[1]
            triangles[:, :, 2] = triangles[:, :, 2] * self.rescale[2]
        else:
            raise ValueError("rescale must be a float or list of length 3")

        # get bounds of stl object
        self.bounds = np.array([np.min(triangles, axis=1).min(axis=0),
                                np.max(triangles, axis=1).max(axis=0),])

        self.triangles = triangles.astype(np.float32)

@attr.s()
class Print:
    # stl file properties
    stl: Stl = attr.ib()  # stl object

    # unit cell stl
    unit_cell: Stl = attr.ib()  # unit cell stl object

    # print parameters for rendering input
    print_prop: PrintProp = attr.ib()  # print properties object

    # gov parameters for rendering
    # [z, y, x]; (µm) stage positions of each gov
    gov_stage_pos: np.ndarray = attr.ib(default=None)
    # [z, y, x]; (µm) top/bottom of each gov window
    gov_bounds: np.ndarray = attr.ib(default=None)
    # [z, y, x]; filenames of the gov parts
    gov_names: list = attr.ib(default=None)

    # has it been rendered boolean
    rendered: bool = attr.ib(default=False)  # True if the stl has been rendered

    # each print gets an id
    file_id: str = attr.ib(default=None)

    def is_already_rendered(self):
        # guilty till proven innocent
        self.rendered = False

        # look for json files
        print_prop_files = list(self.stl.output_directory.glob("print_prop_*.json"))

        if len(print_prop_files) == 0:
            return False

        # loda in all the jsons
        props_dicts = []

        # find all print prop files
        for print_prop_file in print_prop_files:
            with open(print_prop_file, "r") as f:
                props_dicts.append(json.load(f))

        print(f'p:{props_dicts}')

        # Compare each print_props dictionary to Print.print_prop
        for i, prop_dict in enumerate(props_dicts):
            ignored_keys = {
                "file_id",
                "log",
                "piezo_settling_time",
                "galvo_acceleration",
                "scan_speed",
                "power_scale",
                "position_offset",
            }
            keys_to_check = set(prop_dict.keys()) - ignored_keys
            checked_keys_boolean = [
                self.print_prop.__getattribute__(key) == prop_dict[key]
                for key in keys_to_check
            ]
            if all(checked_keys_boolean):
                print(f"{self.stl.filename.name} already rendered..")
                self.rendered = True
                self.file_id = props_dicts[i].get("file_id")
                self.stl.output_directory_files = (
                    self.stl.output_directory / f"{self.print_prop.file_id}_files"
                )

                self.stl.init_stl_class()

                self.process_govs()

                return True
            else:
                keys_that_changed = [
                    keys
                    for (keys, key_bool) in zip(keys_to_check, checked_keys_boolean)
                    if not key_bool
                ]
                # print(f"print properties in print: {i} are differ to this print: {', '.join(keys_that_changed)}")
        return False

    def process_govs(self):
        if self.print_prop.gov > self.print_prop.fov[1]:
            self.print_prop.gov = self.print_prop.fov[1]
            print(
                f"\033[95mWarning: gov size is larger than fov size, "
                f"setting gov size to fov size: {self.print_prop.gov}\033[0m"
            )

        if self.print_prop.gov > self.print_prop.fov[2]:
            self.print_prop.gov = self.print_prop.fov[2]
            print(
                f"\033[95mWarning: gov size is larger than fov size, "
                f"setting gov size to fov size: {self.print_prop.gov}\033[0m"
            )

        # fov should otherwise be a multiple of gov
        if self.print_prop.fov[1] % self.print_prop.gov != 0:
            print(
                f"\033[95mfov size is not a multiple of gov size, "
                f"could make weird overlapping fovs/govs\033[0m"
            )
        if self.print_prop.fov[2] % self.print_prop.gov != 0:
            print(
                f"\033[95mfov size is not a multiple of gov size, "
                f"could make weird overlapping fovs/govs\033[0m"
            )

        # generate fov and gov positions
        initial_position = np.array([
            self.print_prop.fov[0] / 2,
            self.stl.bounds[0, 1] + self.print_prop.fov[1] / 2,
            self.stl.bounds[0, 2] + self.print_prop.fov[2] / 2,
        ])

        stl_range = np.array([
            self.stl.bounds[1, 0],  # structure should be zero'd when imported in stl class
            self.stl.bounds[1, 1] - self.stl.bounds[0, 1],
            self.stl.bounds[1, 2] - self.stl.bounds[0, 2],
        ])

        no_fov = np.ceil(
            stl_range / np.array([
                self.print_prop.fov[0] - self.print_prop.fov_overlap[0],
                self.print_prop.fov[1] - self.print_prop.fov_overlap[1],
                self.print_prop.fov[2] - self.print_prop.fov_overlap[2],
            ])).astype("int")

        if ((self.print_prop.gov == self.print_prop.fov[1]) and
                (self.print_prop.gov == self.print_prop.fov[2])):
            no_gov = np.array([1, 1])
        else:
            no_gov = np.ceil(
                np.minimum(np.array(self.print_prop.fov[1::]), stl_range[1::])
                / np.array(self.print_prop.gov - self.print_prop.gov_overlap)
            ).astype("int")

        fov_stage_pos = np.zeros((no_fov[0],
                                  no_fov[1],
                                  no_fov[2], 3))
        gov_stage_pos = np.zeros((no_fov[0],
                                  no_fov[1] * no_gov[0],
                                  no_fov[2] * no_gov[1], 3))
        fov_bounds = np.zeros((no_fov[0],
                               no_fov[1],
                               no_fov[2], 2, 3))
        gov_bounds = np.zeros((no_fov[0],
                               no_fov[1] * no_gov[0],
                               no_fov[2] * no_gov[1], 2, 3))
        gov_names = []
        for k in range(no_fov[0]):
            for j in range(no_fov[1]):
                for i in range(no_fov[2]):
                    fov_position = initial_position + np.array([k, j, i]) * (
                        np.array(self.print_prop.fov)
                        - np.array(self.print_prop.fov_overlap)
                    )
                    fov_stage_pos[k, j, i, :] = fov_position
                    # center the fov
                    fov_lower = fov_position - np.array(self.print_prop.fov) / 2
                    fov_bounds[k, j, i, 0, :] = fov_lower
                    fov_upper = fov_position + np.array(self.print_prop.fov) / 2
                    fov_bounds[k, j, i, 1, :] = fov_upper

                    for J in range(no_gov[0]):
                        for I in range(no_gov[1]):
                            gov_lower = (
                                fov_lower
                                + np.array([0, J, I]) * self.print_prop.gov
                                - (np.array([0, J, I]) * self.print_prop.gov_overlap)
                            )
                            # gov_lower = np.minimum(gov_lower, fov_upper) - self.print_prop.resolution
                            gov_lower = gov_lower - self.print_prop.resolution
                            gov_bounds[
                                k, j * no_gov[0] + J, i * no_gov[1] + I, 0, :
                            ] = gov_lower
                            gov_upper = gov_lower + [
                                self.print_prop.fov[0],
                                self.print_prop.gov,
                                self.print_prop.gov,
                            ]
                            # gov_upper = np.minimum(gov_upper, fov_upper) + self.print_prop.resolution
                            gov_upper = gov_upper + self.print_prop.resolution
                            gov_bounds[
                                k, j * no_gov[0] + J, i * no_gov[1] + I, 1, :
                            ] = gov_upper
                            gov_stage_pos[
                                k, j * no_gov[0] + J, i * no_gov[1] + I, :
                            ] = fov_position
                            gov_names.append(f"{k}-{j}-{i}_{J}-{I}.gwl")

        # flatten gov bounds and gov stage pos -> (S, 3) and (S, 2, 3)
        gov_bounds = gov_bounds.reshape((-1, 2, 3))
        # remove duplicate gov bounds
        gov_bounds, ind = np.unique(
            gov_bounds, axis=0, return_index=True
        )
        gov_stage_pos = gov_stage_pos.reshape((-1, 3))
        gov_stage_pos = gov_stage_pos[ind]
        gov_names = [gov_names[i] for i in ind]

        self.gov_stage_pos = gov_stage_pos
        self.gov_bounds = gov_bounds.astype(np.float32)
        self.gov_names = gov_names

    def init_print_class(self):
        if self.file_id is None:
            self.file_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")

        # update file name in print props
        self.print_prop.file = str(self.stl.filename.name)

        # check if equivalent stl has been rendered with these print props
        if self.is_already_rendered():
            return

        self.print_prop.file_id = self.file_id

        # check for triangles in stl object; if not load them
        if not isinstance(self.stl.triangles, np.ndarray):
            self.stl.init_stl_class()

        # check for triangles in unit cell object; if not load them
        if isinstance(self.unit_cell, Stl):
            self.unit_cell.init_stl_class()
        else:
            self.unit_cell = "not stl"

        if isinstance(self.stl.output_directory, Path):
            self.stl.output_directory.mkdir(exist_ok=True)
            self.stl.output_directory_files = (
                self.stl.output_directory / f"{self.file_id}_files"
            )
            self.stl.output_directory_files.mkdir(exist_ok=True)
        else:
            raise ValueError("output_directory must be specified (pathlib:Path)")

        # initialize gov parameters
        self.process_govs()


def triangles_in_bounds(triangles, bounds):
    """
    returns a subset of triangles when any part of yx coordinates of a given
    triangle extends into the yx coordinates of the bounding box defined by bounds

    triangles: (M, 3, 3) array of triangles, m triangles, 3 points, 3 zyx
    bounds: (2, 3) array of bounds of the vol; [[z_min, y_min, x_min], [z_max,
    y_max, x_max]]
    """

    tri_min = np.min(triangles[:, :, 1:], axis=1)
    tri_max = np.max(triangles[:, :, 1:], axis=1)

    y_overlap = np.logical_not(
        (tri_max[:, 0] < bounds[0, 1]) | (tri_min[:, 0] > bounds[1, 1])
    )
    x_overlap = np.logical_not(
        (tri_max[:, 1] < bounds[0, 2]) | (tri_min[:, 1] > bounds[1, 2])
    )

    in_bounds = np.logical_and(y_overlap, x_overlap)

    return in_bounds

def stl_to_arr(triangles, bounds, resolution, interp=1):
    """
    uses the Möller–Trumbore algorithm to calculate the intersections of rays
    and triangles in a given volume defined by bounds and resolution.

    triangles: (M, 3, 3) np.ndarray of triangles, (m triangles, 3 points, 3 zyx)
    bounds: (2, 3) np.ndarray of bounds of the vol, ((low, high) (z, y, x))
    resolution: (3,) list of voxel edge length in vol, (z, y, x)
    interp: int, interpolation factor (can downsize for speed, if needed)

    NOTE: torch, numpy and a Rust implementation were tested for this function;
    the parrallel numpy was generally more stable with minimal improvements from
    torch and Rust.

    """
    if triangles.shape[0] == 0:
        return np.zeros((2,2,2)), np.zeros(1), np.zeros(1), np.zeros(1)

    # setup all the parameters and grid
    r = np.array(resolution) * interp
    pi = np.pi

    # offset by an irrational number to avoid hitting grid/voxel lines
    offset = r * pi / 6

    z_max = triangles[:, :, 0].max()
    z_min = triangles[:, :, 0].min()
    y_max = triangles[:, :, 1].max()
    y_min = triangles[:, :, 1].min()
    x_max = triangles[:, :, 2].max()
    x_min = triangles[:, :, 2].min()

    # reduce the bounds to the minimum required
    if bounds[1, 0] > z_max:
        bounds[1, 0] = z_max
    if bounds[0, 0] < z_min:
        bounds[0, 0] = z_min
    if bounds[1, 1] > y_max:
        bounds[1, 1] = y_max
    if bounds[0, 1] < y_min:
        bounds[0, 1] = y_min
    if bounds[1, 2] > x_max:
        bounds[1, 2] = x_max
    if bounds[0, 2] < x_min:
        bounds[0, 2] = x_min

    # make grid
    z = np.arange(bounds[0, 0] - r[0] * (offset[0] * 0.75), bounds[1, 0], r[0])
    y, x = np.meshgrid(
        np.arange(bounds[0, 1] - r[1] * (offset[1] * 0.75), bounds[1, 1], r[1]),
        np.arange(bounds[0, 2] - r[2] * (offset[2] * 0.75), bounds[1, 2], r[2]),
        indexing="ij",
    )

    vol = np.zeros([len(z), y.shape[0], x.shape[1]])

    # flatten and draw lines from an irrational number close to the center of the voxel
    z = z + offset[0]
    y_flat = y.flatten() + offset[1] * 1 / 3
    x_flat = x.flatten() + offset[2] * 2 / 3

    if np.any(np.isin(triangles[:, :, 1], y_flat) | np.isin(triangles[:, :, 2], x_flat)):
        print("Triangles and rays share coordinates, potential voxelization errors.")

    # calculate intersections of lines and triangles with Möller–Trumbore algorithm
    def calc_intersections(i, q1, q2, q1_minus_q2, e1, e2, t1, bounds, r):
        q1_ = q1[i : i + batch_size]
        q2_ = q2[i : i + batch_size]
        q1_minus_q2_ = q1_minus_q2[i : i + batch_size]
        c1 = q1_minus_q2_[..., 1] * e2[..., 2] - q1_minus_q2_[..., 2] * e2[..., 1]
        c2 = q1_minus_q2_[..., 2] * e2[..., 0] - q1_minus_q2_[..., 0] * e2[..., 2]
        c3 = q1_minus_q2_[..., 0] * e2[..., 1] - q1_minus_q2_[..., 1] * e2[..., 0]
        h = np.stack((c1, c2, c3), axis=-1)
        a = np.einsum("...ij,...ij->...i", e1, h)
        f = np.divide(1, a, out=np.zeros_like(a), where=a != 0)
        s = q1_ - t1
        u = f * np.einsum("...ij,...ij->...i", s, h)
        c1 = s[..., 1] * e1[..., 2] - s[..., 2] * e1[..., 1]
        c2 = s[..., 2] * e1[..., 0] - s[..., 0] * e1[..., 2]
        c3 = s[..., 0] * e1[..., 1] - s[..., 1] * e1[..., 0]
        q = np.stack((c1, c2, c3), axis=-1)
        v = f * np.einsum("...ij,...ij->...i", q2_ - q1_, q)
        t = f * np.einsum("...ij,...ij->...i", e2, q)
        intersections = (u >= 0) & (u <= 1) & (v >= 0) & ((u + v) <= 1)
        intersection_coordinates = q1_ + t[..., None] * (q2_ - q1_)
        intersection_index = intersection_coordinates[intersections]
        intersection_index = np.clip(intersection_index, bounds[0], bounds[1])
        intersection_index = (intersection_index - bounds[0]) / r
        intersection_index = np.round(intersection_index).astype(int)
        return intersection_index

    def batch_process_intersections(q1, q2, e1, e2, t1, bounds, r, vol, batch_size, n_jobs):
        q1_minus_q2 = q2 - q1
        intersection_indices = Parallel(n_jobs=n_jobs)(
            delayed(calc_intersections)(i, q1, q2, q1_minus_q2, e1, e2, t1, bounds, r)
            for i in range(0, q1.shape[0], batch_size)
        )
        vshape = vol.shape

        for intersection_index in intersection_indices:
            intersection_index[:, 0] = np.clip(intersection_index[:, 0], 0, vshape[0])
            intersection_index[:, 1] = np.clip(intersection_index[:, 1], 0, vshape[1])
            intersection_index[:, 2] = np.clip(intersection_index[:, 2], 0, vshape[2])
            np.add.at(vol, (intersection_index[:, 0], intersection_index[:, 1], intersection_index[:, 2], ), 1)
        return vol

    q1 = np.stack([np.ones_like(x_flat) * bounds[1, 0] + r[0], y_flat, x_flat], axis=1)[:, np.newaxis]
    q2 = np.stack([np.ones_like(x_flat) * bounds[0, 0] - 1.5 * r[0], y_flat, x_flat], axis=1)[:, np.newaxis]

    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]

    n_jobs = cpu_count() - 1
    batch_size = np.minimum(q1.shape[0], 5_000)  # change if memory issues?
    vol = batch_process_intersections(
        q1, q2, e1, e2, triangles[:, 0], bounds, r, vol, batch_size, n_jobs
    )

    vol = np.cumsum(vol, axis=0)
    vol = vol % 2
    vol = vol > 0
    # plot.plot_boolean(vol)

    return vol, z, y, x


def intersect_unit_cell_and_vol(vol, unit_cell, bounds_min, resolution, stage_pos):

    # calculate size of vol and unit cell
    vol_size = np.array(vol.shape)
    unit_cell_size = np.array(unit_cell.shape)

    # calc number of times unit cell repeated in each direction to cover vol
    repeat_counts = np.ceil(vol_size / unit_cell_size).astype(int) + 1

    # tile unit cell
    tiled_unit_cells = np.tile(unit_cell, repeat_counts)

    # slice to adjust to start to the appropriate position
    start_mod = np.mod(
        utils.custom_round_to_int((bounds_min + stage_pos) / resolution),
        unit_cell_size.astype(np.int64),
    )

    tiled_unit_cells = tiled_unit_cells[
        start_mod[0] : start_mod[0] + vol_size[0],
        start_mod[1] : start_mod[1] + vol_size[1],
        start_mod[2] : start_mod[2] + vol_size[2],
    ]

    # intersection of vol and the tiled unit cells
    return np.logical_and(vol, tiled_unit_cells)


def render_gov_to_gwl(
        stage_pos,
        gov_filename,
        triangles,
        gov_bounds,
        unit_cell,
        print_prop
    ):

    # using the gov_bounds, go get the vol, and make the mesh
    triangle_boolean = triangles_in_bounds(
        triangles,
        np.array(
            [
                gov_bounds[0]
                - np.array(print_prop.resolution) * print_prop.interp_factor,
                gov_bounds[1]
                + np.array(print_prop.resolution) * print_prop.interp_factor,
            ]
        ),
    )

    triangles = triangles[triangle_boolean]

    vol, z, y, x = stl_to_arr(
        triangles, gov_bounds, print_prop.resolution, print_prop.interp_factor
    )

    # plot_boolean(vol)

    # remove vol on boundaries
    vol[0, :, :] = 0
    vol[-1, :, :] = 0
    vol[:, 0, :] = 0
    vol[:, -1, :] = 0
    vol[:, :, 0] = 0
    vol[:, :, -1] = 0

    if vol.sum() == 0:
        return

    if print_prop.interp_factor != 1:
        vol = rescale(vol, print_prop.interp_factor, order=0, mode="reflect")
        z = rescale(z, print_prop.interp_factor, order=1, mode="reflect")
        y = rescale(y, print_prop.interp_factor, order=1, mode="reflect")
        x = rescale(x, print_prop.interp_factor, order=1, mode="reflect")

    y -= stage_pos[1]
    x -= stage_pos[2]

    # tile unit cell and intersect with vol
    if isinstance(unit_cell, np.ndarray):
        bounds_min = np.array([z[0], y[0, 0], x[0, 0]])
        vol = intersect_unit_cell_and_vol(
            vol,
            unit_cell,
            bounds_min,
            print_prop.resolution,
            stage_pos
        )

    vol = vol.astype("uint8")

    ones = np.ones_like(x)

    # gov to gwl; one gov_vol slice at a time
    if print_prop.unit_cell_fill < 1:
        sigma = (print_prop.unit_cell_period / 2) * print_prop.unit_cell_fill
        sigma = np.argmin(np.abs(x[0, :] - x[0, :].min() - sigma))
    else:
        sigma = len(x) // 2

    # pick a random offset here in degree 0 -> 180
    rotation_offset = np.random.randint(0, 180)

    with (open(gov_filename, "w") as f):
        # store gwl strings in a buffer
        buffer = io.StringIO()

        first_line_printed = False

        # for ind in np.arange(vol.shape[0]):
        for ind in np.arange(len(z)):
            v = vol[ind, :, :]

            if z[ind] < 0:
                continue

            if np.sum(v) == 0:
                continue

            if not first_line_printed:
                # put stage position in buffer
                buffer.write(
                    f"% stage_position: "
                    f"{stage_pos[2]:0.2f}, "
                    f"{stage_pos[1]:0.2f}, "
                    f"{stage_pos[0]:0.2f}; position_offset: "
                    f"{print_prop.position_offset[2]}, "
                    f"{print_prop.position_offset[1]}, "
                    f"{print_prop.position_offset[0]}; power_scale: "
                    f"{print_prop.power_scale}\n"
                )  # in xyz

                # reset stage starting pos
                buffer.write("0 0 0 1\n")
                buffer.write("1 0 0 1\n")
                buffer.write("Write\n")
                buffer.write("Wait 0.5\n\n")

                first_line_printed = True

            mask = v == 1

            # highest index where np.cumsum(mask.sum(0) > 0) == 0
            row_sum = np.cumsum(mask.sum(0) > 0)
            col_sum = np.cumsum(mask.sum(1) > 0)
            emp = np.array([[np.argwhere(row_sum == 0).flatten().max(),
                             np.argwhere(row_sum == row_sum.max()).flatten().min()],
                            [np.argwhere(col_sum == 0).flatten().max(),
                             np.argwhere(col_sum == col_sum.max()).flatten().min()]])

            v = v[emp[1, 0]:emp[1, 1], emp[0, 0]:emp[0, 1]]
            y_ = y[emp[1, 0]:emp[1, 1], emp[0, 0]:emp[0, 1]]
            x_ = x[emp[1, 0]:emp[1, 1], emp[0, 0]:emp[0, 1]]

            # illumination functions
            if "none" in print_prop.illumination_function:
                p = print_prop.laser_power * ones
            elif "gaussian" in print_prop.illumination_function:
                pad_amount = min(int(np.round(3 * sigma)), x.shape[0] // 2)
                mask = v == 1
                v_blur = np.pad(v, pad_amount, mode="constant")
                v_blur = np.abs(v_blur.astype("float32") - 1)
                v_blur = gaussian_filter(v_blur, sigma=sigma)
                v_blur = v_blur[pad_amount:-pad_amount, pad_amount:-pad_amount]
                v_blur -= v_blur.min()
                v_blur *= mask
                v_blur /= np.max(v_blur)
                v_blur *= print_prop.illumination_function_minus
                v_blur += print_prop.laser_power
                v_blur -= print_prop.illumination_function_minus
                v_blur[~mask] = print_prop.laser_power
                p = v_blur
                if not np.isfinite(p).any():
                    raise ValueError("power values should be finite")
            elif "linear" in print_prop.illumination_function:
                p = np.pad(ones, 1, mode="constant")
                p = cv2.distanceTransform(p.astype("uint8"), cv2.DIST_L2, 3)
                p = print_prop.laser_power * (
                    1 - (print_prop.illumination_function_minus / 100) * (p / np.max(p))
                )
                p = np.clip(p, 0, print_prop.laser_power)
            else:
                print("no illumination function selected")

            # apply a power mapping by print depth function
            if isinstance(print_prop.depth_power_map, interp1d):
                p = (p / 100.0) * print_prop.depth_power_map(z[ind])

            # start the rotation of the gov
            if print_prop.unit_cell == "gyroid":
                # gyroids have an optimal angle of attack
                rotation_degrees = (90 - z[ind] * 180 / print_prop.unit_cell_period * 2) % 180
            else:
                rotation_degrees = (15 * ind + rotation_offset) % 180

            # override the above if it was specified
            if print_prop.rotation_degrees > 0:
                rotation_degrees = (print_prop.rotation_degrees * ind  + rotation_offset) % 180

            # rotate slices; !bug at boundary values -> cv2
            p_rot = utils.rotate(p, rotation_degrees, p=False)
            y_rot = utils.rotate(y_, rotation_degrees, p=False)
            x_rot = utils.rotate(x_, rotation_degrees, p=False)
            v_rot = utils.rotate(v, rotation_degrees, p=True)
            v_rot = np.pad(v_rot, 1, mode="constant")

            # calculate where the laser would hit the sample
            v_rot = np.diff(v_rot, n=1, axis=0, prepend=0)
            v_rot = v_rot[1:, 1:]
            laser_off = v_rot == -1
            laser_on = v_rot == 1

            # find the indicies of the laser hits
            laser_off = np.asarray(np.where(np.transpose(laser_off)))
            laser_off[1] = laser_off[1] - 1
            laser_on = np.asarray(np.where(np.transpose(laser_on)))

            # remove laser hits that turn on and off at the same point
            laser_on_laser_off = np.argwhere(laser_on[1] == laser_off[1])
            laser_off = np.delete(laser_off, laser_on_laser_off, axis=1)
            laser_on = np.delete(laser_on, laser_on_laser_off, axis=1)

            # combine arrays for on and off
            laser = np.vstack((laser_on, laser_off)).T

            # if all entries for laser were removed restart the loop
            if not np.any(laser):
                continue

            # left/right and up/down * wizardry *,
            laser_up = laser[
                np.cumsum(np.abs(np.diff(np.hstack((laser[0, 0], laser[:, 0])))) > 0)
                % 2
                == 0,
                :,
            ]
            laser_down = laser[
                np.cumsum(np.abs(np.diff(np.hstack((laser[0, 0], laser[:, 0])))) > 0)
                % 2
                == 1,
                :,
            ]
            laser_down = laser_down[:, [0, 3, 2, 1]]
            laser_down[:, 1] = laser_down[:, 1] * -1
            laser_down = laser_down[np.lexsort(laser_down[:, [1, 0]].T), :]
            laser_down[:, 1] = laser_down[:, 1] * -1
            laser = np.vstack((laser_up, laser_down))
            laser = laser[np.argsort(laser[:, 0], kind="stable"), :]

            # calculate distance of line between laser indices
            line_distance = np.abs(laser[:, 3] - laser[:, 1]) * print_prop.resolution[1]
            if print_prop.illumination_function == None:
                line_distance = line_distance * 0.0

            # if line_distance is > unit_cell_period, split the line into segments of unit_cell_period
            long_line_index = np.squeeze(np.argwhere(line_distance > print_prop.unit_cell_period))
            long_line_num_segments = np.array(np.squeeze(np.ceil(line_distance[long_line_index] / print_prop.unit_cell_period + 1)))

            if long_line_index.size == 1:
                long_line_index = [long_line_index]
                long_line_num_segments = [long_line_num_segments]
            # split the long lines into segments and put back inside the laser array (in reverse order)
            for i, num_segments in zip(long_line_index[::-1], long_line_num_segments[::-1]):
                short_lines = np.linspace(laser[i, 1], laser[i, 3], int(num_segments)).astype("int")
                short_lines = np.squeeze([short_lines[i : i + 2] for i in range(short_lines.shape[0] - 1)])
                short_lines = np.vstack((laser[i][0] * np.ones(short_lines.shape[0]), short_lines[:, 0], laser[i][2] * np.ones(short_lines.shape[0]), short_lines[:, 1],))
                laser = np.delete(laser, i, axis=0)
                laser = np.insert(laser, i, short_lines.T, axis=0)

            # alternative left/right to right/left every slice
            if ind % 2 == 1:
                laser = laser[::-1, :]

            # points to be written are in xyzp format rather than zyx
            laser_ones = np.ones(laser.shape[0])
            points_on = np.vstack(
                (x_rot[laser[:, 1], laser[:, 0]],
                 y_rot[laser[:, 1], laser[:, 0]],
                 laser_ones * z[ind] - stage_pos[0] + print_prop.fov[0] / 2,
                 p_rot[laser[:, 1], laser[:, 0]],)).T

            points_off = np.vstack(
                (x_rot[laser[:, 3], laser[:, 2]],
                 y_rot[laser[:, 3], laser[:, 2]],
                 laser_ones * z[ind] - stage_pos[0] + print_prop.fov[0] / 2,
                 p_rot[laser[:, 3], laser[:, 2]],)).T

            for i in np.arange(points_on.shape[0]):
                buffer.write(
                    f"{points_on[i, 0]:0.2f} "
                    f"{points_on[i, 1]:0.2f} "
                    f"{points_on[i, 2]:0.2f} "
                    f"{points_on[i, 3]:0.0f}\n"
                    f"{points_off[i, 0]:0.2f} "
                    f"{points_off[i, 1]:0.2f} "
                    f"{points_off[i, 2]:0.2f} "
                    f"{points_off[i, 3]:0.0f}\n"
                    f"Write\n"
                )

        f.write(buffer.getvalue())


def render_vol_to_gov(print_obj):

    # initialise print object, includes check if already rendered
    print_obj.init_print_class()

    if print_obj.rendered:
        return print_obj

    # update resolution
    if print_obj.print_prop.resolution[1] != print_obj.print_prop.resolution[2]:
        raise ValueError(f"Resolution in y and z must be the same")

    # update depth map to a spline
    if not print_obj.print_prop.depth_power_map[0] == 'None' and \
                not print_obj.print_prop.depth_power_map[0] == None:
        if (isinstance(print_obj.print_prop.depth_power_map, list) and \
                len(print_obj.print_prop.depth_power_map) == 2):

            x = print_obj.print_prop.depth_power_map[0]
            y = print_obj.print_prop.depth_power_map[1]

            if len(print_obj.print_prop.depth_power_map[0]) >= 2:
                print_obj.print_prop.depth_power_map = interp1d(x, y, kind=print_obj.print_prop.depth_power_map_fittype, bounds_error=False, fill_value=(y[0], y[-1]))
            else:
                raise ValueError(f"Depth power map must have at least 2 points for both x and y")

    # render unit cell if stl
    if (print_obj.print_prop.unit_cell_inverse and print_obj.print_prop.unit_cell_fill == 1):
        print("Unit_cell_fill==1.0 & Unit_cell_inverse==True," "... fill==0.0 skipping")
        return print_obj
    elif (not print_obj.print_prop.unit_cell_inverse and print_obj.print_prop.unit_cell_fill == 0):
        print("Unit_cell_fill==0.0 & Unit_cell_inverse==False," "... fill==0.0 skipping")
        return print_obj

    elif print_obj.print_prop.unit_cell_fill == 1 or (not isinstance(print_obj.print_prop.unit_cell, str) and not isinstance(print_obj.unit_cell, Stl)):
        print(f"\033[35;1;4m Rendering gwl\033[0m for "
            f"'\033[32;1;4m{print_obj.stl.filename.name} \033[0m'"
            f"without unit cell")
        unit_cell_boolean = False

    elif isinstance(print_obj.unit_cell, Stl):
        print(f"\033[35;1;4m Rendering gwl\033[0m for "
            f"'\033[32;1;4m{print_obj.stl.filename.name}\033[0m' "
            f"with stl defined "
            f"'\033[32;1;4m{print_obj.unit_cell.filename.name}\033[0m' "
            f"unit cells")
        unit_cell_range = print_obj.unit_cell.bounds[1] - print_obj.unit_cell.bounds[0]
        master_stl_range = print_obj.stl.bounds[1] - print_obj.stl.bounds[0]
        if any([i > j for (i, j) in zip(unit_cell_range, master_stl_range)]):
            raise ValueError(f"Unit cell bounds {print_obj.unit_cell.bounds} are"
                f"larger than master stl bounds {print_obj.stl.bounds}")

        # using the gov_bounds, go get the vol, and make the mesh
        unit_cell_boolean, _, _, _ = stl_to_arr(
            print_obj.unit_cell.triangles,
            print_obj.unit_cell.bounds,
            print_obj.print_prop.resolution,
            1,
        )

        # can be introduced through the gui conversion and casting
        print_obj.print_prop.unit_cell_period = float(np.min(print_obj.unit_cell.bounds[1] - print_obj.unit_cell.bounds[0]))
        print_obj.print_prop.unit_cell_fill = 1.0
        if print_obj.print_prop.unit_cell_inverse:
            unit_cell_boolean = np.logical_not(unit_cell_boolean)

    # render if unit cell is math defined
    elif isinstance(print_obj.print_prop.unit_cell, str):
        is_inverse = ""
        if print_obj.print_prop.unit_cell_inverse:
            is_inverse = "INVERSE "
        print(f"\033[35;1;4m Rendering gwl\033[0m for "
            f"'\033[32;1;4m{print_obj.stl.filename.name}\033[0m' "
            f"with {is_inverse}math defined "
            f"'\033[32;1;4m{print_obj.print_prop.unit_cell}\033[0m' "
            f"unit cells, fill: {print_obj.print_prop.unit_cell_fill}")
        unit_cell_boolean = unit_cell.from_math(
            print_obj.print_prop.unit_cell,
            print_obj.print_prop.unit_cell_period,
            print_obj.print_prop.unit_cell_fill,
            np.array(print_obj.print_prop.resolution),
            print_obj.print_prop.unit_cell_inverse,)
    else:
        raise ValueError(f"Unit cell, or lack there of; not defined")

    extended_gov_names = [print_obj.stl.output_directory / f"{print_obj.file_id}_files" / gov_name for gov_name in print_obj.gov_names]

    print(f"Rendering govs into gwl", end="",)
    for stage_pos, gov_name, gov_bounds in tqdm(
        zip(print_obj.gov_stage_pos, extended_gov_names, print_obj.gov_bounds),
        total=len(print_obj.gov_names),
    ):
        render_gov_to_gwl(
            stage_pos,
            gov_name,
            print_obj.stl.triangles,
            gov_bounds,
            unit_cell_boolean,
            print_obj.print_prop,
        )

    # save print props / add to print_prop.json
    print("Saving print_prop.json to output directory")
    for i in range(1, 9999):
        if (print_obj.stl.output_directory / f"print_prop_{i}.json").exists():
            continue
        else:
            print_prop_json_path = print_obj.stl.output_directory / f"print_prop_{i}.json"
            break
    print_prop_dict = {prop: getattr(print_obj.print_prop, prop) for prop in
                       dir(print_obj.print_prop) if not prop.startswith("__")}
    for key, value in print_prop_dict.items():  # interp1d unserializeable
        if isinstance(value, interp1d):
            print_prop_dict[key] = np.random.randint(100000, 999999)
    with open(print_prop_json_path, "a+") as f:
        json.dump(print_prop_dict, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print_obj.rendered = True  # if rendered successfully

    return print_obj


def draw_assembly(printer, gov_names, gov_stage_pos):
    # draw figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # extract for efficient draw
    power = np.zeros(len(printer))
    position = np.zeros((len(printer), 3))
    for n, (p, gn, gsp) in enumerate(zip(printer, gov_names, gov_stage_pos)):
        power[n] = p.print_prop.laser_power
        gn = gn.replace("_", " ").replace("-", " ")[:-4].split()
        position[n] = np.array(
            [
                gsp[0],
                gsp[1] + float(gn[3]) * p.print_prop.gov,
                gsp[2] + float(gn[4]) * p.print_prop.gov,
            ]
        )

    power /= power.max()

    ax.scatter(
        position[:, 2], position[:, 1], position[:, 0], c=power, cmap="viridis", s=10
    )

    print_enumeration = [str(i) for i in np.arange(len(printer))]

    texts = []
    for pos, label in zip(position, print_enumeration):
        texts.append(ax.text(pos[2], pos[1], pos[0] + position[:, 0].max() * 0.05, label, fontsize=3))

    # adjust_text(texts)
    # plt.show()
    plt.savefig(p.stl.output_directory / "assembly.png", dpi=300)


def tiling_sorter(pos, tiling_strategy, fov_size, bounds):
    """
    pos: Nx5 array of FOVZ, FOVY, FOVX; GOVY, GOVX
    tile_strategy: 'checkerboard' or 'spiral' or 'zyx' or 'xyz'
    """
    if pos.shape[1] != 3:
        raise ValueError(f"pos should be nx5 (FOV Z, Y, X; gov y, x), not nx{pos.shape}")

    # pos to ints
    pos -= pos.min(axis=0)
    for i in range(3):
        pos_col = pos[:, i]
        pos_unique = np.unique(pos_col)
        for n, num in enumerate(pos_unique):
            pos_col[pos_col == num] = n
        pos[:, i] = pos_col

    if (pos % 1).sum() != 0.0:
        raise ValueError(f"pos should be integers, not {pos}")

    pos = pos.astype(int)

    if tiling_strategy in ["checkerboard", "check", "cb", "checkers"]:
        checkerboard = (pos[:, 1] + pos[:, 2]) % 2
        ind = np.lexsort((checkerboard, pos[:, 0]))

    elif tiling_strategy in ["spiral"]:
        fov_center = pos[:, 1].mean(), pos[:, 2].mean()
        fov_dist = np.max(np.stack((np.abs(pos[:, 1] - fov_center[0]), np.abs(pos[:, 2] - fov_center[1]))), axis=0,)
        fov_angle = np.arctan2(pos[:, 1] - fov_center[0], pos[:, 2] - fov_center[1])
        ind = np.lexsort((fov_angle, -fov_dist, pos[:, 0]))

    elif tiling_strategy in ["zyx" or "none" or "None" or "NONE"]:
        ind = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))

    elif tiling_strategy in ["xyz" or "XYZ"]:
        # round to fov from the center of bounds
        pos = pos - bounds.mean(axis=0)
        pos[:, 0] = 0
        pos[:, 1:] = np.round(pos[:, 1:] / fov_size[1:])
        ind = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))

    else:
        # raise ValueError(f"Tiling strategy {tiling_strategy} not defined")
        print("that wasn't a valid tiling strategy")
        ind = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))

    return ind


def assemble_gwl(print_list):
    """
    Print_list is a list of print_obj class
    """
    print(f"Assembling all gwl files in a {print_list[0].print_prop.tiling_strategy}")
    # destroy stl triangles, save memory
    for p in print_list:
        p.stl.triangles = None
        if isinstance(p.unit_cell, Stl):
            p.unit_cell.triangles = None

    # check all prints are rendered, and rendered with same tiling strategy
    for n, p in enumerate(print_list):
        if not p.rendered:
            del print_list[n]
            print("Not all prints are rendered")

        if p.print_prop.tiling_strategy != print_list[0].print_prop.tiling_strategy:
            raise ValueError(f"Prints have different tiling strategies")

    # load all gov.GWL with stage position & print properties
    for n, p in enumerate(print_list):
        if n == 0:
            gov_names = p.gov_names
            gov_stage_pos = p.gov_stage_pos + np.array(p.print_prop.position_offset)
            printer = [p] * len(p.gov_names)
            pos = np.mean(p.gov_bounds, axis=1)
            continue

        gov_names.extend(p.gov_names)
        gov_stage_pos = np.vstack((gov_stage_pos, p.gov_stage_pos + np.array(p.print_prop.position_offset)))
        printer.extend([p] * len(p.gov_names))
        pos = np.vstack((pos, np.mean(p.gov_bounds, axis=1)))

    # get the index for sorting by the tiling strategy
    tiling_index = tiling_sorter(pos, p.print_prop.tiling_strategy, p.print_prop.fov, p.stl.bounds)

    # sort by tiling index
    gov_stage_pos = gov_stage_pos[tiling_index, :]
    gov_names = [gov_names[n] for n in tiling_index]
    printer = [printer[n] for n in tiling_index]

    # write the *master*_job.gwl file
    job_name = [p.stl.filename.name for p in print_list]
    job_name = list(dict.fromkeys(job_name))
    job_name = "_".join(job_name) + "_job.txt"
    job_name = job_name.replace(".STL", "")
    job_name = p.stl.output_directory / job_name
    with open(str(job_name), "w") as f:

        # store gwl strings in a buffer
        buffer = io.StringIO()

        # header with files info
        for print_obj in print_list:
            buffer.write(f"% file: {print_obj.stl.filename.stem}\n")
            if isinstance(print_obj.unit_cell, Stl):
                f"% unit_cell: {print_obj.unit_cell.filename.stem}\n"
            buffer.write(f"% created: {datetime.datetime.now()}\n"
                f"% print_prop: {print_obj.print_prop}\n\n")

        # header with print meta parameters
        buffer.write(
            f"InvertZAxis 1\n"
            f"TimeStampOn\n"
            f'MessageOut "starting print."\n'
            f"GalvoScanMode\n"
            f"ContinuousMode\n"
            f"StageVelocity {p.print_prop.stage_velocity} % 1000 is fast 200 is normal / slower\n"
            f"XOffset {0:0.2f}\n"
            f"YOffset {0:0.2f}\n"
            f"ZOffset 20"
        )

        p_ = p
        stage_position = np.array([0, 0, 0])  # initialize
        for n, (p, file, pos) in enumerate(zip(printer, gov_names, gov_stage_pos)):

            # check if file exists
            if not (p.stl.output_directory / (p.file_id + "_files") / file).exists():
                continue

            # check file size is not zero
            if (p.stl.output_directory / (p.file_id + "_files") / file).stat().st_size == 0:
                (p.stl.output_directory / (p.file_id + "_files") / file).unlink()  # delete if zero size
                continue

            buffer.write(f"\nWait 1.0\n")

            if (p.print_prop.piezo_settling_time != p_.print_prop.piezo_settling_time or n == 0):
                buffer.write(f"\nPiezoSettlingTime {p.print_prop.piezo_settling_time}\n")

            if p.print_prop.power_scale != p_.print_prop.power_scale or n == 0:
                buffer.write(f"\nPowerScaling {p.print_prop.power_scale}\n")

            if (p.print_prop.galvo_acceleration != p_.print_prop.galvo_acceleration or n == 0):
                buffer.write(f"\nGalvoAcceleration {p.print_prop.galvo_acceleration} % 1-10 recommended for x25 obj\n")

            if p.print_prop.scan_speed != p_.print_prop.scan_speed or n == 0:
                buffer.write(f"\nScanSpeed {p.print_prop.scan_speed} % in um/s\n")

            # check if stage position changed; # if so, add move stage command
            move_pos = pos - stage_position
            if move_pos[2] != 0:
                buffer.write(f"MoveStageX {move_pos[2]:0.3f}\n")
            if move_pos[1] != 0:
                buffer.write(f"MoveStageY {move_pos[1]:0.3f}\n")
            if move_pos[0] != 0 and n != 0:
                buffer.write(f"AddZDrivePosition {move_pos[0]:0.3f}\n")
            # update stage position
            stage_position = pos

            # include gov object
            s = f'include {Path(p.file_id + "_files") / file}\n'
            buffer.write(s.replace("/", "\\"))  # windows-ification
            s = np.array2string(pos[::-1], separator=", ", formatter={"float_kind": lambda x: f"{x:.2f}"},)[1:-1]
            buffer.write(f"% stage pos: {s}\n")
            buffer.write(f'MessageOut "finished gov {n}: {file} out of {len(gov_names)}"\n')
            buffer.write(f'CapturePhoto "{p.file_id}_{file[0:-4]}.tiff"\n')

            p_ = p

        f.write(buffer.getvalue())

    # rename _data.txt to _data.gwl
    os.rename(str(job_name), str(job_name)[:-4] + ".gwl")

    if len(printer) > 1:
        draw_assembly(printer, gov_names, gov_stage_pos)


def remove_previous_renders(output_directory):
    # make directory for output
    if output_directory.exists():

        # check the size of the directory
        dir_size = sum(
            f.stat().st_size for f in output_directory.glob("**/*") if f.is_file()
        )
        if dir_size > 0:
            print(
                f"Directory '{output_directory}'"
                f"already exists and is not empty, overwriting."
            )

        # error if directory is big
        too_big_size = 1_000_000_000  # ~1 GB
        if dir_size > too_big_size:
            raise ValueError(
                f"Directory '{output_directory}'"
                f"already exists and is pretty big: {dir_size / too_big_size:.2f} GB"
            )

        # delete directory if it already exists
        shutil.rmtree(output_directory)


def stl_to_gwl(print_list):

    # confirm output directory is the same for all prints
    output_directory = print_list[0].stl.output_directory
    for n, p in enumerate(print_list):
        if n == 0:
            continue
        if isinstance(p.stl.output_directory, Path):
            p.stl.output_directory = output_directory
        elif isinstance(p.stl.output_directory, str) and p.stl.output_directory == "":
            p.stl.output_directory = output_directory
        elif not p.stl.output_directory == output_directory:
            raise ValueError("All prints must have the same output directory")

    remove_previous_renders(output_directory)

    # render all prints
    for n, p in enumerate(print_list):
        if p.print_prop.unit_cell_fill == 1.0 and p.print_prop.unit_cell_inverse:
            print(
                "Unit_cell_fill==1.0 & Unit_cell_inverse==True,"
                "... fill==0.0 skipping"
            )
            continue
        print_list[n] = render_vol_to_gov(p)

    assemble_gwl(print_list)


if __name__ == "__main__":
    print("\n\033[95m--- don't run me here ---\033[0m\n\n\n")
