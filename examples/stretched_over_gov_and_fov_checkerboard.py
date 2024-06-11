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

This shows a print with a cube stl file, and a gyroid unit cell, stretched over
a gov and fov with a checkerboard tiling strategy.

All units are in um / microns unless specified elsewhere.

Coordinate system is ZYX except for within the gwl file where it is XYZ.

The part dimensions and that you designed / imported (probably mm) using your
CAD software will be assumed to be um / micron once imported.

"""

# std
from pathlib import Path
import time

# 3rd

# local
import src.plotter as plot
from src.stl2gwl import Stl, Print, PrintProp, stl_to_gwl


def main_function():
    # time how long it takes to run
    render_start_time = time.time()

    output_directory = Path('~/Desktop').expanduser() / 'stretched_over_gov_and_fov_checkerboard_output'

    print_objects = []

    pr = Print(
        # specify stl
        stl=Stl(filename=Path('../resources/stl.nosync/cylinder_100diamx100.STL'),
                output_directory=output_directory,
                rescale=[1.0, 3.0, 5.0],
                rotate=[0, 1, -2]), # flip for negative values

        # specify if stl unit cell
        unit_cell=None,

        # specify properties of print
        print_prop=PrintProp(
            resolution=[2, 2, 2],  # zyx
            unit_cell='gyroid',  # gyroid, cubic, octet, kelvin, wp
            unit_cell_fill=0.10,
            unit_cell_period=50,
            power_scale=0.50,
            laser_power=100.0,
            illumination_function_minus=0,
            illumination_function='none',
            interp_factor=2,
            unit_cell_inverse=False,
            tiling_strategy='cb',  # None=xyz or Checkerboard(xy) -> z
            gov=160,
            fov=[280, 320, 320],)
    )

    stl_to_gwl([v for v in locals().values() if isinstance(v, Print)])

    print(f'\n\033[95m--- Writing the gwl took: {(time.time() - render_start_time) / 60:0.2f} mins ---\033[0m')

    plotter_start_time = time.time()
    plot.plot_lines_pyvista(
        path=output_directory,
        max_lines=200_000,
        opacity=0.1,
        line_width=10,
        render_power=False,
        render_time=True,
        render_boolean=False,
    )
    time_taken = (time.time() - plotter_start_time) / 60
    if time_taken > 0.1:
        print(f'\n\033[95m--- Plotting the gwl took: {time_taken:0.2f} mins ---\033[0m')

if __name__ == '__main__':
    main_function()
