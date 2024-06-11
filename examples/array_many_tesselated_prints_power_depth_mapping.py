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

This shows printing arrays of tessellated prints with different power and depth
mapping properties.

All units are in um / microns unless specified elsewhere.

Coordinate system is ZYX except for within the gwl file where it is XYZ.

The part dimensions and that you designed / imported (probably mm) using your
CAD software will be assumed to be um / micron once imported.

"""

# std
import copy
from pathlib import Path
import time

# 3rd

# local
import src.plotter as plot
from src.stl2gwl import Stl, Print, PrintProp, stl_to_gwl


if __name__ == '__main__':

    # time how long it takes to run
    render_start_time = time.time()

    output_directory = Path('~/Desktop').expanduser() / 'array_many_tesselated_prints_power_depth_mapping'

    print_objects = []

    scan_speed = [33_000, 66_000]
    unit_cell_period = [30, 88]

    spacing_x = 120
    spacing_y = 120

    for ni, _scan_speed in enumerate(scan_speed):
        for nj, _ucp in enumerate(unit_cell_period):

            _position_offset = [0, nj * spacing_y, ni * spacing_x]

            pr = Print(
                # specify stl
                stl=Stl(filename=Path('../resources/stl.nosync/cube_100x100x100.STL'),
                        output_directory=output_directory,
                        rescale=[1, 1, 1],  # stretch in z, shrink in y, no changes in x
                        rotate=[0, 1, 2]),

                # specify if stl unit cell
                unit_cell=None,

                # specify properties of print
                print_prop=PrintProp(
                    resolution=[1.0, 1.0, 1.0],  # zyx grid
                    unit_cell='octet', # gyroid, cubic, octet, kelvin, wp
                    unit_cell_fill=0.25,
                    unit_cell_period=_ucp,
                    position_offset=_position_offset,
                    interp_factor=2,
                    scan_speed=_scan_speed,
                    gov=500,
                    fov=[280, 500, 500],
                    depth_power_map=[[0, 50, 100], [90, 100, 85]],
                )
            )

            pr_anti = copy.deepcopy(pr)
            pr_anti.print_prop.unit_cell_inverse = True
            pr_anti.print_prop.power_scale = 0.5

            print_objects.append(copy.deepcopy(pr))
            print_objects.append(copy.deepcopy(pr_anti))

    stl_to_gwl(print_objects)

    print(
        f'\n\033[95m--- Writing the gwl took: {(time.time() - render_start_time) / 60:0.2f} mins ---\033[0m')

    plotter_start_time = time.time()
    plot.plot_lines_pyvista(
        path=output_directory,
        max_lines=200_000,
        opacity=0.1,
        line_width=10,
        render_power=True,
        render_time=False,
        render_boolean=False,
    )
    time_taken = (time.time() - plotter_start_time) / 60
    if time_taken > 0.1:
        print(f'\n\033[95m--- Plotting the gwl took: {time_taken:0.2f} mins ---\033[0m')