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

This show how to print arrays of stl files with different properties.

All units are in um / microns unless specified elsewhere.

Coordinate system is ZYX except for within the gwl file where it is XYZ.

The part dimensions and that you designed / imported (probably mm) using your
CAD software will be assumed to be um / micron once imported.

"""

# std
import copy
import os
from pathlib import Path
import time

# local
import src.plotter as plot
from src.stl2gwl import Stl, Print, PrintProp, stl_to_gwl


if __name__ == '__main__':

    # time how long it takes to run
    render_start_time = time.time()

    print_objects = []
    output_directory = Path('~/Desktop').expanduser() / 'array_of_many_stl_many_properties_output'

    stl = ['cylinder_100diamx100.STL',
           'cube_100x100x100.STL']

    spacing_x = 110
    spacing_y = 110

    for nj, _unit_cell in enumerate(['gyroid', 'kelvin']):
        for ni, (_stl, _unit_cell_fill, _power_scale) in enumerate(zip(stl, [0.1, 0.3], [1.2, 0.5])):

            _position_offset = [0, nj * spacing_y, ni * spacing_x]

            pr = Print(
                # specify stl
                stl=Stl(filename=Path(f'../resources/stl.nosync/{_stl}'),
                        output_directory=output_directory,
                        rescale=[1.0, 1.0, 1.0],  # zyx
                        rotate=[0, 1, 2]),  # zyx

                # specify if stl unit cell
                unit_cell=None,

                # specify properties of print
                print_prop=copy.deepcopy(PrintProp(
                    resolution=[0.5, 0.5, 0.5],  # this is the slicing / laser path distances
                    position_offset=list(_position_offset),
                    unit_cell=_unit_cell,  # gyroid, cubic, octet, kelvin, weaire_phelan
                    unit_cell_fill=_unit_cell_fill,
                    unit_cell_period=30,
                    gov=300,
                    fov=[280, 300, 300],
                    power_scale=_power_scale,
                    laser_power=100.0,
                    scan_speed=100_000,
                    interp_factor=1,
                    unit_cell_inverse=False,
                    tiling_strategy='xyz',))
            )

            print_objects.append(pr)

    stl_to_gwl(print_objects)

    print(f'\n\033[95m--- Writing the gwl took: {(time.time() - render_start_time) / 60:0.2f} mins ---\033[0m')

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
