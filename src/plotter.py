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

"""


# std
import os
import glob
import itertools
import mmap
from multiprocessing import Pool
from pathlib import Path
import time
from tkinter.messagebox import showerror

# 3rd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm

def process_file(filename, stage_pos, s=1):
    id = str(Path(filename).parent)[-10:-6]
    id += ''.join(filter(str.isdigit, Path(filename).name))
    id = int(id)

    with open(filename, 'r') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        lines = mmapped_file.read().decode().split('\n')

    power_scale_idx = lines[0].find('power_scale:')
    power_scale = float(lines[0][power_scale_idx + 12:])

    move_y = float(lines[0].split(' ')[-1])
    move_x = float(lines[1].split(' ')[-1])

    lines = lines[6:]

    if len(lines) < 10:
        return None

    p1 = lines[0::3]
    p1 = [p for p in p1 if p != '']
    p1 = np.array([p.strip('\n').split() for p in p1]).astype(np.float32)
    p1 = np.column_stack((p1[::s, 0] + stage_pos[0] + move_x,
                          p1[::s, 1] + stage_pos[1] + move_y,
                          p1[::s, 2] + stage_pos[2],
                          p1[::s, 3],
                          np.full(int(np.ceil(len(p1)/s)), id)))

    p2 = lines[1::3]
    p2 = [p for p in p2 if p != '']
    p2 = np.array([p.strip('\n').split() for p in p2]).astype(np.float32)
    p2 = np.column_stack((p2[::s, 0] + stage_pos[0] + move_x,
                          p2[::s, 1] + stage_pos[1] + move_y,
                          p2[::s, 2] + stage_pos[2],
                          p2[::s, 3],
                          np.full(int(np.ceil(len(p2)/s)), id)))

    lines = np.stack((p1, p2), axis=1) # p1(n, 4), p2(n, 4) -> (n, 2, 4)

    # multiply power scale (everythign in 3rd column
    lines[:, :, 3] *= power_scale

    return lines

def process_file_wrapper(args):
    filename, stage_pos, less_lines = args
    result = process_file(filename, stage_pos, less_lines)
    return result

def read_lines(path, serial=False):
    path = Path(path)

    # check if it contains _job.gwl file
    if glob.glob(str(path) + '/*_job.gwl', recursive=True):
        with open(glob.glob(str(path) + '/*_job.gwl', recursive=True)[0], 'r') as file:
            lines = file.readlines()
            included_files = [path / ''.join(line.replace('\\','/').replace('\n','').split(' ')[1:3]) for line in lines if 'include' in line]
            included_files_stage_pos = [np.array(line.replace(',', '').replace('\n', '').split(' ')[3:6]).astype(np.float64).astype(int) for line in lines if '% stage pos:' in line]
    else:
        included_files = [path / f for f in os.listdir(path) if f.endswith('.gwl')]
        included_files_stage_pos = [np.array([0, 0, 0]) for _ in included_files]

    less_lines = int(np.ceil(len(included_files) / 2_000))

    if serial:
        lines = []
        for file, pos, less_lines in tqdm(zip(included_files, included_files_stage_pos, [less_lines] * len(included_files)),
                                          total=len(included_files),
                                          desc='Loading all gwl files'):
            lines.append(process_file(file, pos, less_lines))


    else:  # parallel
        with Pool() as p:
            arguments = [(filename, stage_pos, less_lines) for filename, stage_pos in
                         zip(included_files, included_files_stage_pos)]

            with tqdm(total=len(arguments), desc="Processing files") as pbar:
                lines = list(tqdm(p.imap(process_file_wrapper, arguments), total=len(arguments), position=0, leave=True))
                pbar.update(len(arguments))

    # remove all None values
    lines = [l for l in lines if l is not None]

    lines = np.array(list(itertools.chain.from_iterable(lines)))

    return lines


def plot_boolean(vol):
    print('plotting boolean')
    # Create a Light object for the additional lighting
    light = pv.Light()

    white = [1, 1, 1, 0]

    # Set the light's properties
    light.set_direction_angle(45, 45)
    light.ambient = 0.2
    light.diffuse = 0.8
    light.specular = 0.3

    pv.set_plot_theme("document")
    pv_plot = pv.Plotter(window_size=[2000, 2000])
    pv_plot.set_background('white')
    pv_plot.show_grid(color='white')

    # Add the light to the pv_plot
    pv_plot.add_light(light)

    volume = (255 * vol).astype('uint8')

    # Specify your own opacity transfer function
    opacity = np.array([1, 1, 1, 1]) * (0.05)

    # Create a UnivariateData object
    volume_data = pv.ImageData()
    volume_data.dimensions = np.array(volume.shape) + 1
    volume_data["values"] = volume.flatten(order="F")  # Flatten the array!
    volume_data.set_active_scalars("values")

    pv_plot.add_volume(volume_data, clim=[0, 255], cmap='gist_ncar', opacity=opacity)
    pv_plot.remove_legend()
    pv_plot.show()


def plot_lines_pyvista(path,
                       max_lines=100_000,
                       opacity=0.10,
                       line_width=10,
                       line_color=0.95,
                       pv_plot=None,
                       render_power=True,
                       render_time=False,
                       render_boolean=False,
                       ):

    print(f'Generating 3D model of print, color is power: {render_power}, time: {render_time}')

    # lines (line, start/stop, x/y/z/p/obj) -> (N x 2 x 5)
    lines = read_lines(path)
    lines = lines.astype(np.float32)

    if lines.shape[0] == 0:
        showerror("Error", "the output is empty?")

    if not render_power:
        lines[:, :, 3] = line_color * 100

    if len(lines) > max_lines:
        lines = lines[::int(np.round(len(lines) / max_lines)), :]

    # plotter object and lights
    if pv_plot is None:
        pv_plot = pv.Plotter(window_size=(2000, 2000))
        pv_plot.set_background('white')
        pv_plot.show_axes()  # remove the grid lines

        pv_plot.camera.position = np.array([1.33, 3.0, 1.66]) * lines[:, :, :3].reshape(-1, 3).max(0)
        pv_plot.camera.focal_point = lines[:, :, :3].reshape(-1, 3).mean(axis=0)
        pv_plot.camera.up = [0, 0, 1]
        pv_plot.camera.zoom(1.0)

    light = pv.Light()
    light.set_direction_angle(45, 45)  # Direction in degrees
    light.intensity = 10.0
    light.position = lines[:, :, :3].reshape(-1, 3).mean(axis=0)
    light.position = light.position + np.array([0, 0, 3]) * lines[:, :, :3].reshape(-1, 3).max(0)
    light.light_type = 3  # scene light
    pv_plot.add_light(light)
    pv_plot.enable_shadows()

    # colors
    num_colors = 25
    color_indexing = line_color * 100

    if render_power:
        color_indexing = np.linspace(lines[:, :, 3].min(), lines[:, :, 3].max(), num_colors)
        color_indexing = color_indexing.astype(np.float32)
        color_indexing = np.round(color_indexing, 3)
        power_by_line = np.digitize(lines[:, :, 3].mean(1), color_indexing)
        power_by_line = color_indexing[power_by_line - 1]
        power_by_line *= line_color
        lines[:, 0, 3] = power_by_line

    elif render_time:
        color_indexing = np.linspace(0, 100, num_colors)
        color_indexing = color_indexing.astype(np.float32)
        color_indexing = np.round(color_indexing, 3)
        power_by_time = np.digitize(np.arange(lines.shape[0]) / lines.shape[0] * 100, color_indexing)
        power_by_time = color_indexing[power_by_time - 1]
        power_by_time *= line_color
        lines[:, 0, 3] = power_by_time

    # jet
    plotter_line_color = np.array(plt.cm.jet((100 - color_indexing) / 100))

    if len(plotter_line_color.shape) == 1:
        color_indexing = [color_indexing]
    color_indexing *= line_color
    color_indexing = np.round(color_indexing, 3)

    # plot a solid semi transparnet ghost shape
    if render_boolean:
        print('Rendering solid boolean ghost shape...')
        solid_col = ['black', 'magenta', 'cyan', 'red', 'blue']
        # color_offset = np.random.randint(0, len(solid_col))
        color_offset = 0
        objects = np.unique(lines[:, :, 4])
        solid_mesh = []
        for n, obj in enumerate(objects):
            points = lines[:, :, :3][lines[:, :, 4] == obj]
            points = points.reshape(-1, 3)
            point_cloud = pv.PolyData(points)
            surface = point_cloud.delaunay_3d(alpha=1.0)
            solid_mesh.append(surface.extract_geometry())
            pv_plot.add_mesh(solid_mesh[n],
                             color=solid_col[(n + color_offset) % len(solid_col)],
                             opacity=0.1,)

    lines = np.round(lines, 3)

    # render the print lines with a slider
    actors = []
    def create_mesh(value):
        for actor in actors:
            pv_plot.remove_actor(actor)
        actors.clear()

        lines_for_plotter = lines[:int(value), :, :3]
        power_for_plotter = lines[:int(value), 0, 3]

        for color_n, power_value in enumerate(np.unique(color_indexing)):
            if render_power or render_time:
                # index conn and p to get the right lines for the right power
                _l = lines_for_plotter[power_for_plotter == power_value]

            else:
                _l = lines_for_plotter

            if len(_l) == 0:
                continue

            # _l is (#lines x 2 (start/stop) x 3(xyz)) - populate points and lines
            points = np.empty((0, 3), dtype=np.float32)
            _lines = []
            for line_segment in _l:
                start_idx = len(points)  # index start point
                end_idx = start_idx + 1  # index end point
                points = np.vstack([points, line_segment])
                _lines.append([2, start_idx, end_idx])

            if len(color_indexing) == 1:
                line_col =plotter_line_color[:3]
            else:
                line_col = plotter_line_color[color_n, :3]

            _lines = np.array(_lines, dtype=np.int64)
            poly_data = pv.PolyData(points, lines=_lines)
            actor = pv_plot.add_mesh(poly_data, line_width=line_width,
                                     color=line_col,
                                     opacity=opacity, smooth_shading=True,
                                     render_lines_as_tubes=True,
                                     ambient=0.2,
                                     diffuse=0.5,
                                     specular=0.5,
                                     specular_power=90,
                                     cmap='jet',
                                     )

            actors.append(actor)

        if power_value == color_indexing[-1] and (render_power or render_time):
            pv_plot.add_scalar_bar(title='Power', n_labels=num_colors,
                    width=0.5, height=0.1,)

    print('Creating laser path plot...')
    slider = pv_plot.add_slider_widget(
        create_mesh,
        [1, len(lines)],
        title=" ",
        title_color="white",
        fmt="%,0.0f",
        title_height=0.025,
        color='white',
        pointa=(0.95, 0.05),
        pointb=(0.95, 0.95),
        style='modern',
        value=len(lines),
    )

    pv_plot.show()
    pv_plot.close()

    return pv_plot

def plot_solid_from_point_cloud(path, max_lines=4_000_000, opacity=0.5):
    print('Generating 3D model of print')

    lines = read_lines(path)
    lines = lines.reshape(-1, 5)
    # lines = lines[np.lexsort((lines[:, 0], lines[:, 1], lines[:, 2]))]
    _, indices = np.unique(lines[:, 4], return_inverse=True)
    lines = lines[:, :3]

    lines = lines + np.random.random(lines.shape) * 0.1
    indices_max = indices.max()

    if len(lines) > max_lines:
        lines = lines[::int(np.round(len(lines) / max_lines)), :]
        indices = indices[::int(np.round(len(indices) / max_lines))]

    pv_plot = pv.Plotter(window_size=(2_000, 2_000))
    pv_plot.set_background('black')
    # pv_plot.show_grid(color='white')

    col = ['yellow', 'green', 'magenta', 'cyan', 'red', 'blue']

    for n in range(indices_max + 1):
        line = lines[indices == n, :3]
        point_cloud = pv.PolyData(line)
        surface = point_cloud.delaunay_3d(alpha=1.0)
        solid_mesh = surface.extract_geometry()
        pv_plot.add_mesh(solid_mesh, color=col[n % len(col)], opacity=opacity)

    pv_plot.show()


if __name__ == '__main__':
    # plot_boolean()

    start = time.time()

    path = None
    folders = [Path('~/Desktop').expanduser() / f for f in os.listdir(Path('~/Desktop').expanduser())]
    for f in folders:
        if glob.glob(str(f) + '/*_job.gwl', recursive=True):
            path = f
            break

    if not path:
        print('No _job.gwl file found')
        exit()

    pv_plot = plot_lines_pyvista(path=path)

    print(f'\n\033[95m--- Plotter took: {time.time() - start:0.2f} seconds ---\033[0m')
