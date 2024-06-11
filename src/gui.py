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

All units are in um / microns unless specified elsewhere.

Coordinate system is ZYX except for within the gwl file where it is XYZ.

"""

# std
import os
from multiprocessing import Process
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showerror
from pathlib import Path
import sys

# 3rd
import numpy as np

# local
sys.path.append("./src")
import plotter as plot
from stl2gwl import Stl, Print, PrintProp, stl_to_gwl


class PrinterApp:

    def __init__(self, root):
        self.print_objects_dictionary = {}

        # Setting up the main window
        self.w = 900
        self.h = 1100
        self.b = 20

        self.root = root
        self.root.title("OpenScribe")
        self.root.geometry(f'{self.w}x{self.h}')

        # Setting up the frames
        self.left_frame_width = int(0)
        self.left_frame_height = self.h - 2 * self.b
        self.left_frame = tk.Frame(root, width=self.left_frame_width, height=self.left_frame_height, bg='light gray')
        self.left_frame.place(x=self.b, y=self.b)

        self.right_frame_width = int(self.w - self.left_frame_width - 4 * self.b)
        self.right_frame_height = self.h - 4 * self.b
        self.right_frame = ttk.Notebook(width=self.right_frame_width, height=self.right_frame_height)
        self.right_frame.place(x=self.left_frame_width + self.b, y=self.b)

        # Adding "+ new print" button at the bottom of the right frame
        self.new_print_button = tk.Button(root, text="+ new print", command=self.add_new_print)
        self.new_print_button.place(x=self.left_frame_width + 2 * self.b, y=0)

        # Adding "Render prints" button
        button_offset = 120
        self.render_prints_button = tk.Button(root, text="Render prints", command=self.render_prints)
        self.render_prints_button.place(x=self.left_frame_width + 2 * self.b + button_offset, y=0)

        self.render_prints_button = tk.Button(root, text="Visualise rendered print", command=self.visualise_rendered_print)
        self.render_prints_button.place(x=self.left_frame_width + 2 * self.b + 2 * button_offset * 1.05, y=0)

        self.v_p_e = np.load('./resources/vol_power_e.npz')['vol_power_e']

    def select_output_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.print_objects_dictionary['0']['nb']['output_directory_label'].configure(text=f"Output directory | {directory}")

    def select_stl_file(self, key):
        directory = filedialog.askopenfilename()
        if directory:
            self.print_objects_dictionary[key]['nb']['stl_label'].configure(text=f"STL file | {directory}")

    def select_unit_cell_stl_file(self, key):
        directory = filedialog.askopenfilename()
        if directory:
            self.print_objects_dictionary[key]['nb']['unit_cell_stl_label'].configure(text=f"STL unit cell file | {directory}")
            self.print_objects_dictionary[key]['nb']['unit_cell_fill'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_fill'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_period'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_period'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_z'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_z'].insert(0, 1.0)
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_y'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_y'].insert(0, 1.0)
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_x'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_x'].insert(0, 1.0)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_z'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_z'].insert(0, 0)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_y'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_y'].insert(0, 1)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_x'].delete(0, tk.END)
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_x'].insert(0, 2)
            # combobox too
            self.print_objects_dictionary[key]['nb']['unit_cell'].set('stl')

    def render_power_time_checked(self, key):
        # if render power or render time is checked, uncheck the other
        if key == 'render_power_checkbox':
            if self.print_objects_dictionary['0']['render_time_checkbox'].get():
                self.print_objects_dictionary['0']['render_time_checkbox'].set(False)
        elif key == 'render_time_checkbox':
            if self.print_objects_dictionary['0']['render_power_checkbox'].get():
                self.print_objects_dictionary['0']['render_power_checkbox'].set(False)

    def set_youngs_mod(self, key):
        # set resolution, galvo == 10, piezo == 0.1, scan_velocity == 100k
        self.print_objects_dictionary[key]['nb']['resolution_z'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['resolution_y'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['resolution_x'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['galvo_acceleration'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['piezo_settling_time'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['stage_velocity'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['scan_velocity'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['power_scale'].delete(0, tk.END)

        e = self.print_objects_dictionary[key]['nb']['youngs_mod'].get()

        try:
            e = float(e)
        except:
            showerror("Error", "Couldn't parse YM to number")

        e_min = self.v_p_e[2].min()
        e_max = self.v_p_e[2].max()

        if (e > e_max or e < e > e_min):
            showerror("Error", f"{e_min} < e < {e_max}")

        e = np.abs(self.v_p_e[2] - e).argmin()
        dim = self.v_p_e[0, e]
        dim = dim ** (1 / 3)  # linear
        pow = self.v_p_e[1, e] / 100

        self.print_objects_dictionary[key]['nb']['resolution_z'].insert(0, dim)
        self.print_objects_dictionary[key]['nb']['resolution_y'].insert(0, dim)
        self.print_objects_dictionary[key]['nb']['resolution_x'].insert(0, dim)
        self.print_objects_dictionary[key]['nb']['galvo_acceleration'].insert(0, 10)
        self.print_objects_dictionary[key]['nb']['piezo_settling_time'].insert(0, 0.1)
        self.print_objects_dictionary[key]['nb']['stage_velocity'].insert(0, 100)
        self.print_objects_dictionary[key]['nb']['scan_velocity'].insert(0, 100_000)
        self.print_objects_dictionary[key]['nb']['power_scale'].insert(0, pow)

    def set_defaults_for_objective(self, key):
        # set fov, gov, resolution, galvo_acceleration, piezo_settling_time, stage_velocity, scan_velocity

        # check the objective combo box
        objective = self.print_objects_dictionary[key]['nb']['objective'].get()
        self.print_objects_dictionary[key]['nb']['fov_z'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['fov_y'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['fov_x'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['gov'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['resolution_z'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['resolution_y'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['resolution_x'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['galvo_acceleration'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['piezo_settling_time'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['stage_velocity'].delete(0, tk.END)
        self.print_objects_dictionary[key]['nb']['scan_velocity'].delete(0, tk.END)
        if objective == '10x':
            self.print_objects_dictionary[key]['nb']['fov_z'].insert(0, 280)
            self.print_objects_dictionary[key]['nb']['fov_y'].insert(0, 1000)
            self.print_objects_dictionary[key]['nb']['fov_x'].insert(0, 1000)
            self.print_objects_dictionary[key]['nb']['gov'].insert(0, 1000)
            self.print_objects_dictionary[key]['nb']['resolution_z'].insert(0, 5)
            self.print_objects_dictionary[key]['nb']['resolution_y'].insert(0, 1)
            self.print_objects_dictionary[key]['nb']['resolution_x'].insert(0, 1)
            self.print_objects_dictionary[key]['nb']['galvo_acceleration'].insert(0, 1)
            self.print_objects_dictionary[key]['nb']['piezo_settling_time'].insert(0, 0.1)
            self.print_objects_dictionary[key]['nb']['stage_velocity'].insert(0, 100)
            self.print_objects_dictionary[key]['nb']['scan_velocity'].insert(0, 100_000)
        elif objective == '25x':
            self.print_objects_dictionary[key]['nb']['fov_z'].insert(0, 280)
            self.print_objects_dictionary[key]['nb']['fov_y'].insert(0, 400)
            self.print_objects_dictionary[key]['nb']['fov_x'].insert(0, 400)
            self.print_objects_dictionary[key]['nb']['gov'].insert(0, 400)
            self.print_objects_dictionary[key]['nb']['resolution_z'].insert(0, 1)
            self.print_objects_dictionary[key]['nb']['resolution_y'].insert(0, 0.5)
            self.print_objects_dictionary[key]['nb']['resolution_x'].insert(0, 0.5)
            self.print_objects_dictionary[key]['nb']['galvo_acceleration'].insert(0, 5)
            self.print_objects_dictionary[key]['nb']['piezo_settling_time'].insert(0, 0.1)
            self.print_objects_dictionary[key]['nb']['stage_velocity'].insert(0, 100)
            self.print_objects_dictionary[key]['nb']['scan_velocity'].insert(0, 100_000)
        elif objective == '63x':
            self.print_objects_dictionary[key]['nb']['fov_z'].insert(0, 280)
            self.print_objects_dictionary[key]['nb']['fov_y'].insert(0, 200)
            self.print_objects_dictionary[key]['nb']['fov_x'].insert(0, 200)
            self.print_objects_dictionary[key]['nb']['gov'].insert(0, 200)
            self.print_objects_dictionary[key]['nb']['resolution_z'].insert(0, 0.5)
            self.print_objects_dictionary[key]['nb']['resolution_y'].insert(0, 0.3)
            self.print_objects_dictionary[key]['nb']['resolution_x'].insert(0, 0.3)
            self.print_objects_dictionary[key]['nb']['galvo_acceleration'].insert(0, 5)
            self.print_objects_dictionary[key]['nb']['piezo_settling_time'].insert(0, 0.1)
            self.print_objects_dictionary[key]['nb']['stage_velocity'].insert(0, 100)
            self.print_objects_dictionary[key]['nb']['scan_velocity'].insert(0, 100_000)

    def init_print_object(self):
        """
        see stl2gwl for more complete definitions of the print object
        and the variables therein
        """
        print_object = dict(
            name='',
            key='',
            # print variables
            render_checkbox=tk.BooleanVar(),
            render_ghost_boolean=tk.BooleanVar(),
            render_power_checkbox=tk.BooleanVar(),
            render_time_checkbox=tk.BooleanVar(),
            render_line_width=None,
            render_line_opacity=10.5,
            render_line_color=20,

            illumination_function="gaussian",
            illumination_function_minus=10.0,

            # output_directory=Path(os.getcwd()) / 'gwl_output',
            output_directory=Path('~/Desktop').expanduser() / 'gwl_output',

            stl=Stl(filename=Path('../resources/stl.nosync/cube_100x100x100.STL'),
                      output_directory='',
                      rescale=[1.0, 1.0, 1.0],
                      rotate=[0, 1, 2]),

            # specify if stl unit cell
            unit_cell='gyroid',
            # unit_cell_stl=Stl(filename=Path('../resources/stl.nosync/cylinder_100diamx100.STL'),
            #                     output_directory='',
            #                     rescale=[0.1, 0.1, 0.1],
            #                     rotate=[0, 1, 2]),
            unit_cell_stl='',

            # specify properties of print
            resolution=[1.0, 0.5, 0.5],
            rotation_degrees='default',
            position_offset=[0, 0, 0],
            unit_cell_fill=0.2,
            power_scale=1.0,
            laser_power=100.0,
            interp_factor=1,
            unit_cell_inverse=False,
            unit_cell_period=10,
            tiling_strategy='checkerboard',
            gov=400,
            gov_overlap=4.0,
            fov=[100, 400, 400],
            fov_overlap=[2, 4, 4],
            objective='25x',
            galvo_acceleration=10,
            piezo_settling_time=0.1,
            stage_velocity=1_000,
            scan_velocity=100_000,
            depth_power_map_h=None,
            depth_power_map_p=None,
            depth_power_map_fittype="slinear",

            # holds tk notebook tab related to the print
            nb={},
        )
        print_object['stl'].output_directory = print_object['output_directory']

        return print_object

    def add_new_print(self):
        # Add a new tab for a new print
        new_tab = tk.Frame(self.right_frame, width=(self.w - self.h), height=self.h)
        new_tab.pack_propagate(0)  # Don't allow the widgets inside to dictate the frame's size

        if len(self.print_objects_dictionary) > 0:
            print_names = [v['name'] for v in self.print_objects_dictionary.values()]
            print_name_to_add = max(print_names) + 1
            for i in range(0, max(print_names)):
                if i not in print_names:
                    print_name_to_add = i
                    break
        else:
            print_name_to_add = 0
        self.right_frame.add(new_tab, text=f"Print {str(print_name_to_add + 1)}")
        key = str(print_name_to_add)
        p = self.init_print_object()
        p['name'] = print_name_to_add
        p['key'] = key
        p['new_tab'] = new_tab

        # check if its the first entry in self.print_objects_dictionary
        video_props_are_defined = False
        if any([v for v in self.print_objects_dictionary.values() if v['render_line_width'] is not None]):
            video_props_are_defined = True

        line_height = 20
        last_line_height = 35
        col_w = 50

        current_line_height_lhs = 30
        x_lhs = self.b = 20
        current_line_height_rhs = 30
        x_rhs = self.right_frame_width // 2 + self.b

        # Adding "Delete print" button inside new tab
        p['nb']['delete_print_button'] = tk.Button(new_tab, text="Delete print", command=lambda: self.delete_print(self.print_objects_dictionary[key]), bg='red', activebackground='red')
        p['nb']['delete_print_button'].place(x=self.right_frame_width - 4 * col_w, y=0)

        # Adding widgets to the new tab, that reflect the options available to the print object
        if not video_props_are_defined:
            p['nb']['render_checkbox'] = tk.Checkbutton(new_tab, text="Visualise print after rendering?", variable=p['render_checkbox'])
            p['nb']['render_checkbox'].place(x=x_lhs, y=current_line_height_lhs - current_line_height_lhs)

            p['nb']['render_ghost_boolean'] = tk.Checkbutton(new_tab, text="Render ghost object?", variable=p['render_ghost_boolean'])
            p['nb']['render_ghost_boolean'].place(x=x_lhs + col_w * 5, y=current_line_height_lhs - current_line_height_lhs)

            p['nb']['render_power_checkbox'] = tk.Checkbutton(new_tab, text="Render power as color", variable=p['render_power_checkbox'], command=lambda key=key: self.render_power_time_checked('render_power_checkbox'))
            p['nb']['render_power_checkbox'].place(x=x_lhs, y=current_line_height_lhs - 0.2 * current_line_height_lhs)

            p['nb']['render_time_checkbox'] = tk.Checkbutton(new_tab, text="Render time as color", variable=p['render_time_checkbox'], command=lambda key=key: self.render_power_time_checked('render_time_checkbox'))
            p['nb']['render_time_checkbox'].place(x=x_lhs + col_w * 5, y=current_line_height_lhs - 0.2 * current_line_height_lhs)

            current_line_height_lhs += line_height

            p['nb']['render_line_width_label'] = tk.Label(new_tab, text="Render line width")
            p['nb']['render_line_width_label'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += line_height
            p['nb']['render_line_width'] = tk.Entry(new_tab, width=12)
            p['nb']['render_line_width'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += last_line_height

            p['nb']['render_line_opacity_label'] = tk.Label(new_tab, text="Render line opacity (0 -> 1)")
            p['nb']['render_line_opacity_label'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += line_height
            p['nb']['render_line_opacity'] = tk.Entry(new_tab, width=12)
            p['nb']['render_line_opacity'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += last_line_height

            p['nb']['render_line_color_label'] = tk.Label(new_tab, text="Line color modifier (0 -> 1)")
            p['nb']['render_line_color_label'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += line_height
            p['nb']['render_line_color'] = tk.Entry(new_tab, width=12)
            p['nb']['render_line_color'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += last_line_height

            p['nb']['output_directory_label'] = tk.Label(new_tab, text="Output directory: ... path will show here ...")
            p['nb']['output_directory_label'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += line_height
            p['nb']['output_directory_button'] = tk.Button(new_tab, text="Select output directory", command=self.select_output_directory)
            p['nb']['output_directory_button'].place(x=x_lhs, y=current_line_height_lhs)
            current_line_height_lhs += last_line_height

            # set defaults
            p['render_line_width'] = 10  # default 10

            p['nb']['render_line_width'].insert(0, p['render_line_width'])
            p['render_line_opacity'] = 0.5
            p['nb']['render_line_opacity'].insert(0, p['render_line_opacity'])
            p['render_line_color'] = 1.0
            p['nb']['render_line_color'].insert(0, p['render_line_color'])

        else:
            self.render_line_width_label = tk.Label(new_tab, text="See Print 1 to define\noutput path &\nprint simulation\nrendering properties")
            self.render_line_width_label.place(x=x_lhs + col_w, y=current_line_height_lhs + col_w)
            current_line_height_lhs += line_height
            current_line_height_lhs += last_line_height
            current_line_height_lhs += line_height
            current_line_height_lhs += last_line_height
            current_line_height_lhs += line_height
            current_line_height_lhs += last_line_height
            current_line_height_lhs += line_height
            current_line_height_lhs += last_line_height

        p['nb']['stl_label'] = tk.Label(new_tab, text="STL file | ... path will show here ...")
        p['nb']['stl_label'].place(x=x_lhs, y=current_line_height_lhs)
        current_line_height_lhs += line_height
        p['nb']['stl_button'] = tk.Button(new_tab, text="Select STL file", command=lambda key=key: self.select_stl_file(key))
        p['nb']['stl_button'].place(x=x_lhs, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        current_line_height_lhs += last_line_height
        current_line_height_lhs += last_line_height

        p['nb']['power_scale_label'] = tk.Label(new_tab, text="Power scale (0.5 -> ~1.2)")
        p['nb']['power_scale_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['laser_power_label'] = tk.Label(new_tab, text="Laser power (1-100)")
        p['nb']['laser_power_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['power_depth_map_label'] = tk.Label(new_tab, text="Depth mapping of laser power [0 -> height µm], [0 -> 100]", fg="#1890FF")
        p['nb']['power_depth_map_label'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        current_line_height_lhs += line_height

        p['nb']['power_scale'] = tk.Entry(new_tab, width=4)
        p['nb']['power_scale'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['laser_power'] = tk.Entry(new_tab, width=4)
        p['nb']['laser_power'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['depth_power_map_h'] = tk.Entry(new_tab, width=16, fg="#1890FF")
        p['nb']['depth_power_map_h'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['depth_power_map_p'] = tk.Entry(new_tab, width=16, fg="#1890FF")
        p['nb']['depth_power_map_p'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        p['nb']['illumination_function_label'] = tk.Label(new_tab, text="Illumination function")
        p['nb']['illumination_function_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['illumination_minus_label'] = tk.Label(new_tab, text="Illumination subtraction (%)")
        p['nb']['illumination_minus_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['depth_power_map_fittype_label'] = tk.Label(new_tab, text="Fit type", fg="#1890FF")
        p['nb']['depth_power_map_fittype_label'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        current_line_height_lhs += line_height
        p['nb']['illumination_function'] = ttk.Combobox(new_tab, values=["gaussian", "none", "linear"], width=12)
        p['nb']['illumination_function'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['illumination_function_minus'] = tk.Entry(new_tab, width=4)
        p['nb']['illumination_function_minus'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        style_depth_map = ttk.Style()
        style_depth_map.configure('Custom.TCombobox', foreground='#1890FF')
        p['nb']['depth_power_map_fittype'] = ttk.Combobox(new_tab, values=["slinear", "cubic"], style='Custom.TCombobox', width=12)
        p['nb']['depth_power_map_fittype'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        p['nb']['tiling_strategy_label'] = tk.Label(new_tab, text="Tiling strategy")
        p['nb']['tiling_strategy_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['rotation_degrees_label'] = tk.Label(new_tab, text="∆θ raster (degrees), for each slice [0->180]")
        p['nb']['rotation_degrees_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['depth_power_map_expose'] = tk.Label(new_tab,
                                                     text="i.e. 50% power, 0-20 µm"
                                                          "\nthen 100% (object height 200 µm):"
                                                          "\n[0, 20, 20.01, 200], [50, 50, 100, 100]"
                                                          "\n50 -> 100% linear interpolation over 314 µm"
                                                          "\n[0, 314], [50, 100]"
                                                          "\n(overrides laser power but not power scale)",
                                                     justify='right', fg="#1890FF")
        p['nb']['depth_power_map_expose'].place(x=x_lhs + col_w * 10, y=current_line_height_lhs)
        current_line_height_lhs += line_height
        p['nb']['tiling_strategy'] = ttk.Combobox(new_tab, values=["xyz", "checkerboard"], width=12)
        p['nb']['tiling_strategy'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['rotation_degrees'] = tk.Entry(new_tab, width=4)
        p['nb']['rotation_degrees'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height
        p['nb']['interp_factor_label'] = tk.Label(new_tab,text="Interp factor (integer > 1)")
        p['nb']['interp_factor_label'].place(x=x_lhs, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height
        p['nb']['interp_factor'] = tk.Entry(new_tab, width=3)
        p['nb']['interp_factor'].place(x=x_lhs, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height
        current_line_height_lhs += last_line_height

        p['nb']['objective_label'] = tk.Label(new_tab, text="Objective")
        p['nb']['objective_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['objective_default_button_label'] = tk.Label(new_tab, text="set default values for objective")
        p['nb']['objective_default_button_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        current_line_height_lhs += line_height

        p['nb']['objective'] = ttk.Combobox(new_tab, values=["10x", "25x", "63x"], width=12)
        p['nb']['objective'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['objective_default_button'] = tk.Button(new_tab, text="Set default values", command=lambda key=key: self.set_defaults_for_objective(key))
        p['nb']['objective_default_button'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['youngs_mod_button'] = tk.Button(new_tab, text="set Young's Mod.", command=lambda key=key: self.set_youngs_mod(key))
        p['nb']['youngs_mod_button'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['youngs_mod'] = tk.Entry(new_tab, width=3)
        p['nb']['youngs_mod'].place(x=x_lhs + col_w * 11, y=current_line_height_lhs)
        p['nb']['youngs_mod_label'] = tk.Label(new_tab, text=" (kPa)")
        p['nb']['youngs_mod_label'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        p['nb']['fov_label'] = tk.Label(new_tab, text="FOV [z, y, x] (µm)")
        p['nb']['fov_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['fov_overlap_label'] = tk.Label(new_tab, text="FOV overlap [z, y, x] (µm)")
        p['nb']['fov_overlap_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['gov_label'] = tk.Label(new_tab, text="Galvo sub-field y-x (µm)")
        p['nb']['gov_label'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['gov_overlap_label'] = tk.Label(new_tab, text="Galvo overlap y-x (µm)")
        p['nb']['gov_overlap_label'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)
        current_line_height_lhs += line_height
        p['nb']['fov_z'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_z'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['fov_y'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_y'].place(x=x_lhs + col_w, y=current_line_height_lhs)
        p['nb']['fov_x'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_x'].place(x=x_lhs + col_w * 2, y=current_line_height_lhs)
        p['nb']['fov_overlap_z'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_overlap_z'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['fov_overlap_y'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_overlap_y'].place(x=x_lhs + col_w * 4 + col_w, y=current_line_height_lhs)
        p['nb']['fov_overlap_x'] = tk.Entry(new_tab, width=3)
        p['nb']['fov_overlap_x'].place(x=x_lhs + col_w * 4 + col_w * 2, y=current_line_height_lhs)
        p['nb']['gov'] = tk.Entry(new_tab, width=3)
        p['nb']['gov'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['gov_overlap'] = tk.Entry(new_tab, width=3)
        p['nb']['gov_overlap'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        p['nb']['rescale_label'] = tk.Label(new_tab, text="Rescale [z, y, x] (1 = 100%)")
        p['nb']['rescale_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['rotate_label'] = tk.Label(new_tab, text="Rotate [z, y, x] -> [0, 1, 2]")
        p['nb']['rotate_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['resolution_label'] = tk.Label(new_tab, text="Resolution [z, y, x] (µm)")
        p['nb']['resolution_label'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        current_line_height_lhs += line_height

        p['nb']['rescale_z'] = tk.Entry(new_tab, width=3)
        p['nb']['rescale_z'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['rescale_y'] = tk.Entry(new_tab, width=3)
        p['nb']['rescale_y'].place(x=x_lhs + col_w, y=current_line_height_lhs)
        p['nb']['rescale_x'] = tk.Entry(new_tab, width=3)
        p['nb']['rescale_x'].place(x=x_lhs + 2 * col_w, y=current_line_height_lhs)

        p['nb']['rotate_z'] = tk.Entry(new_tab, width=3)
        p['nb']['rotate_z'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['rotate_y'] = tk.Entry(new_tab, width=3)
        p['nb']['rotate_y'].place(x=x_lhs + col_w * 5, y=current_line_height_lhs)
        p['nb']['rotate_x'] = tk.Entry(new_tab, width=3)
        p['nb']['rotate_x'].place(x=x_lhs + col_w * 6, y=current_line_height_lhs)

        p['nb']['resolution_z'] = tk.Entry(new_tab, width=3)
        p['nb']['resolution_z'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['resolution_y'] = tk.Entry(new_tab, width=3)
        p['nb']['resolution_y'].place(x=x_lhs + col_w * 9, y=current_line_height_lhs)
        p['nb']['resolution_x'] = tk.Entry(new_tab, width=3)
        p['nb']['resolution_x'].place(x=x_lhs + col_w * 10, y=current_line_height_lhs)

        current_line_height_lhs += last_line_height

        p['nb']['position_offset_label'] = tk.Label(new_tab, text="Position offset [z, y, x] (µm)")
        p['nb']['position_offset_label'].place(x=x_lhs, y=current_line_height_lhs)
        current_line_height_lhs += line_height
        p['nb']['position_offset_z'] = tk.Entry(new_tab, width=6)
        p['nb']['position_offset_z'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['position_offset_y'] = tk.Entry(new_tab, width=6)
        p['nb']['position_offset_y'].place(x=x_lhs + col_w * 1.5, y=current_line_height_lhs)
        p['nb']['position_offset_x'] = tk.Entry(new_tab, width=6)
        p['nb']['position_offset_x'].place(x=x_lhs + col_w * 3.0, y=current_line_height_lhs)
        current_line_height_lhs += last_line_height

        p['nb']['galvo_acceleration_label'] = tk.Label(new_tab, text="Galvo acceleration (µm/s²)\n[1 -> 10]")
        p['nb']['galvo_acceleration_label'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['piezo_settling_time_label'] = tk.Label(new_tab, text="Piezo settling time (secs)")
        p['nb']['piezo_settling_time_label'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['stage_velocity_label'] = tk.Label(new_tab, text="Stage velocity (µm/s)\n[50 -> 1000]")
        p['nb']['stage_velocity_label'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['scan_velocity_label'] = tk.Label(new_tab, text="Scan velocity (µm/s)\n[10k -> 100k]")
        p['nb']['scan_velocity_label'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)

        current_line_height_lhs += line_height * 2
        p['nb']['galvo_acceleration'] = tk.Entry(new_tab, width=6)
        p['nb']['galvo_acceleration'].place(x=x_lhs, y=current_line_height_lhs)
        p['nb']['piezo_settling_time'] = tk.Entry(new_tab, width=6)
        p['nb']['piezo_settling_time'].place(x=x_lhs + col_w * 4, y=current_line_height_lhs)
        p['nb']['stage_velocity'] = tk.Entry(new_tab, width=6)
        p['nb']['stage_velocity'].place(x=x_lhs + col_w * 8, y=current_line_height_lhs)
        p['nb']['scan_velocity'] = tk.Entry(new_tab, width=6)
        p['nb']['scan_velocity'].place(x=x_lhs + col_w * 12, y=current_line_height_lhs)

        # Unit cell stuff
        p['nb']['unit_cell_label'] = tk.Label(new_tab, text="Unit cell")
        p['nb']['unit_cell_label'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += line_height
        p['nb']['unit_cell'] = ttk.Combobox(new_tab, values=["gyroid", "cubic", "octet truss", "kelvin foam", "weaire-phelan foam", "stl"], width=12)
        p['nb']['unit_cell'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += last_line_height

        p['nb']['unit_cell_stl_label'] = tk.Label(new_tab, text="STL unit cell file: ... path will show here ...")
        p['nb']['unit_cell_stl_label'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += line_height
        p['nb']['unit_cell_stl_button'] = tk.Button(new_tab, text="Select STL unit cell file", command=lambda key=key: self.select_unit_cell_stl_file(key))
        p['nb']['unit_cell_stl_button'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += last_line_height

        p['nb']['unit_cell_fill_label'] = tk.Label(new_tab, text="Unit cell fill [0 -> 1]")
        p['nb']['unit_cell_fill_label'].place(x=x_rhs, y=current_line_height_rhs)
        p['nb']['unit_cell_period_label'] = tk.Label(new_tab, text="period of unit cell (µm)")
        p['nb']['unit_cell_period_label'].place(x=x_rhs + col_w * 4, y=current_line_height_rhs)
        current_line_height_rhs += line_height
        p['nb']['unit_cell_fill'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_fill'].place(x=x_rhs, y=current_line_height_rhs)
        p['nb']['unit_cell_period'] = tk.Entry(new_tab, width=4)
        p['nb']['unit_cell_period'].place(x=x_rhs + col_w * 4, y=current_line_height_rhs)
        current_line_height_rhs += last_line_height

        p['nb']['unit_cell_inverse_label'] = tk.Label(new_tab, text="Unit cell inverse")
        p['nb']['unit_cell_inverse_label'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += line_height
        p['nb']['unit_cell_inverse_var'] = tk.BooleanVar()
        p['nb']['unit_cell_inverse'] = tk.Checkbutton(new_tab, variable=p['nb']['unit_cell_inverse_var'])
        p['nb']['unit_cell_inverse'].place(x=x_rhs, y=current_line_height_rhs)
        current_line_height_rhs += last_line_height

        p['nb']['unit_cell_rescale_label'] = tk.Label(new_tab, text="Rescale [z, y, x] (1 = 100%)")
        p['nb']['unit_cell_rescale_label'].place(x=x_rhs, y=current_line_height_rhs)
        p['nb']['unit_cell_rotate_label'] = tk.Label(new_tab, text="Rotate [z, y, x] -> [0, 1, 2]")
        p['nb']['unit_cell_rotate_label'].place(x=x_rhs + col_w * 4, y=current_line_height_rhs)
        current_line_height_rhs += line_height

        p['nb']['unit_cell_rescale_z'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rescale_z'].place(x=x_rhs, y=current_line_height_rhs)
        p['nb']['unit_cell_rescale_y'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rescale_y'].place(x=x_rhs + col_w, y=current_line_height_rhs)
        p['nb']['unit_cell_rescale_x'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rescale_x'].place(x=x_rhs + col_w * 2, y=current_line_height_rhs)
        p['nb']['unit_cell_rotate_z'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rotate_z'].place(x=x_rhs + col_w * 4, y=current_line_height_rhs)
        p['nb']['unit_cell_rotate_y'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rotate_y'].place(x=x_rhs + col_w * 5, y=current_line_height_rhs)
        p['nb']['unit_cell_rotate_x'] = tk.Entry(new_tab, width=3)
        p['nb']['unit_cell_rotate_x'].place(x=x_rhs + col_w * 6, y=current_line_height_rhs)
        current_line_height_rhs += last_line_height

        self.print_objects_dictionary[key] = p

        self.set_value_from_print_object(key, self.print_objects_dictionary[key])

    def set_value_from_print_object(self, key, po):
        # to be updated
        if key == '0':
            self.print_objects_dictionary[key]['nb']['output_directory_label'].config(text="Output directory | " + str(po['output_directory']))
        self.print_objects_dictionary[key]['nb']['stl_label'].config(text="STL file | " + str(po['stl'].filename))
        self.print_objects_dictionary[key]['nb']['power_scale'].insert(0, po['power_scale'])
        self.print_objects_dictionary[key]['nb']['laser_power'].insert(0, po['laser_power'])
        self.print_objects_dictionary[key]['nb']['interp_factor'].insert(0, po['interp_factor'])
        self.print_objects_dictionary[key]['nb']['tiling_strategy'].set(po['tiling_strategy'])
        self.print_objects_dictionary[key]['nb']['illumination_function'].set(po['illumination_function'])
        self.print_objects_dictionary[key]['nb']['illumination_function_minus'].insert(0, po['illumination_function_minus'])
        self.print_objects_dictionary[key]['nb']['objective'].set(po['objective'])
        self.print_objects_dictionary[key]['nb']['fov_z'].insert(0, po['fov'][0])
        self.print_objects_dictionary[key]['nb']['fov_y'].insert(0, po['fov'][1])
        self.print_objects_dictionary[key]['nb']['fov_x'].insert(0, po['fov'][2])
        self.print_objects_dictionary[key]['nb']['fov_overlap_z'].insert(0, po['fov_overlap'][0])
        self.print_objects_dictionary[key]['nb']['fov_overlap_y'].insert(0, po['fov_overlap'][1])
        self.print_objects_dictionary[key]['nb']['fov_overlap_x'].insert(0, po['fov_overlap'][2])
        self.print_objects_dictionary[key]['nb']['gov'].insert(0, po['gov'])
        self.print_objects_dictionary[key]['nb']['gov_overlap'].insert(0, po['gov_overlap'])
        self.print_objects_dictionary[key]['nb']['rescale_z'].insert(0, po['stl'].rescale[0])
        self.print_objects_dictionary[key]['nb']['rescale_y'].insert(0, po['stl'].rescale[1])
        self.print_objects_dictionary[key]['nb']['rescale_x'].insert(0, po['stl'].rescale[2])
        self.print_objects_dictionary[key]['nb']['rotate_z'].insert(0, po['stl'].rotate[0])
        self.print_objects_dictionary[key]['nb']['rotate_y'].insert(0, po['stl'].rotate[1])
        self.print_objects_dictionary[key]['nb']['rotate_x'].insert(0, po['stl'].rotate[2])
        self.print_objects_dictionary[key]['nb']['resolution_z'].insert(0, po['resolution'][0])
        self.print_objects_dictionary[key]['nb']['resolution_y'].insert(0, po['resolution'][1])
        self.print_objects_dictionary[key]['nb']['resolution_x'].insert(0, po['resolution'][2])
        self.print_objects_dictionary[key]['nb']['position_offset_z'].insert(0, po['position_offset'][0])
        self.print_objects_dictionary[key]['nb']['position_offset_y'].insert(0, po['position_offset'][1])
        self.print_objects_dictionary[key]['nb']['position_offset_x'].insert(0, po['position_offset'][2])
        self.print_objects_dictionary[key]['nb']['unit_cell'].set(po['unit_cell'])
        self.print_objects_dictionary[key]['nb']['depth_power_map_h'].insert(0, str(po['depth_power_map_h']))
        self.print_objects_dictionary[key]['nb']['depth_power_map_p'].insert(0, str(po['depth_power_map_p']))
        self.print_objects_dictionary[key]['nb']['depth_power_map_fittype'].set(po['depth_power_map_fittype'])
        if po['unit_cell_stl'] == '':
            self.print_objects_dictionary[key]['nb']['unit_cell_stl_label'].config(text='')
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_z'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_y'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_x'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_z'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_y'].insert(0, 'n/a')
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_x'].insert(0, 'n/a')
        else:
            self.print_objects_dictionary[key]['nb']['unit_cell_stl_label'].config(text="STL unit cell path | " + str(po['unit_cell_stl'].filename))
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_z'].insert(0, po['unit_cell_stl'].rescale[0])
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_y'].insert(0, po['unit_cell_stl'].rescale[1])
            self.print_objects_dictionary[key]['nb']['unit_cell_rescale_x'].insert(0, po['unit_cell_stl'].rescale[2])
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_z'].insert(0, po['unit_cell_stl'].rotate[0])
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_y'].insert(0, po['unit_cell_stl'].rotate[1])
            self.print_objects_dictionary[key]['nb']['unit_cell_rotate_x'].insert(0, po['unit_cell_stl'].rotate[2])
        self.print_objects_dictionary[key]['nb']['unit_cell_fill'].insert(0, po['unit_cell_fill'])
        self.print_objects_dictionary[key]['nb']['unit_cell_inverse_var'].set(po['unit_cell_inverse'])
        self.print_objects_dictionary[key]['nb']['unit_cell_period'].insert(0, po['unit_cell_period'])
        self.print_objects_dictionary[key]['nb']['galvo_acceleration'].insert(0, po['galvo_acceleration'])
        self.print_objects_dictionary[key]['nb']['piezo_settling_time'].insert(0, po['piezo_settling_time'])
        self.print_objects_dictionary[key]['nb']['stage_velocity'].insert(0, po['stage_velocity'])
        self.print_objects_dictionary[key]['nb']['scan_velocity'].insert(0, po['scan_velocity'])
        self.print_objects_dictionary[key]['nb']['rotation_degrees'].insert(0, po['rotation_degrees'])

    def update_values_from_gui(self, key, po):
        # collect from the gui
        if po['name'] == 0:
            po['render_line_width'] = self.print_objects_dictionary[key]['nb']['render_line_width'].get()
            po['render_line_opacity'] = self.print_objects_dictionary[key]['nb']['render_line_opacity'].get()
            po['render_line_color'] = self.print_objects_dictionary[key]['nb']['render_line_color'].get()
            po['output_directory'] = self.print_objects_dictionary[key]['nb']['output_directory_label'].cget("text")
            po['output_directory'] = po['output_directory'][po['output_directory'].find(" | ") + 3:]
        po['stl'].filename = self.print_objects_dictionary[key]['nb']['stl_label'].cget("text")
        po['stl'].filename = po['stl'].filename[po['stl'].filename.find(" | ") + 3:]
        po['power_scale'] = self.print_objects_dictionary[key]['nb']['power_scale'].get()
        po['laser_power'] = self.print_objects_dictionary[key]['nb']['laser_power'].get()
        po['illumination_function'] = self.print_objects_dictionary[key]['nb']['illumination_function'].get()
        po['illumination_function_minus'] = self.print_objects_dictionary[key]['nb']['illumination_function_minus'].get()
        po['depth_power_map_h'] = self.print_objects_dictionary[key]['nb']['depth_power_map_h'].get()
        po['depth_power_map_p'] = self.print_objects_dictionary[key]['nb']['depth_power_map_p'].get()
        if po['depth_power_map_h'] != 'None' and po['depth_power_map_p'] != 'None':
            po['depth_power_map_h'] = po['depth_power_map_h'].replace('[', '').replace(']', '')
            po['depth_power_map_h'] = [float(i) for i in po['depth_power_map_h'].split(',')]
            po['depth_power_map_p'] = po['depth_power_map_p'].replace('[', '').replace(']', '')
            po['depth_power_map_p'] = [float(i) for i in po['depth_power_map_p'].split(',')]
        po['depth_power_map_fittype'] = self.print_objects_dictionary[key]['nb']['depth_power_map_fittype'].get()
        po['interp_factor'] = self.print_objects_dictionary[key]['nb']['interp_factor'].get()
        po['tiling_strategy'] = self.print_objects_dictionary[key]['nb']['tiling_strategy'].get()
        po['objective'] = self.print_objects_dictionary[key]['nb']['objective'].get()
        po['fov'] = [self.print_objects_dictionary[key]['nb']['fov_z'].get(),
                     self.print_objects_dictionary[key]['nb']['fov_y'].get(),
                     self.print_objects_dictionary[key]['nb']['fov_x'].get()]
        po['gov'] = self.print_objects_dictionary[key]['nb']['gov'].get()
        po['fov_overlap'] = [self.print_objects_dictionary[key]['nb']['fov_overlap_z'].get(),
                             self.print_objects_dictionary[key]['nb']['fov_overlap_y'].get(),
                             self.print_objects_dictionary[key]['nb']['fov_overlap_x'].get()]
        po['gov_overlap'] = self.print_objects_dictionary[key]['nb']['gov_overlap'].get()
        po['stl'].rescale = [self.print_objects_dictionary[key]['nb']['rescale_z'].get(),
                             self.print_objects_dictionary[key]['nb']['rescale_y'].get(),
                             self.print_objects_dictionary[key]['nb']['rescale_x'].get()]
        po['stl'].rotate = [self.print_objects_dictionary[key]['nb']['rotate_z'].get(),
                            self.print_objects_dictionary[key]['nb']['rotate_y'].get(),
                            self.print_objects_dictionary[key]['nb']['rotate_x'].get()]
        po['resolution'] = [self.print_objects_dictionary[key]['nb']['resolution_z'].get(),
                            self.print_objects_dictionary[key]['nb']['resolution_y'].get(),
                            self.print_objects_dictionary[key]['nb']['resolution_x'].get()]
        po['position_offset'] = [self.print_objects_dictionary[key]['nb']['position_offset_z'].get(),
                                 self.print_objects_dictionary[key]['nb']['position_offset_y'].get(),
                                 self.print_objects_dictionary[key]['nb']['position_offset_x'].get()]
        po['unit_cell'] = self.print_objects_dictionary[key]['nb']['unit_cell'].get()
        po['unit_cell_inverse'] = self.print_objects_dictionary[key]['nb']['unit_cell_inverse_var'].get()
        po['unit_cell_period'] = self.print_objects_dictionary[key]['nb']['unit_cell_period'].get()
        po['unit_cell_fill'] = self.print_objects_dictionary[key]['nb']['unit_cell_fill'].get()
        if po['nb']['unit_cell_stl_label'].cget("text") != '':
            po['unit_cell_stl'] = Stl()
            po['unit_cell_stl'].filename = self.print_objects_dictionary[key]['nb']['unit_cell_stl_label'].cget("text")
            po['unit_cell_stl'].filename = po['unit_cell_stl'].filename[po['unit_cell_stl'].filename.find(" | ") + 3:]
            po['unit_cell_stl'].rescale = [self.print_objects_dictionary[key]['nb']['unit_cell_rescale_z'].get(),
                                          self.print_objects_dictionary[key]['nb']['unit_cell_rescale_y'].get(),
                                          self.print_objects_dictionary[key]['nb']['unit_cell_rescale_x'].get()]
            po['unit_cell_stl'].rotate = [self.print_objects_dictionary[key]['nb']['unit_cell_rotate_z'].get(),
                                          self.print_objects_dictionary[key]['nb']['unit_cell_rotate_y'].get(),
                                          self.print_objects_dictionary[key]['nb']['unit_cell_rotate_x'].get()]

            # if either rescale or rotate wern't set to a number then break and pop up a warning
            try:
                po['unit_cell_stl'].rescale = [float(_) for _ in po['unit_cell_stl'].rescale]
                po['unit_cell_stl'].rotate = [float(_) for _ in po['unit_cell_stl'].rotate]
            except ValueError:
                showerror("Error", "Unit cell rescale and rotate must be numbers")
                return
        else:
            po['unit_cell_fill'] = float(po['unit_cell_fill'])
            po['unit_cell_period'] = float(po['unit_cell_period'])
        po['galvo_acceleration'] = self.print_objects_dictionary[key]['nb']['galvo_acceleration'].get()
        po['piezo_settling_time'] = self.print_objects_dictionary[key]['nb']['piezo_settling_time'].get()
        po['stage_velocity'] = self.print_objects_dictionary[key]['nb']['stage_velocity'].get()

        po['rotation_degrees'] = self.print_objects_dictionary[key]['nb']['rotation_degrees'].get()
        if isinstance(po['rotation_degrees'], str):
            po['rotation_degrees'] = -1.0
        else:
            po['rotation_degrees'] = float(po['rotation_degrees'])

        # cast to the correct type
        if po['name'] == 0:
            po['render_line_width'] = float(po['render_line_width'])
            po['render_line_opacity'] = float(po['render_line_opacity'])
            po['render_line_color'] = float(po['render_line_color'])
            # po['output_directory'] = str
        # po['stl'].filename = str
        po['power_scale'] = float(po['power_scale'])
        po['laser_power'] = float(po['laser_power'])
        po['interp_factor'] = int(po['interp_factor'])
        # po['tiling_strategy'] = str
        # po['objective'] = str
        po['fov'] = [int(i) for i in po['fov']]
        po['fov_overlap'] = [float(i) for i in po['fov_overlap']]
        po['gov'] = int(po['gov'])
        po['gov_overlap'] = float(po['gov_overlap'])
        po['stl'].rescale = [float(i) for i in po['stl'].rescale]
        po['stl'].rotate = [int(i) for i in po['stl'].rotate]
        po['resolution'] = [float(i) for i in po['resolution']]
        po['position_offset'] = [float(i) for i in po['position_offset']]
        # po['unit_cell'] = str
        # po['unit_cell_stl'].filename = str
        if po['nb']['unit_cell_stl_label'].cget("text") != '':
            if not os.path.isfile(po['unit_cell_stl'].filename):
                po['unit_cell_fill'] = float(po['unit_cell_fill'])
                po['unit_cell_period'] = float(po['unit_cell_period'])
        # po['unit_cell_inverse'] = bool
        if po['unit_cell_stl'] != '':
            po['unit_cell_stl'].rescale = [float(i) for i in po['unit_cell_stl'].rescale]
            po['unit_cell_stl'].rotate = [float(i) for i in po['unit_cell_stl'].rotate]
        po['galvo_acceleration'] = float(po['galvo_acceleration'])
        po['piezo_settling_time'] = float(po['piezo_settling_time'])
        po['stage_velocity'] = float(po['stage_velocity'])
        po['illumination_function_minus'] = float(po['illumination_function_minus'])

        return po

    def delete_print(self, po):
        # Delete a print tab
        del self.print_objects_dictionary[str(po['name'])]
        self.right_frame.forget(po['new_tab'])

    def render_prints(self):

        print("Render prints")
        for key, po in self.print_objects_dictionary.items():
            self.print_objects_dictionary[key] = self.update_values_from_gui(key, po)

        # time how long it takes to run
        render_start_time = time.time()

        # video parameters which are only in the first print object
        render_line_color = [v['render_line_color'] for v in self.print_objects_dictionary.values()]

        if not any(render_line_color):
            # pop up dialog warming
            showerror("Error", "Print 1 doesn't seem to exist, nor does a video time,\nUsing default of a 20 second video")
            render_line_color = 0.1
        else:
            render_line_color = float(render_line_color[0])
            render_line_width = [v['render_line_width'] for v in self.print_objects_dictionary.values()][0]
            output_directory = Path([v['output_directory'] for v in self.print_objects_dictionary.values()][0])
            render_line_opacity = float([v['render_line_opacity'] for v in self.print_objects_dictionary.values()][0])

        # convert the dictionary to a Print object
        pr = Print(Stl(), '', PrintProp())
        print_objects = []
        for key, po in self.print_objects_dictionary.items():

            pr = Print(
                # specify stl
                stl=Stl(
                    filename=Path(po['stl'].filename),
                    output_directory=Path(output_directory),
                    rescale=po['stl'].rescale,
                    rotate=po['stl'].rotate
                ),

                # specify if stl unit cell
                unit_cell='',  # spec'd below

                # specify properties of print
                print_prop=PrintProp(
                    resolution=po['resolution'],
                    unit_cell=po['unit_cell'],  # gyroid, cubic, octet, kelvin, wp
                    unit_cell_fill=po['unit_cell_fill'],
                    power_scale=po['power_scale'],
                    laser_power=po['laser_power'],
                    depth_power_map=[po['depth_power_map_h'], po['depth_power_map_p']],
                    depth_power_map_fittype=po['depth_power_map_fittype'],
                    interp_factor=po['interp_factor'],
                    unit_cell_inverse=po['unit_cell_inverse'],
                    unit_cell_period=po['unit_cell_period'],
                    tiling_strategy=po['tiling_strategy'],  # None=xyz or Checkerboard(xy) -> z
                    gov_overlap=po['gov_overlap'],
                    gov=po['gov'],
                    fov_overlap=po['fov_overlap'],
                    fov=po['fov'],
                    scan_speed=po['scan_velocity'],
                    position_offset=po['position_offset'],
                    galvo_acceleration=po['galvo_acceleration'],
                    piezo_settling_time=po['piezo_settling_time'],
                    stage_velocity=po['piezo_settling_time'],
                    illumination_function=po['illumination_function'],
                    illumination_function_minus=po['illumination_function_minus'],
                    rotation_degrees=po['rotation_degrees'],
                )
            )

            if po['unit_cell_stl'] != '' and os.path.isfile(po['unit_cell_stl'].filename):
                pr.unit_cell = Stl(filename=Path(po['unit_cell_stl'].filename),
                                   output_directory='',
                                   rescale=po['unit_cell_stl'].rescale,
                                   rotate=po['unit_cell_stl'].rotate)

            print_objects.append(pr)

        stl_to_gwl(print_objects)  # <----- parallel or serial processing of govs

        print(f'\n\033[95m--- Writing the gwl took: {(time.time() - render_start_time) / 60:0.2f} mins ---\033[0m')

        if [v['render_checkbox'] for v in self.print_objects_dictionary.values()][0].get():
            plotter_start_time = time.time()
            render_power = [v['render_power_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_time = [v['render_time_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_boolean = [v['render_ghost_boolean'] for v in self.print_objects_dictionary.values()][0].get()

            # guard this in a new process to stop the crashing
            plot_process = Process(target=render_plots_process,
                                                   args=(output_directory,
                                                         render_line_opacity,
                                                         render_line_width,
                                                         render_line_color,
                                                         render_power,
                                                         render_time,
                                                         render_boolean,))
            plot_process.start()
            plot_process.join()
            time_taken = (time.time() - plotter_start_time) / 60
            if time_taken > 0.1:
                print(f'\n\033[95m--- Plotting the gwl took: {time_taken:0.2f} mins ---\033[0m')

        print("Done.")

    def visualise_rendered_print(self):

        if len(self.print_objects_dictionary) == 0:
            # assuming default output dictionary
            showerror("Error", "No print objects, assuming default output directory")
            self.add_new_print()
            render_line_color = float([v['render_line_color'] for v in self.print_objects_dictionary.values()][0])
            render_line_width = [v['render_line_width'] for v in self.print_objects_dictionary.values()][0]
            output_directory = Path([v['output_directory'] for v in self.print_objects_dictionary.values()][0])
            render_line_opacity = float([v['render_line_opacity'] for v in self.print_objects_dictionary.values()][0])
            render_power = [v['render_power_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_time = [v['render_time_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_boolean = [v['render_ghost_boolean'] for v in self.print_objects_dictionary.values()][0].get()
            self.delete_print(self.print_objects_dictionary['0'])
            self.root.update()  # ensure that the error window is gone
        else:
            # pull values if they exist
            for key, po in self.print_objects_dictionary.items():
                self.print_objects_dictionary[key] = self.update_values_from_gui(key, po)
            render_line_color = float([v['render_line_color'] for v in self.print_objects_dictionary.values()][0])
            render_line_width = [v['render_line_width'] for v in self.print_objects_dictionary.values()][0]
            output_directory = Path([v['output_directory'] for v in self.print_objects_dictionary.values()][0])
            render_line_opacity = float([v['render_line_opacity'] for v in self.print_objects_dictionary.values()][0])
            render_power = [v['render_power_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_time = [v['render_time_checkbox'] for v in self.print_objects_dictionary.values()][0].get()
            render_boolean = [v['render_ghost_boolean'] for v in self.print_objects_dictionary.values()][0].get()

        plotter_start_time = time.time()

        # guard this in a new process to stop the crashing
        plot_process = Process(target=render_plots_process,
                                               args=(output_directory,
                                                     render_line_opacity,
                                                     render_line_width,
                                                     render_line_color,
                                                     render_power,
                                                     render_time,
                                                     render_boolean))
        plot_process.start()
        plot_process.join()
        time_taken = (time.time() - plotter_start_time) / 60
        if time_taken > 0.1:
            print(f'\n\033[95m--- Plotting the gwl took: {time_taken:0.2f} mins ---\033[0m')

        print("Done.")
        pass


def render_plots_process(output_directory,
                         line_opacity,
                         line_width,
                         line_color,
                         render_power,
                         render_time,
                         render_boolean):
    plot.plot_lines_pyvista(path=output_directory,
                            opacity=line_opacity,
                            line_width=line_width,
                            line_color=line_color,
                            render_power=render_power,
                            render_time=render_time,
                            render_boolean=render_boolean)


if __name__ == "__main__":
    root = tk.Tk()
    app = PrinterApp(root)
    root.mainloop()
