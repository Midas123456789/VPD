import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import os

def hide_plots():
    return "NBGRADER_EXECUTION" in os.environ and (os.environ["NBGRADER_EXECUTION"] == "autograde" or os.environ["NBGRADER_EXECUTION"] == "validate")

def draw_circle_in_frame(frame, xy, radius, color):
    # Same function in numpy (100x faster)
    cx, cy = np.float32(xy)
    ys = np.linspace(0, 1, frame.shape[0])
    xs = np.linspace(0, 1, frame.shape[1])

    xdist2 = (xs - cx) ** 2
    ydist2 = (ys - cy) ** 2
    distances2 = xdist2.reshape((1, -1)) + ydist2.reshape((-1, 1))
    radius2 = radius * radius

    frame[distances2 < radius2] += color


def draw_rgb_circle_diagram(resolution):
    if hide_plots():
        return

    fig, ax = plt.subplots()
    ax.axis("off")
    stencil = np.zeros((resolution, resolution), dtype=np.int32)
    frame = np.zeros((resolution, resolution, 3), dtype=np.float32)
    image = ax.imshow(np.zeros((resolution, resolution, 3), dtype=np.float32))

    red_slider = widgets.FloatSlider(
        value=1.0, min=0, max=1, description="Red")
    green_slider = widgets.FloatSlider(
        value=1.0, min=0, max=1, description="Green")
    blue_slider = widgets.FloatSlider(
        value=1.0, min=0, max=1, description="Blue")
    overlap_only_button = widgets.Checkbox(
        False, description="Only show overlap")

    @interact(red=red_slider, green=green_slider, blue=blue_slider, overlap=overlap_only_button)
    def draw(red, green, blue, overlap):
        frame[:, :] = (0, 0, 0)
        draw_circle_in_frame(frame, (1/3, 1/3), 0.3, (red, 0.0, 0.0))
        draw_circle_in_frame(frame, (2/3, 1/3), 0.3, (0.0, green, 0.0))
        draw_circle_in_frame(frame, (1/2, 2/3), 0.3, (0.0, 0.0, blue))

        if overlap:
            stencil[:, :] = 0
            draw_circle_in_frame(stencil, (1/3, 1/3), 0.3, 1)
            draw_circle_in_frame(stencil, (2/3, 1/3), 0.3, 1)
            draw_circle_in_frame(stencil, (1/2, 2/3), 0.3, 1)
            frame[stencil != 3] = (0, 0, 0)

        image.set_data(frame)
        fig.canvas.draw_idle()

    plt.show()
