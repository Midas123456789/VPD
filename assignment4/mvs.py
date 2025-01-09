import helpers
from ipywidgets import Dropdown, interact, IntSlider
import matplotlib.patches as patches
from matplotlib.ticker import StrMethodFormatter
from numba import njit, jit, prange
import numpy as np
import glm
import os
import random
import matplotlib.pyplot as plt
import sys
import plotly.offline as py
import plotly.graph_objs as go
import re
import parallel
import functools
import progressbar
from collections import namedtuple
sys.path.append("../../")

Camera = namedtuple(
    "Camera", ["pos", "world_to_screen", "screen_to_world"])

def in_grading_mode():
    result = os.environ.get('NBGRADER_EXECUTION')
    return result == "autograde" or result == "validate"

def project_point(camera, point3D):
    return camera.world_to_screen(point3D)

def project_points(camera, points3D):
    return [camera.world_to_screen(p) for p in points3D]

@njit
def bresenham(p0, p1):  # p0 = input camera, p1 =
    x0, y0 = p0
    x1, y1 = p1

    pixels = []
    if x0 == x1 and y0 == y1:
        return pixels  # Returning [] does not compile because Numba can't infer type

    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) > abs(dy):
        derr = abs(dy / dx)
        error = 0.0
        y = y0
        for x in range(x0, x1, np.sign(dx)):
            pixels.append((x, y))
            error += derr
            if error >= 0.5:
                y += np.sign(dy)
                error -= 1
    else:
        derr = abs(dx / dy)
        error = 0.0
        x = x0
        for y in range(y0, y1, np.sign(dy)):
            pixels.append((x, y))
            error += derr
            if error >= 0.5:
                x += np.sign(dx)
                error -= 1
    return pixels

def snap_line_to_screen(p0, p1, resolution):
    height, width = resolution
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    tx0 = -p0.x / dx  # x = p0 + t * dx = 0 => t = -p / dx
    tx1 = (width-1-p0.x) / dx  # x = p0 + t * dx = width - 1 => t = (width-1-p0) / dx
    txmin = min(tx0, tx1)
    txmax = max(tx0, tx1)
    ty0 = -p0.y / dy
    ty1 = (height-1-p0.x) / dy
    tymin = min(ty0, ty1)
    tymax = max(ty0, ty1)
    tmin = max(txmin, tymin)
    tmax = min(txmax, tymax)

    x0 = p0.x + tmin * dx
    y0 = p0.y + tmin * dy
    x1 = p0.x + tmax * dx
    y1 = p0.y + tmax * dy
    return (glm.ivec2(x0+0.5, y0+0.5), glm.ivec2(x1+0.5, y1+0.5))

def draw_line_through_pixels(image, p0, p1, color):
    height, width, _ = image.shape

    result = image.copy()
    edge_p0, edge_p1 = snap_line_to_screen(p0, p1, (height, width)) # Pixels through line at edge of screen
    for x, y in bresenham(np.array(edge_p0), np.array(edge_p1)):
        result[y, x, :] = color
    return result

def project_ray_to_visible_line_segment(camera, ray, resolution):
    origin, direction = ray
    p0 = project_point(camera, origin)
    p1 = project_point(camera, origin + direction)
    return snap_line_to_screen(p0, p1, resolution)

# Asume that pixel0 and pixel1 are inside the image but the support domain might not be. Image0 and image1 are assumed
# to be of the same size.
def get_support_domains(image0, image1, pixel0, pixel1, s):
    x0, y0 = pixel0
    x1, y1 = pixel1
    h, w = image0.shape[:2]

    sx_min = min(s, x0, x1)
    sy_min = min(s, y0, y1)
    sx_max = min(s, w-1-x0, w-1-x1)
    sy_max = min(s, h-1-y0, h-1-y1)

    support0 = image0[y0-sy_min:y0+sy_max+1, x0-sx_min:x0+sx_max+1]
    support1 = image1[y1-sy_min:y1+sy_max+1, x1-sx_min:x1+sx_max+1]
    return (support0, support1)


@njit
def get_support_domains_no_bounds_check(image0, image1, pixel0, pixel1, s):
    x0, y0 = pixel0
    x1, y1 = pixel1
    support0 = image0[y0-s:y0+s+1, x0-s:x0+s+1]
    support1 = image1[y1-s:y1+s+1, x1-s:x1+s+1]
    return (support0, support1)


def pixel_to_ray(camera, pixel):
    world_space_coordinate = camera.screen_to_world(pixel, 1)
    direction = glm.normalize(world_space_coordinate - camera.pos)
    return (camera.pos, direction)


def compute_photo_consistency_along_ray(image0, image1, camera0, camera1, pixel0, photo_consistency_function, half_support_domain):
    # Estimate the depth of the ray through image0 at the given pixel by comparing the photoconsistency
    # along the viewing ray to image1
    height, width = image0.shape[:2]

    # Project 3D ray (from camera0) onto image1
    ray = pixel_to_ray(camera0, pixel0)  # 3D ray
    p0, p1 = project_ray_to_visible_line_segment(camera1, ray, (height, width))

    if pixel0.x < half_support_domain or pixel0.x >= width-half_support_domain or pixel0.y < half_support_domain or pixel0.y >= height-half_support_domain:
        print("WARNING: input pixel to close to image border")
        return (np.empty((0, 3), np.int32), np.empty((0), np.float32))

    pixels = []
    photo_consistencies = []
    for pixel1 in bresenham((p0.x, p0.y), (p1.x, p1.y)):
        x, y = pixel1
        if x < half_support_domain or x >= width - half_support_domain or y < half_support_domain or y >= height - half_support_domain:
            continue

        support0, support1 = get_support_domains_no_bounds_check(
            image0, image1, glm.ivec2(pixel0), pixel1, half_support_domain)
        photo_consistency = photo_consistency_function(support0, support1)

        pixels.append(glm.ivec2(x, y))
        photo_consistencies.append(photo_consistency)

    if len(pixels) > 0:
        return (pixels, np.array(photo_consistencies))
    else:
        return ([], np.empty((0)))

@njit
def compute_photo_consistency_along_ray_jit_impl_part1(p0, p1, width, height, half_support_domain):
    n = 0
    for pixel1 in bresenham(p0, p1):
        x, y = pixel1
        if x < half_support_domain or x >= width - half_support_domain or y < half_support_domain or y >= height - half_support_domain:
            continue
        n += 1
    return n

    pixels = np.empty((n, 2), np.float32)
    photo_consistencies = np.empty(n, np.float32)

@njit
def compute_photo_consistency_along_ray_jit_impl_part2(p0, p1, image0, image1, pixel0, photo_consistency_function, half_support_domain, out_pixels, out_photo_consistencies):
    height, width = image0.shape[:2]
    n = 0
    for pixel1 in bresenham(p0, p1):
        x, y = pixel1
        if x < half_support_domain or x >= width - half_support_domain or y < half_support_domain or y >= height - half_support_domain:
            continue

        support0, support1 = get_support_domains_no_bounds_check(
            image0, image1, pixel0, pixel1, half_support_domain)
        photo_consistency = photo_consistency_function(support0, support1)

        out_pixels[n, 0] = x
        out_pixels[n, 1] = y
        out_photo_consistencies[n] = photo_consistency
        n += 1

def compute_photo_consistency_along_ray_jit(image0, image1, camera0, camera1, pixel0, photo_consistency_function, half_support_domain):
    # Estimate the depth of the ray through image0 at the given pixel by comparing the photoconsistency
    # along the viewing ray to image1
    height, width = image0.shape[:2]

    # Project 3D ray (from camera0) onto image1
    ray = pixel_to_ray(camera0, pixel0)  # 3D ray
    p0, p1 = project_ray_to_visible_line_segment(camera1, ray, (height, width))

    if pixel0.x < half_support_domain or pixel0.x >= width-half_support_domain or pixel0.y < half_support_domain or pixel0.y >= height-half_support_domain:
        print("WARNING: input pixel to close to image border")
        return (np.empty((0, 3), np.int32), np.empty((0), np.float32))

    n = compute_photo_consistency_along_ray_jit_impl_part1(p0, p1, width, height, half_support_domain)
    pixels = np.empty((n, 2), np.float32)
    photo_consistencies = np.empty(n, np.float32)
    if n > 0:
        compute_photo_consistency_along_ray_jit_impl_part2(p0, p1, image0, image1, pixel0, photo_consistency_function, half_support_domain, pixels, photo_consistencies)
    return [glm.vec2(xy) for xy in pixels], photo_consistencies

def draw_square(ax, center, size, color, linewidth=1):
    rect = patches.Rectangle((center.x-size, center.y-size), 2*size+1, 2*size+1, linewidth=linewidth, edgecolor=color, facecolor="none")
    ax.add_patch(rect)

def plot_photoconsistency_along_ray(image0, image1, camera0, camera1, points2D_0, points3D, pc_functions, default_pid = 0):
    if in_grading_mode(): # Skip when grading
        return

    pc_name = Dropdown(options=["SSD", "SAD", "NCC"], description="Photo Consistency Measure")
    sd_size = IntSlider(min=3, max=31, value=15, step=2, continuous_update=False, description="Support Domain (x2)")
    pid = IntSlider(min=0, max=len(points3D)-1, value=default_pid, continuous_update=False, description="3D Point")

    fig2, ((support0_ax, support1_ax), (image0_ax, image1_ax)) = plt.subplots(2, 2, figsize=(helpers.default_fig_size[0], 2*helpers.default_fig_size[1]))
    fig1, plot_ax = plt.subplots(1, 1, figsize=helpers.default_fig_size)
    axes = [support0_ax, support1_ax, image0_ax, image1_ax, plot_ax]

    height, width = image0.shape[:2]

    @interact(pc_name=pc_name, sd_size=sd_size, pid=pid)
    def draw(pc_name, sd_size, pid):
        pc_func = pc_functions[pc_name]
        half_support_domain = sd_size // 2

        # Project points to 2D
        point3D = points3D[pid]
        pixel0 = points2D_0[pid]
        pixel1 = project_point(camera1, point3D)

        if pixel0.x < half_support_domain or pixel0.x >= width-half_support_domain or pixel0.y < half_support_domain or pixel0.y >= height-half_support_domain:
            print("ERROR: Input pixel too close to image border")
            return

        # Shoot a ray from camera0 through the pixel. Walk along it in image space (of image1) and compute photo consistency.
        pixels, photo_consistency = compute_photo_consistency_along_ray(
            image0, image1, camera0, camera1, pixel0, pc_func, half_support_domain)

        if len(pixels) == 0:
            print("ERROR: Epipolar line not in image")
            return

        # Compute where along the ray the point should actually be
        pixel1_pos_along_ray = np.argmin(
            np.array([helpers.SSD(glm.vec2(pixel), pixel1) for pixel in pixels]))

        print(f"Pixel with max photo consistency: {np.argmax(photo_consistency)}")
        print(f"Correct answer: {pixel1_pos_along_ray}")
        [ax.clear() for ax in axes]

        # Show the support domains at the input (image 0) and output (image 1) pixels.
        best_pixel_index = np.argmax(photo_consistency)
        support0, support1 = get_support_domains(
            image0, image1, glm.ivec2(pixel0), pixels[best_pixel_index], half_support_domain)
        print(f"Photo consistency at pixel {best_pixel_index} (support domain shown below): ", pc_func(
            support0, support1))
        if len(image0.shape) == 2:
            support0_ax.imshow(support0, cmap="gray")
            support1_ax.imshow(support1, cmap="gray")
        else:
            support0_ax.imshow(support0)
            support1_ax.imshow(support1)

        # Plot the photo consistency value along the epipolar line.
        plot_ax.axvline(pixel1_pos_along_ray, c="red")
        plot_ax.axvline(best_pixel_index, c="green")
        plot_ax.plot(photo_consistency)

        # Show the input and output image and the support domains
        image0_ax.axis("off")
        image1_ax.axis("off")
        if len(image0.shape) == 2:
            image0_ax.imshow(image0, cmap="gray")
            image1_ax.imshow(image1, cmap="gray")
        else:
            image0_ax.imshow(image0)
            image1_ax.imshow(image1)
        image0_ax.scatter([p.x for p in points2D_0], [p.y for p in points2D_0], s=2, c="orange", alpha=0.5)
        draw_square(image0_ax, pixel0, half_support_domain, "cyan", 2)
        draw_square(image1_ax, pixel1, half_support_domain, "cyan", 2)

        # Reset x/y axis
        plot_ax.relim()
        plot_ax.autoscale()
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()

    plt.show()

def plot_photoconsistency_accuracy(image0, image1, camera0, camera1, points2D_0, points2D_1, points3D, pc_functions):
    if in_grading_mode(): # Skip when grading
        return

    pc_name = Dropdown(options=["SSD", "SAD", "NCC"], description="Photo Consistency Measure")
    sd_size = IntSlider(min=3, max=31, value=15, step=2, continuous_update=False, description="Support Domain")

    fig, ax = plt.subplots(1, 1, figsize=helpers.default_fig_size)

    @interact(pc_name=pc_name, sd_size=sd_size)
    def draw(pc_name, sd_size):
        pc_func = pc_functions[pc_name]
        half_support_domain = sd_size // 2
        
        errors_in_pixels = []
        for pixel0, pixel1 in progressbar.progressbar(list(zip(points2D_0, points2D_1))):
            height, width = image0.shape[:2]
            x, y = pixel0
            if x < half_support_domain or x >= width - half_support_domain or y < half_support_domain or y >= height - half_support_domain:
                continue    # Walk along the ray and compute the photo consistency at each pixel
            
            pixels, photo_consistency = compute_photo_consistency_along_ray(
                image0, image1, camera0, camera1, pixel0, pc_func, half_support_domain)
            
            if len(pixels) == 0:
                continue
            
            # Compute where along the ray the point should actually be
            pixels = np.array(pixels)
            pixel1_pos_along_ray = np.argmin(np.sum((pixels - pixel1)**2, axis=1)) # Closest pixel = minimize SSD
            best_pixel_index = np.argmax(photo_consistency)
            errors_in_pixels.append(abs(pixel1_pos_along_ray - best_pixel_index))

        print("Number of correct matches (<3): ", np.count_nonzero(np.array(errors_in_pixels) < 3), " out of ", len(errors_in_pixels))

        x = np.linspace(0, 100, 100)
        y = np.percentile(errors_in_pixels, x)

        ax.clear()
        ax.set_title("Distance in image between highest photo consistency pixel and correct pixel")
        ax.plot(x, y)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Distance (pixels)")
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}%'))

    plt.show()


def local_maxima(signal):
    result = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            result.append(signal[i])
    return result

def normalize(signal):
    min_signal = np.min(signal)
    return (signal - min_signal) / (np.max(signal) - min_signal)


@njit
def compute_photo_consistency_along_ray_max_pixel_jit(p0, p1, image0, image1, pixel0, photo_consistency_function, half_support_domain):
    height, width = image0.shape[:2]
    best_photo_consistency = -10000000.0
    best_pixel_x = best_pixel_y = 0
    for pixel1 in bresenham(p0, p1):
        x, y = pixel1
        if x < half_support_domain or x >= width - half_support_domain or y < half_support_domain or y >= height - half_support_domain:
            continue

        support0, support1 = get_support_domains_no_bounds_check(
            image0, image1, pixel0, pixel1, half_support_domain)
        photo_consistency = photo_consistency_function(support0, support1)

        if photo_consistency > best_photo_consistency:
            best_pixel_x = x
            best_pixel_y = y
            best_photo_consistency = photo_consistency

    return (best_pixel_x, best_pixel_y)

@njit(parallel=True)
def compute_photo_consistency_along_ray_max_pixel_line_jit(min_x, max_x, y, in_line_segments, out_matching_pixels, image0, image1, photo_consistency_function, half_support_domain):
    # Estimate the depth of the ray through image0 at the given pixel by comparing the photoconsistency
    # along the viewing ray to image1
    height, width = image0.shape[:2]

    for x in prange(min_x, max_x):
        pixel0 = (x, y)
        p0x, p0y, p1x, p1y = in_line_segments[y, x]
        p0 = (p0x, p0y)
        p1 = (p1x, p1y)
        out_matching_pixels[y, x] = compute_photo_consistency_along_ray_max_pixel_jit(p0, p1, image0, image1, pixel0, photo_consistency_function, half_support_domain)

def compute_point_cloud(image0, image1, camera0, camera1, half_support_domain, pc_func, closest_point_between_rays, min_dist, max_dist):
    if in_grading_mode(): # Skip when grading
        return
    
    assert(image0.shape == image1.shape)
    height, width = image0.shape[:2]
    assert(image0.shape == image1.shape)
    margin = 3 * half_support_domain

    line_segments = np.empty((height, width, 4))
    for y in progressbar.progressbar(range(margin, height - margin)):
        for x in range(margin, width - margin):
            pixel0 = glm.ivec2(x, y)
            # Project 3D ray (from camera0) onto image1
            ray = pixel_to_ray(camera0, pixel0)  # 3D ray
            p0, p1 = project_ray_to_visible_line_segment(camera1, ray, (height, width))
            line_segments[y, x] = [p0.x, p0.y, p1.x, p1.y]

    matching_pixels = np.empty((height, width, 2))
    for y in progressbar.progressbar(range(margin, height - margin)):
        compute_photo_consistency_along_ray_max_pixel_line_jit(margin, width-margin, y, line_segments, matching_pixels, image0, image1, pc_func, half_support_domain)
    #    for x in range(margin, width - margin):
    #        pixel0 = (x, y)
    #        p0x, p0y, p1x, p1y = line_segments[y, x]
    #        p0 = (p0x, p0y)
    #        p1 = (p1x, p1y)
    #        matching_pixels[y, x] = compute_photo_consistency_along_ray_max_pixel_jit(p0, p1, image0, image1, pixel0, pc_func, half_support_domain)

    point3D_image = np.zeros((height, width, 3))
    for y in progressbar.progressbar(range(margin, height - margin)):
        for x in range(margin, width - margin):
            pixel0 = glm.ivec2(x, y)
            pixel1 = glm.ivec2(matching_pixels[y, x])
            ray0 = pixel_to_ray(camera0, pixel0)
            ray1 = pixel_to_ray(camera1, pixel1)
            point3D = closest_point_between_rays(ray0, ray1)

            # Discard points that lie behind the cameras
            if glm.dot(point3D - camera0.pos, ray0[1]) < 0 or glm.dot(point3D - camera1.pos, ray1[1]) < 0:
                continue

            # Discard obviously wrong rays (behind the camera or very far away)
            dist = glm.length(point3D - camera0.pos)
            if dist < min_dist or dist > max_dist:
                continue

            point3D_image[y, x] = point3D

    return point3D_image

# Computes the photo consistency along the epipolar line in image1 of the ray through pixel0 (of camera0).
# The epipolar line is colored using gray scale according the photo consistency measure of each pixel.
def draw_epipolar_photo_consistency(image0, image1, camera0, camera1, pixel0, pc_func, half_support_domain):
    out_image = image1.copy()
    
    pixel1s, photo_consistencies = compute_photo_consistency_along_ray(image0, image1, camera0, camera1, pixel0, pc_func, half_support_domain)

    min_pc = np.min(photo_consistencies)
    max_pc = np.max(photo_consistencies)
    for pixel1, photo_consistency in zip(pixel1s, photo_consistencies):
        photo_consistency = ((photo_consistency - min_pc) / (max_pc - min_pc)) ** 4
        # Draw a slightly thicker (vertically) line
        pixel1 = glm.ivec2(pixel1)
        out_image[pixel1.y-1, pixel1.x] = photo_consistency
        out_image[pixel1.y+0, pixel1.x] = photo_consistency
        out_image[pixel1.y+1, pixel1.x] = photo_consistency
    
    # Return the pixel with the highest photo consistency
    return out_image, pixel1s[np.argmax(photo_consistencies)]


def plot_epipolar_line_with_photoconsistency(image0, image1, camera0, camera1, min_dist, max_dist, pc_functions, closest_point_between_rays):
    if in_grading_mode(): # Skip when grading
        return

    pc_name = Dropdown(options=["SSD", "SAD", "NCC"], description="Photo Consistency Measure")
    sd_size = IntSlider(min=3, max=31, value=15, step=2, continuous_update=False, description="Support Domain (x2)")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    output_image = image1.copy()
    ax1.imshow(image0)
    ax2_imshow = ax2.imshow(output_image)
    ax1.axis("off")
    ax2.axis("off")

    ax1_scatter = ax2_scatter = None

    @interact(pc_name=pc_name, sd_size=sd_size)
    def draw(pc_name, sd_size):
        half_support_domain = sd_size // 2
        pc_func = pc_functions[pc_name]
        def onclick(pc_func, event):
            # === User clicked on the image ===
            nonlocal output_image, ax1_scatter, ax2_scatter
            pixel0 = glm.vec2(int(event.xdata), int(event.ydata))
            
            # Compute the epipolar line and draw it onto image 1
            output_image, pixel1 = draw_epipolar_photo_consistency(image0, image1, camera0, camera1, pixel0, pc_func, half_support_domain)
            
            # Shoot a ray through the clicked pixel and the matching pixel (with the highest photo consistency)
            ray0 = pixel_to_ray(camera0, pixel0)
            ray1 = pixel_to_ray(camera1, pixel1)
            # Find closest point between the two rays
            point3D = closest_point_between_rays(ray0, ray1)
            
            # Ignore matches which are clearly wrong (triangulated point extremely close/far from the camera)
            dist = np.linalg.norm(camera0.pos - point3D)
            if dist < min_dist or dist > max_dist:
                return
            
            if pixel0:
                if ax1_scatter:
                    ax1_scatter.remove()
                ax1_scatter = ax1.scatter([pixel0.x], [pixel0.y], c="r")
            if pixel1:
                if ax2_scatter:
                    ax2_scatter.remove()
                ax2_scatter = ax2.scatter([pixel1.x], [pixel1.y], c="r")

            ax2_imshow.set_data(output_image)
            fig.canvas.draw_idle()

        fig.tight_layout()
        fig.canvas.mpl_disconnect("button_press_event")
        fig.canvas.mpl_connect("button_press_event", functools.partial(onclick, pc_func))
    
        from collections import namedtuple
        Event = namedtuple("Event", ["xdata", "ydata"])
        ev = Event(100, 100)
        tmp = functools.partial(onclick, pc_func)
        tmp(ev)



def plotly_points(points, size=2, color=None, label="Points"):
    # 3D plot
    if color is None:
         color = np.random.rand(len(points))
    elif isinstance(color, list) or isinstance(color, np.ndarray):
        color = [f"rgb({255*r},{255*g},{255*b})" for r, g, b in color]
        
    points = np.array(points)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color, # set color to an array/list of desired values
            colorscale='Viridis', # choose a colorscale
            opacity=1.0
        ),
        name=label
    )
    
def plotly_point(point, **kwargs):
    return plotly_points(np.array([point]), **kwargs)

def plotly_camera(camera, size=8, label="Camera", **kwargs):
    return plotly_point(camera[0], size=size, label=label, **kwargs)

def plotly_frustum(camera, resolution):
    h, w = resolution
    p0 = camera[0]
    _, d0 = pixel_to_ray(camera, glm.vec2(0, 0))
    _, d1 = pixel_to_ray(camera, glm.vec2(w, 0))
    _, d2 = pixel_to_ray(camera, glm.vec2(w, h))
    _, d3 = pixel_to_ray(camera, glm.vec2(0, h))
    
    dist = 30
    vertices = np.array([p0, p0 + dist*d0, p0 + dist*d1, p0 + dist*d2, p0 + dist*d3])
    indices = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    # In graphics z = depth, when plotting z = height
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=indices[:,0],
        j=indices[:,1],
        k=indices[:,2],
        facecolor=["red", "green", "blue", "yellow"],
        opacity=0.2
    )
    
def plotly_epipolar_plane(camera0, camera1, points):
    camera0_pos = camera0[0]
    camera1_pos = camera1[0]
    
    dist = 5
    vertices = np.append(np.array([camera0_pos, camera1_pos]), points, axis=0)
    indices = np.array([[0, 1, i+2] for i in range(len(points))])
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=indices[:,0],
        j=indices[:,1],
        k=indices[:,2],
        facecolor=["red"],
        opacity=1.0
    )

def plot_cameras_3D(image0, camera0, camera1, points3D):
    if in_grading_mode(): # Skip when grading
        return

    # Project points onto image and get color from pixel
    #points2D_0 = project_points(camera0, points3D)
    #colors = []
    #for x, y in points2D_0:
    #    colors.append(image0[helpers.round_to_int(y), helpers.round_to_int(x)])
    colors = "orange"

    trace1 = plotly_points(points3D, color=colors)
    trace2 = plotly_camera(camera0, size=5, color="red", label="Camera 0")
    trace3 = plotly_frustum(camera0, image0.shape[:2])
    trace4 = plotly_camera(camera1, size=5, color="blue", label="Camera 1")
    trace5 = plotly_frustum(camera1, image0.shape[:2])
    trace6 = plotly_epipolar_plane(camera0, camera1, points3D[:1])
    data = [trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0}, scene={"aspectmode": "data"})
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3d-scatter-colorscale')


def draw_epipolar_line2(image1, ax2, camera0, camera1, pixel0, color):
    # Create a ray going through pixel0 of camera0
    ray = pixel_to_ray(camera0, pixel0)
    origin, direction = ray
    
    # Take two arbitrary points on the ray and transform them into the second image
    p0 = project_point(camera1, origin + 10 * direction)
    p1 = project_point(camera1, origin + 40 * direction)

    # Draw the two points that we used in the right image
    #ax2.scatter([p0[0], p1[0]], [p0[1], p1[1]], c=color.reshape(1,3))
    
    # Draw a line through the points in the right image
    return draw_line_through_pixels(image1, p0, p1, color)


def plot_interactive_epipolar_lines(image0, image1, camera0, camera1):
    if in_grading_mode(): # Skip when grading
        return

    output_image = image1.copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis("off")
    ax2.axis("off")
    ax1.imshow(image0)
    ax2_imshow = ax2.imshow(output_image)
    
    def onclick(event):
        nonlocal output_image
                
        color = np.random.rand((3)).astype(np.float32)
        pixel = glm.vec2(event.xdata, event.ydata)
        
        output_image = draw_epipolar_line2(output_image, ax2, camera0, camera1, pixel, color)
        ax2_imshow.set_data(output_image)
        ax1.scatter([pixel.x], [pixel.y], c=[color])
    
    fig.tight_layout()
    fig.canvas.mpl_connect("button_press_event", onclick)

    from collections import namedtuple
    Event = namedtuple("Event", ["xdata", "ydata"])
    ev = Event(100, 100)
    onclick(ev)


def plot_point_cloud(color_image, estimated_points3D_image, mask_image, camera_pos):
    if in_grading_mode(): # Skip when grading
        return
    
    # Pixels for which no match could be found are set to black (0, 0, 0)
    known_pixels_mask = (estimated_points3D_image != (0, 0, 0))[:,:,0]
    known_pixels_mask *= mask_image # Mask out pixels of the stairs in the background
    # Convert 3D position to depth/distance
    depth_image = np.linalg.norm(estimated_points3D_image - camera_pos, axis=2)
    #depth_image = depth_image * known_pixels_mask

    helpers.show_images({
        "Input Image": color_image,
        "Depth Image": depth_image / 10
    }, nrows=1, ncols=2)

    estimated_points3D = estimated_points3D_image[known_pixels_mask]
    colors3D = color_image[known_pixels_mask]

    trace = plotly_points(estimated_points3D, color=colors3D, size=4) # Only show one in every twenty points to improve performance
    layout = go.Layout(width=800, height=800, margin={"l": 0, "r": 0, "b": 0, "t": 0}, scene={"aspectmode": "data"})
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)


def plot_point_cloud_from_distance_map(color_image, depth_image, fovx = 65.2):
    if in_grading_mode(): # Skip when grading
        return
    
    height, width = depth_image.shape
    pos = glm.vec3(0)
    resolution = glm.vec2(width, height)

    fovx = np.radians(fovx)
    aspect_ratio = width / height

    half_image_plane = glm.vec2(0, np.sin(fovx * 0.5))
    half_image_plane.x = half_image_plane.y * aspect_ratio

    def screen_to_world(pixel, depth):
        uv = (pixel / resolution) * 2 - 1
        direction = glm.vec3(uv * half_image_plane, 1)
        return pos + direction * depth

    points3D_image = np.empty((height, width, 3))
    for y in range(height):
        for x in range(width):
            points3D_image[y, x] = screen_to_world(glm.vec2(x, y), depth_image[y, x])

    one_in_n = 17
    estimated_points3D = points3D_image.reshape(-1, 3)[::one_in_n]
    colors3D = color_image.reshape(-1, 3)[::one_in_n]

    tmp = estimated_points3D.copy()
    # Flip the y and z axis
    tmp[:,1], tmp[:,2] = tmp[:,2].copy(), tmp[:,1].copy()
    tmp *= -1 # Negate all axis

    trace = plotly_points(tmp, color=colors3D, size=4) # Only show one in every twenty points to improve performance
    layout = go.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0}, scene={"aspectmode": "data"})
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)