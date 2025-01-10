import itertools
import functools
import glm
from collections import defaultdict, namedtuple
import helpers
import os
import numpy as np
import glm
import mvs
from numba import jit, njit

# https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
def group_by(it, group_size):
    return zip(*(iter(it),) * group_size)

def _world_to_screen(rot, trans, camera_to_screen, p):
    return camera_to_screen(rot * (p + trans))
def _screen_to_world(rot, trans, screen_to_camera, p, z):
    return (rot * screen_to_camera(p, z)) + trans

def _camera_to_screen(swap_uv, f, c, p):
    uv_xy = glm.vec2(p) / p.z
    out = uv_xy * f + c
    if swap_uv:
        out = glm.vec2(out.y, out.x)
    return out
def _screen_to_camera(swap_uv, f, c, uv, z):
    if swap_uv:
        uv = glm.vec2(uv.y, uv.x)
    return glm.vec3(z * (uv - c) / f, z)

def load_eth3d_dataset(base_folder, image_id_filter, scale = 1.0, swap_uv = False):
    calibration_folder = os.path.join(base_folder, "dslr_calibration_undistorted")
    images_folder = os.path.join(base_folder, "images")

    internal_cameras = {}
    with open(os.path.join(calibration_folder, "cameras.txt")) as file:
        for line in file:
            if line.startswith("#"):
                continue
            
            camera_id, model, width, height, *params = line.split(" ")
            assert(model == "PINHOLE")
            fx, fy, cx, cy = [float(p) * scale for p in params]
            f = glm.vec2(fx, fy)
            c = glm.vec2(cx, cy)

            # https://www.eth3d.net/documentation#camera-models-pinhole-camera-model
            camera_to_screen = functools.partial(_camera_to_screen, swap_uv, f, c)
            screen_to_camera = functools.partial(_screen_to_camera, swap_uv, f, c)

            internal_cameras[camera_id] = {
                "model": model,
                "camera_to_screen": camera_to_screen,
                "screen_to_camera": screen_to_camera,
            }
            
    cameras = {}
    images = {}
    internal_points2d_per_image = {}
    with open(os.path.join(calibration_folder, "images.txt")) as file:
        lines = [line for line in file if not line.startswith("#")]
        for line1, line2 in group_by(lines, 2):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line1[:-1].split(" ")
            if image_id not in image_id_filter:
                continue
            
            world_to_camera_rotation = glm.quat()
            world_to_camera_rotation.x = float(qx)
            world_to_camera_rotation.y = float(qy)
            world_to_camera_rotation.z = float(qz)
            world_to_camera_rotation.w = float(qw)
            camera_to_world_rotation = glm.conjugate(world_to_camera_rotation)

            # For some inexplicable reason the ETH3D dataset uses the obscure convention of [rotate] THEN [translate] when going from world to screen coordinates.
            # Undo the mess to save our collective sanity.
            stupid_world_to_camera = glm.vec3(float(tx), float(ty), float(tz))
            world_to_camera_translation = camera_to_world_rotation * stupid_world_to_camera
            camera_pos = camera_to_world_translation = -world_to_camera_translation

            # Python captures "by reference" so it will always refer to captured variables by their latest value / loop iteration.
            # Using partial bindings is a work around for a feature that C++ does much better.
            internal_camera = internal_cameras[camera_id]
            world_to_screen = functools.partial(_world_to_screen, world_to_camera_rotation, world_to_camera_translation, internal_camera["camera_to_screen"])
            screen_to_world = functools.partial(_screen_to_world, camera_to_world_rotation, camera_to_world_translation, internal_camera["screen_to_camera"])

            cameras[image_id] = mvs.Camera(
                camera_pos, world_to_screen, screen_to_world)
            if swap_uv:
                internal_points2d_per_image[image_id] = [scale * glm.vec2(float(y), float(x)) for x, y, _ in group_by(line2.split(" "), 3)]
            else:
                internal_points2d_per_image[image_id] = [scale * glm.vec2(float(x), float(y)) for x, y, _ in group_by(line2.split(" "), 3)]

            image = helpers.imread_normalized_float(os.path.join(images_folder, name), scale)
            if swap_uv:
                image = np.swapaxes(image, 0, 1)
            images[image_id] = image
            
    points3d = []
    points2d = defaultdict(lambda: [])
    with open(os.path.join(calibration_folder, "points3D.txt")) as file:
        for line in list(file):
            if line.startswith("#"):
                continue
            point_id, x, y, z, r, g, b, error, *track = line[:-1].split(" ")
            track = dict(group_by(track, 2))

            visible_in_all_filter_images = functools.reduce(lambda a, b: a and b, [fi in track for fi in image_id_filter])
            if visible_in_all_filter_images:
                points3d.append(glm.vec3(float(x), float(y), float(z)))
                for image_id in image_id_filter:
                    points2d[image_id].append(internal_points2d_per_image[image_id][int(track[image_id])])
            
    cameras = [cameras[image_id] for image_id in image_id_filter]
    images = [images[image_id] for image_id in image_id_filter]
    points2d = [points2d[image_id] for image_id in image_id_filter]
    return (points3d, cameras, images, points2d)