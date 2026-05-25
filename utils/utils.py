import os
import sys
import time
import numpy as np

# import cvxpy as cp
# import shapely as sp
import matplotlib.pyplot as plt
import tkinter as tk
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from PIL import Image
from xml.dom import minidom
from functools import partial
from io import BytesIO
import math
from .transforms import t2p, rpy2r
import cv2
from PIL import ImageDraw, ImageFont


def trim_scale(x, th):
    """
    Trim scale
    """
    x = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x * th / x_abs_max
    return x


def compute_view_params(
    camera_pos,
    target_pos,
    up_vector=np.array([0, 0, 1]),
):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth)  # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation)  # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat


def get_idxs(list_query, list_domain):
    """
    Get corresponding indices of either two lists or ndarrays
    """
    if isinstance(list_query, list) and isinstance(list_domain, list):
        idxs = [list_query.index(item) for item in list_domain if item in list_query]
    else:
        print("[get_idxs] inputs should be 'List's.")
    return idxs


def get_idxs_contain(list_query, list_substring):
    """
    Get corresponding indices of either two lists
    """
    idxs = [
        i for i, s in enumerate(list_query) if any(sub in s for sub in list_substring)
    ]
    return idxs


def get_colors(n_color=10, cmap_name="gist_rainbow", alpha=1.0):
    """
    Get diverse colors
    """
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0, 1, n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors


def sample_xyzs(
    n_sample=1,
    x_range=[0, 1],
    y_range=[0, 1],
    z_range=[0, 1],
    min_dist=0.1,
    xy_margin=0.0,
):
    """
    Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample, 3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(
                low=x_range[0] + xy_margin, high=x_range[1] - xy_margin
            )
            y_rand = np.random.uniform(
                low=y_range[0] + xy_margin, high=y_range[1] - xy_margin
            )
            z_rand = np.random.uniform(low=z_range[0], high=z_range[1])
            xyz = np.array([x_rand, y_rand, z_rand])
            if p_idx == 0:
                break
            devc = cdist(
                xyz.reshape((-1, 3)), xyzs[:p_idx, :].reshape((-1, 3)), "euclidean"
            )
            if devc.min() > min_dist:
                break  # minimum distance between objects
        xyzs[p_idx, :] = xyz
    return xyzs


class ObjectSpawner:
    def __init__(self, env):
        """
        env: An environment instance that provides:
            - get_body_names(prefix)
            - set_p_base_body(body_name, p)
            - set_R_base_body(body_name, R)
        """
        self.env = env

    def spawn_objects(self):
        # --- Spawn the tray ---
        # Sample tray position using the provided sampling function.
        tray_xyz = sample_xyzs(
            n_sample=1,
            x_range=[0.3, 0.7],
            y_range=[-0.35, 0.35],
            z_range=[0.82, 0.82],
            min_dist=0.1,
            xy_margin=0.00,
        )[0]
        self.env.set_p_base_body(body_name="body_obj_tray_5", p=tray_xyz)

        # Randomly choose a tray orientation (and swap dimensions if rotated)
        if np.random.rand() > 0.5:
            # Rotate the tray by 90° about the z-axis
            self.env.set_R_base_body(
                body_name="body_obj_tray_5", R=rpy2r(np.deg2rad([0, 0, 90]))
            )

        # --- Get object names to spawn (exclude the tray) ---
        obj_names = self.env.get_body_names(prefix="body_obj_")
        if "body_obj_tray_5" in obj_names:
            obj_names.remove("body_obj_tray_5")

        # List to keep track of already placed objects to avoid collisions.
        placed_positions = []

        # Spawn each object with a non-colliding position and random rotation.
        for name in obj_names:
            # Set x-range based on a heuristic: objects with "can" use a restricted range.
            if "can" in name or "bottle" in name:
                x_range = [0.5, 0.7]
                z = 0.9
            else:
                x_range = [0.3, 0.6]
                z = 0.82
            y_range = [-0.35, 0.35]

            # Find a position that doesn't overlap with previously placed objects.
            pos = self._get_non_colliding_position(
                placed_positions=placed_positions,
                x_range=x_range,
                y_range=y_range,
                min_dist=0.1,
                tray_xyz=tray_xyz,
            )  # Optionally avoid the tray's area if needed.
            placed_positions.append(pos)
            # Set the object's position (using the same z as the tray for simplicity).
            self.env.set_p_base_body(body_name=name, p=[pos[0], pos[1], z])

            # Optionally, assign a random rotation.
            angle = np.random.uniform(0, 360)
            self.env.set_R_base_body(body_name=name, R=rpy2r(np.deg2rad([0, 0, angle])))

    def _get_non_colliding_position(
        self, placed_positions, x_range, y_range, min_dist, tray_xyz
    ):
        """Attempts to sample a position that does not collide with already placed objects (or the tray).
        Raises a ValueError if no valid position is found after a fixed number of attempts.
        """
        max_attempts = 100
        tray_margin = (
            0.3  # Define a margin to avoid overlap with the tray center if needed.
        )
        for attempt in range(max_attempts):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            candidate = np.array([x, y])

            collision = False
            # Check distance from already placed objects.
            for pos in placed_positions:
                if np.linalg.norm(candidate - np.array(pos)) < min_dist:
                    collision = True
                    break
            # Optional: check if candidate is too close to the tray's center.
            if np.linalg.norm(candidate - np.array(tray_xyz[:2])) < tray_margin:
                collision = True
            if not collision:
                return candidate
        raise ValueError(
            "Could not find a non-colliding position after {} attempts".format(
                max_attempts
            )
        )


def sample_xyzs(
    n_sample=1,
    x_range=[0, 1],
    y_range=[0, 1],
    z_range=[0, 1],
    min_dist=0.1,
    xy_margin=0.0,
    rng=None,
):
    """
    Sample a point in three dimensional space with the minimum distance between points

    rng: optional ``numpy.random.Generator``; if None, uses the legacy global ``numpy.random``.
    """
    rnd = np.random if rng is None else rng
    xyzs = np.zeros((n_sample, 3))
    for p_idx in range(n_sample):
        while True:
            x_rand = rnd.uniform(
                low=x_range[0] + xy_margin, high=x_range[1] - xy_margin
            )
            y_rand = rnd.uniform(
                low=y_range[0] + xy_margin, high=y_range[1] - xy_margin
            )
            z_rand = rnd.uniform(low=z_range[0], high=z_range[1])
            xyz = np.array([x_rand, y_rand, z_rand])
            if p_idx == 0:
                break
            devc = cdist(
                xyz.reshape((-1, 3)), xyzs[:p_idx, :].reshape((-1, 3)), "euclidean"
            )
            if devc.min() > min_dist:
                break  # minimum distance between objects
        xyzs[p_idx, :] = xyz
    return xyzs


def save_png(img, png_path, verbose=False):
    """
    Save image
    """
    directory = os.path.dirname(png_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print("[%s] generated." % (directory))
    # Save to png
    plt.imsave(png_path, img)
    if verbose:
        print("[%s] saved." % (png_path))


def finite_difference_matrix(n, dt, order):
    """
    n: number of points
    dt: time interval
    order: (1=velocity, 2=acceleration, 3=jerk)
    """
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c

    # (optional) Handling boundary conditions with backward differences
    if order == 1:  # velocity
        mat[-1, -2:] = np.array([-1, 1])  # backward difference
    elif order == 2:  # acceleration
        mat[-1, -3:] = np.array([1, -2, 1])  # backward difference
        mat[-2, -3:] = np.array([1, -2, 1])  # backward difference
    elif order == 3:  # jerk
        mat[-1, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-2, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-3, -4:] = np.array([-1, 3, -3, 1])  # backward difference

    # Return
    return mat / (dt**order)


def get_A_vel_acc_jerk(n=100, dt=1e-2):
    """
    Get matrices to compute velocities, accelerations, and jerks
    """
    A_vel = finite_difference_matrix(n, dt, order=1)
    A_acc = finite_difference_matrix(n, dt, order=2)
    A_jerk = finite_difference_matrix(n, dt, order=3)
    return A_vel, A_acc, A_jerk


def get_idxs_closest_ndarray(ndarray_query, ndarray_domain):
    return [np.argmin(np.abs(ndarray_query - x)) for x in ndarray_domain]


def get_interp_const_vel_traj_nd(
    anchors,  # [L x D]
    vel=1.0,
    HZ=100,
    ord=np.inf,
):
    """
    Get linearly interpolated constant velocity trajectory
    Output is (times_interp,anchors_interp,times_anchor,idxs_anchor)
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev, p_curr = anchors[tick - 1, :], anchors[tick, :]
            dists[tick] = np.linalg.norm(p_prev - p_curr, ord=ord)
    times_anchor = np.cumsum(dists / vel)  # [L]
    L_interp = int(times_anchor[-1] * HZ)
    times_interp = np.linspace(0, times_anchor[-1], L_interp)  # [L_interp]
    anchors_interp = np.zeros((L_interp, D))  # [L_interp x D]
    for d_idx in range(D):  # for each dim
        anchors_interp[:, d_idx] = np.interp(
            times_interp, times_anchor, anchors[:, d_idx]
        )
    idxs_anchor = get_idxs_closest_ndarray(times_interp, times_anchor)
    return times_interp, anchors_interp, times_anchor, idxs_anchor


def check_vel_acc_jerk_nd(
    times,  # [L]
    traj,  # [L x D]
    verbose=True,
    factor=1.0,
):
    """
    Check velocity, acceleration, jerk of n-dimensional trajectory
    """
    L, D = traj.shape[0], traj.shape[1]
    A_vel, A_acc, A_jerk = get_A_vel_acc_jerk(n=len(times), dt=times[1] - times[0])
    vel_inits, vel_finals, max_vels, max_accs, max_jerks = [], [], [], [], []
    for d_idx in range(D):
        traj_d = traj[:, d_idx]
        vel = A_vel @ traj_d
        acc = A_acc @ traj_d
        jerk = A_jerk @ traj_d
        vel_inits.append(vel[0])
        vel_finals.append(vel[-1])
        max_vels.append(np.abs(vel).max())
        max_accs.append(np.abs(acc).max())
        max_jerks.append(np.abs(jerk).max())

    # Print
    if verbose:
        print(
            "Checking velocity, acceleration, and jerk of a L:[%d]xD:[%d] trajectory (factor:[%.2f])."
            % (L, D, factor)
        )
        for d_idx in range(D):
            print(
                " dim:[%d/%d]: v_init:[%.2e] v_final:[%.2e] v_max:[%.2f] a_max:[%.2f] j_max:[%.2f]"
                % (
                    d_idx,
                    D,
                    factor * vel_inits[d_idx],
                    factor * vel_finals[d_idx],
                    factor * max_vels[d_idx],
                    factor * max_accs[d_idx],
                    factor * max_jerks[d_idx],
                )
            )

    # Return
    return vel_inits, vel_finals, max_vels, max_accs, max_jerks


def np_uv(vec):
    """
    Get unit vector
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0, 0, 1])
    else:
        return x / len


def uv_T_joi(T_joi, joi_fr, joi_to):
    """
    Get unit vector between to JOI poses
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))


def len_T_joi(T_joi, joi_fr, joi_to):
    """
    Get length between two JOI poses
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))


def get_consecutive_subarrays(array, min_element=1):
    """
    Get consecutive sub arrays from an array
    """
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array, split_points)
    return [subarray for subarray in subarrays if len(subarray) >= min_element]


def load_image(image_path):
    """
    Load image to ndarray (unit8)
    """
    return np.array(Image.open(image_path))


def imshows(img_list, title_list, figsize=(8, 2), fontsize=8):
    """
    Plot multiple images in a row
    """
    n_img = len(img_list)
    plt.figure(figsize=(8, 2))
    for img_idx in range(n_img):
        img = img_list[img_idx]
        title = title_list[img_idx]
        plt.subplot(1, n_img, img_idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title, fontsize=fontsize)
    plt.show()


def depth_to_gray_img(depth, max_val=10.0):
    """
    1-channel float-type depth image to 3-channel unit8-type gray image
    """
    depth_clip = np.clip(depth, a_min=0.0, a_max=max_val)  # float-type
    img = np.tile(
        255 * depth_clip[:, :, np.newaxis] / depth_clip.max(), (1, 1, 3)
    ).astype(
        np.uint8
    )  # unit8-type
    return img


def get_monitor_size():
    """
    Return primary monitor (width, height).

    pyautogui pulls mouseinfo which requires DISPLAY at import time; skip it on
    Linux headless runs. Values only affect window/fullscreen layout; override
    with MYI10_MONITOR_WIDTH / MYI10_MONITOR_HEIGHT if needed.
    """
    w = int(os.environ.get("MYI10_MONITOR_WIDTH", "1920"))
    h = int(os.environ.get("MYI10_MONITOR_HEIGHT", "1080"))
    if (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    ):
        return w, h
    try:
        import pyautogui
    except ImportError:
        return w, h

    return pyautogui.size()


def get_xml_string_from_path(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)

    # Get the root element of the XML
    root = tree.getroot()

    # Convert the ElementTree object to a string
    xml_string = ET.tostring(root, encoding="unicode", method="xml")

    return xml_string


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")

    # 불필요한 공백 제거 (빈 줄)
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)


class TicTocClass(object):
    """
    Tic toc
    tictoc = TicTocClass()
    tictoc.tic()
    ~~
    tictoc.toc()
    """

    def __init__(self, name="tictoc", print_every=1):
        """
        Initialize
        """
        self.name = name
        self.time_start = time.time()
        self.time_end = time.time()
        self.print_every = print_every
        self.time_elapsed = 0.0
        self.cnt = 0

    def tic(self):
        """
        Tic
        """
        self.time_start = time.time()

    def toc(self, str=None, cnt=None, print_every=None, verbose=False):
        """
        Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if print_every is not None:
            self.print_every = print_every
        if verbose:
            if self.time_elapsed < 1.0:
                time_show = self.time_elapsed * 1000.0
                time_unit = "ms"
            elif self.time_elapsed < 60.0:
                time_show = self.time_elapsed
                time_unit = "s"
            else:
                time_show = self.time_elapsed / 60.0
                time_unit = "min"
            if cnt is not None:
                self.cnt = cnt
            if (self.cnt % self.print_every) == 0:
                if str is None:
                    print(
                        "%s Elapsed time:[%.2f]%s" % (self.name, time_show, time_unit)
                    )
                else:
                    print("%s Elapsed time:[%.2f]%s" % (str, time_show, time_unit))
        self.cnt = self.cnt + 1
        # Return
        return self.time_elapsed


def sleep(sec):
    """
    Time sleep
    """
    time.sleep(sec)


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.asarray(data)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.asarray(point[:3], dtype=np.float32)
        M[:3, 3] = point - np.dot(R, point)
    return M


def add_title_to_img(
    img,
    text="Title",
    margin_top=30,
    color=(0, 0, 0),
    font_size=20,
    resize=True,
    shape=(300, 300),
):
    """
    Add title to image
    """
    # Resize
    img_copied = img.copy()
    if resize:
        img_copied = cv2.resize(img_copied, shape, interpolation=cv2.INTER_NEAREST)
    # Convert to PIL image
    pil_img = Image.fromarray(img_copied)  #
    width, height = pil_img.size
    new_height = margin_top + height
    # Create new image with top margin
    new_img = Image.new("RGB", (width, new_height), color=(255, 255, 255))
    # Paste the original image
    new_img.paste(pil_img, (0, margin_top))
    # Draw text
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default(size=font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    # Center text
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (margin_top - text_height) // 2
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    img_with_title = np.array(new_img)
    # Return
    return img_with_title


# Inverse kinematics helper
def init_ik_info():
    """
    Initialize IK information
    Usage:
    ik_info = init_ik_info()
    ...
    add_ik_info(ik_info,body_name='BODY_NAME',p_trgt=P_TRGT,R_trgt=R_TRGT)
    ...
    for ik_tick in range(max_ik_tick):
        dq,ik_err_stack = get_dq_from_ik_info(
            env = env,
            ik_info = ik_info,
            stepsize = 1,
            eps = 1e-2,
            th = np.radians(10.0),
            joint_idxs_jac = joint_idxs_jac,
        )
        qpos = env.get_qpos()
        mujoco.mj_integratePos(env.model,qpos,dq,1)
        env.forward(q=qpos)
        if np.linalg.norm(ik_err_stack) < 0.05: break
    """
    ik_info = {
        "body_names": [],
        "geom_names": [],
        "p_trgts": [],
        "R_trgts": [],
        "n_trgt": 0,
    }
    return ik_info


def add_ik_info(
    ik_info,
    body_name=None,
    geom_name=None,
    p_trgt=None,
    R_trgt=None,
):
    """
    Add IK information
    """
    ik_info["body_names"].append(body_name)
    ik_info["geom_names"].append(geom_name)
    ik_info["p_trgts"].append(p_trgt)
    ik_info["R_trgts"].append(R_trgt)
    ik_info["n_trgt"] = ik_info["n_trgt"] + 1


def get_dq_from_ik_info(
    env,
    ik_info,
    stepsize=1,
    eps=1e-2,
    th=np.radians(1.0),
    joint_idxs_jac=None,
):
    """
    Get delta q from augmented Jacobian method
    """
    J_list, ik_err_list = [], []
    for ik_idx, (ik_body_name, ik_geom_name) in enumerate(
        zip(ik_info["body_names"], ik_info["geom_names"])
    ):
        ik_p_trgt = ik_info["p_trgts"][ik_idx]
        ik_R_trgt = ik_info["R_trgts"][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None
        J, ik_err = env.get_ik_ingredients(
            body_name=ik_body_name,
            geom_name=ik_geom_name,
            p_trgt=ik_p_trgt,
            R_trgt=ik_R_trgt,
            IK_P=IK_P,
            IK_R=IK_R,
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    J_stack = np.vstack(J_list)
    ik_err_stack = np.hstack(ik_err_list)

    # Select Jacobian columns that are within the joints to use
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:, joint_idxs_jac] = J_stack_backup[:, joint_idxs_jac]

    # Compute dq from damped least square
    dq = env.damped_ls(J_stack, ik_err_stack, stepsize=stepsize, eps=eps, th=th)
    return dq, ik_err_stack


def plot_ik_info(
    env,
    ik_info,
    axis_len=0.05,
    axis_width=0.005,
    sphere_r=0.01,
):
    """
    Plot IK information
    """
    colors = get_colors(cmap_name="gist_rainbow", n_color=ik_info["n_trgt"])
    for ik_idx, (ik_body_name, ik_geom_name) in enumerate(
        zip(ik_info["body_names"], ik_info["geom_names"])
    ):
        color = colors[ik_idx]
        ik_p_trgt = ik_info["p_trgts"][ik_idx]
        ik_R_trgt = ik_info["R_trgts"][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None

        if ik_body_name is not None:
            # Plot current
            env.plot_body_T(
                body_name=ik_body_name,
                plot_axis=IK_R,
                axis_len=axis_len,
                axis_width=axis_width,
                plot_sphere=IK_P,
                sphere_r=sphere_r,
                sphere_rgba=color,
                label="",
            )  # ''/ik_body_name
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt, r=sphere_r, rgba=color, label="")
                env.plot_line_fr2to(
                    p_fr=env.get_p_body(body_name=ik_body_name),
                    p_to=ik_p_trgt,
                    rgba=color,
                )
            if IK_P and IK_R:
                env.plot_T(
                    p=ik_p_trgt,
                    R=ik_R_trgt,
                    plot_axis=True,
                    axis_len=axis_len,
                    axis_width=axis_width,
                )
            if not IK_P and IK_R:  # rotation only
                p_curr = env.get_p_body(body_name=ik_body_name)
                env.plot_T(
                    p=p_curr,
                    R=ik_R_trgt,
                    plot_axis=True,
                    axis_len=axis_len,
                    axis_width=axis_width,
                )

        if ik_geom_name is not None:
            # Plot current
            env.plot_geom_T(
                geom_name=ik_geom_name,
                plot_axis=IK_R,
                axis_len=axis_len,
                axis_width=axis_width,
                plot_sphere=IK_P,
                sphere_r=sphere_r,
                sphere_rgba=color,
                label="",
            )  # ''/ik_geom_name
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt, r=sphere_r, rgba=color, label="")
                env.plot_line_fr2to(
                    p_fr=env.get_p_geom(geom_name=ik_geom_name),
                    p_to=ik_p_trgt,
                    rgba=color,
                )
            if IK_P and IK_R:
                env.plot_T(
                    p=ik_p_trgt,
                    R=ik_R_trgt,
                    plot_axis=True,
                    axis_len=axis_len,
                    axis_width=axis_width,
                )
            if not IK_P and IK_R:  # rotation only
                p_curr = env.get_p_geom(geom_name=ik_geom_name)
                env.plot_T(
                    p=p_curr,
                    R=ik_R_trgt,
                    plot_axis=True,
                    axis_len=axis_len,
                    axis_width=axis_width,
                )


def solve_ik(
    env,
    joint_names_for_ik,
    body_name_trgt,
    q_init=None,  # IK start from the initial pose
    p_trgt=None,
    R_trgt=None,
    max_ik_tick=1000,
    ik_err_th=1e-2,
    restore_state=True,
    ik_stepsize=1.0,
    ik_eps=1e-2,
    ik_th=np.radians(1.0),
    verbose=False,
    verbose_warning=True,
    reset_env=False,
    render=False,
    render_every=1,
):
    """
    Solve Inverse Kinematics
    """
    # Reset
    if reset_env:
        env.reset()
    if render:
        env.init_viewer()
    # Joint indices
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_idxs_fwd = env.get_idxs_fwd(joint_names=joint_names_for_ik)
    # Joint range
    q_mins = env.joint_ranges[get_idxs(env.joint_names, joint_names_for_ik), 0]
    q_maxs = env.joint_ranges[get_idxs(env.joint_names, joint_names_for_ik), 1]
    # Store MuJoCo state
    if restore_state:
        env.store_state()
    # Initial IK pose
    if q_init is not None:
        env.forward(q=q_init, joint_idxs=joint_idxs_fwd, increase_tick=False)
    # Initialize IK information
    ik_info = init_ik_info()
    add_ik_info(
        ik_info=ik_info,
        body_name=body_name_trgt,
        p_trgt=p_trgt,
        R_trgt=R_trgt,
    )
    # Loop
    q_curr = env.get_qpos_joints(joint_names=joint_names_for_ik)
    for ik_tick in range(max_ik_tick):
        dq, ik_err_stack = get_dq_from_ik_info(
            env=env,
            ik_info=ik_info,
            stepsize=ik_stepsize,
            eps=ik_eps,
            th=ik_th,
            joint_idxs_jac=joint_idxs_jac,
        )
        q_curr = q_curr + dq[joint_idxs_jac]  # update
        q_curr = np.clip(q_curr, q_mins, q_maxs)  # clip
        env.forward(q=q_curr, joint_idxs=joint_idxs_fwd, increase_tick=False)  # fk
        ik_err = np.linalg.norm(ik_err_stack)  # IK error
        if ik_err < ik_err_th:
            break  # terminate condition
        if verbose:
            print("[%d/%d] ik_err:[%.3f]" % (ik_tick, max_ik_tick, ik_err))
        if render:
            if ik_tick % render_every == 0:
                plot_ik_info(env, ik_info)
                env.render()
    # Print if IK error is too high
    if verbose_warning and ik_err > ik_err_th:
        print("ik_err:[%.4f] is higher than ik_err_th:[%.4f]." % (ik_err, ik_err_th))
        print("You may want to increase max_ik_tick:[%d]" % (max_ik_tick))
    # Restore backuped state
    if restore_state:
        env.restore_state()
    # Close viewer
    if render:
        env.close_viewer()
    # Return
    return q_curr, ik_err_stack, ik_info
