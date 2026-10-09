"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

Matrix = np.matrix | np.ndarray

from fusion_lab.workspace_support import get_tracking_params


def _transform_veh_to_sens(x: Matrix, sensor: Any) -> np.ndarray:
    """Helper to transform 3D position x in vehicle frame to sensor frame."""
    pos_veh = np.asarray(x, dtype=float).reshape(-1)[:3]
    p_homo = np.matrix([[pos_veh[0]], [pos_veh[1]], [pos_veh[2]], [1.0]])
    p_sens_homo = sensor.veh_to_sens @ p_homo
    return np.asarray(p_sens_homo[:3]).flatten()


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    pos_sens_arr = _transform_veh_to_sens(x, sensor)
    if not np.all(np.isfinite(pos_sens_arr)):
        return False

    x_s, y_s, z_s = pos_sens_arr[0], pos_sens_arr[1], pos_sens_arr[2]

    # Kiểm tra depth với camera
    if hasattr(sensor, "is_camera") and sensor.is_camera:
        if x_s <= 1e-6:
            return False
    elif hasattr(sensor, "c_i"):  # fallback camera check
        if x_s <= 1e-6:
            return False

    angle = np.arctan2(y_s, x_s)
    fov = sensor.fov
    if isinstance(fov, (tuple, list, np.ndarray)) and len(fov) == 2:
        min_fov, max_fov = min(fov), max(fov)
        return bool(min_fov <= angle <= max_fov)
    else:
        half_fov = float(fov) / 2.0
        return bool(-half_fov <= angle <= half_fov)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    pos_sens_arr = _transform_veh_to_sens(x, sensor)

    if not np.all(np.isfinite(pos_sens_arr)):
        raise ValueError(f"Non-finite sensor coordinates: {pos_sens_arr}")

    x_s, y_s, z_s = pos_sens_arr[0], pos_sens_arr[1], pos_sens_arr[2]
    if x_s <= 1e-6:
        raise ValueError(f"Invalid depth x_s = {x_s} <= 1e-6")

    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)

    return np.matrix([[u], [v]], dtype=float)



def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_mat = np.matrix(np.array(z, dtype=float).reshape(2, 1))

    R = np.matrix(
        [
            [params.sigma_cam_i**2, 0.0],
            [0.0, params.sigma_cam_j**2],
        ],
        dtype=float,
    )

    return {"z": z_mat, "R": R, "sensor": sensor}

