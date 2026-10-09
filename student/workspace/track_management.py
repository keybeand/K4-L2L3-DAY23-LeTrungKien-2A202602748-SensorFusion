"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()

    z_arr = np.asarray(meas.z, dtype=float).flatten()
    px, py, pz = z_arr[0], z_arr[1], z_arr[2]

    x = np.matrix([[px], [py], [pz], [0.0], [0.0], [0.0]], dtype=float)

    P = np.matrix(np.zeros((6, 6), dtype=float))
    R_mat = np.asarray(meas.R, dtype=float)
    P[:3, :3] = R_mat

    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    score = 1.0 / params.window
    state = "initialized"

    return {"x": x, "P": P, "state": state, "score": score}


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    step = 1.0 / params.window

    if associated:
        track["score"] = min(1.0, track["score"] + step)
    else:
        track["score"] = track["score"] - step

    if track["score"] > params.confirmed_threshold:
        track["state"] = "confirmed"

    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P_mat = np.asarray(track["P"], dtype=float)

    pxx_var = P_mat[0, 0]
    pyy_var = P_mat[1, 1]

    if pxx_var > params.max_P or pyy_var > params.max_P:
        return True

    is_confirmed = track.get("state") == "confirmed"

    if is_confirmed:
        if track["score"] < params.delete_threshold:
            return True
    else:
        if track["score"] <= 0.0:
            return True

    return False

