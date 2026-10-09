"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    S_inv = np.linalg.inv(S)
    mhd_mat = gamma.T @ S_inv @ gamma
    mhd_sq = float(np.asarray(mhd_mat).item())
    return mhd_sq



def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    num_tracks = len(track_list)
    num_meas = len(meas_list)
    cost_matrix = np.full((num_tracks, num_meas), np.inf, dtype=float)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if not meas.sensor.in_fov(track.x):
                continue
            mhd_sq = mahalanobis_distance(track, meas)
            if chi2_gate(mhd_sq, meas.sensor):
                cost_matrix[i, j] = mhd_sq

    return np.matrix(cost_matrix)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    cost_arr = np.asarray(association_matrix, dtype=float)
    if cost_arr.size == 0 or not np.any(np.isfinite(cost_arr)):
        return (
            np.nan,
            np.nan,
            association_matrix,
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    min_idx = np.unravel_index(np.argmin(cost_arr), cost_arr.shape)
    min_i, min_j = min_idx[0], min_idx[1]

    if not np.isfinite(cost_arr[min_i, min_j]):
        return (
            np.nan,
            np.nan,
            association_matrix,
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    matched_track = unassigned_tracks[min_i]
    matched_meas = unassigned_meas[min_j]

    new_cost_arr = np.delete(cost_arr, min_i, axis=0)
    new_cost_arr = np.delete(new_cost_arr, min_j, axis=1)

    rem_tracks = [tr for k, tr in enumerate(unassigned_tracks) if k != min_i]
    rem_meas = [m for k, m in enumerate(unassigned_meas) if k != min_j]

    return (
        matched_track,
        matched_meas,
        np.matrix(new_cost_arr),
        rem_tracks,
        rem_meas,
    )


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    assoc_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

    while True:
        track, meas, assoc_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
            assoc_matrix, unassigned_tracks, unassigned_meas
        )
        if track is np.nan or (isinstance(track, float) and np.isnan(track)):
            break

        filter_obj.update(track, meas)
        manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)

