import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"

GRID_COLOR = "#E0E0E0"
UNMATCHED_TARGET_COLOR = "#B0B0B0"
UAV_COLORS = [
    "#4169E1",
    "#2E8B57",
    "#D2691E",
    "#C44E52",
    "#6A5ACD",
    "#008B8B",
]


def _wrap_ula_angle_to_principal_range(angle_deg):
    wrapped_angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
    if wrapped_angle_deg > 90.0:
        wrapped_angle_deg = 180.0 - wrapped_angle_deg
    elif wrapped_angle_deg < -90.0:
        wrapped_angle_deg = -180.0 - wrapped_angle_deg
    return wrapped_angle_deg


def _build_steering_vectors(angle_grid_deg, antenna_nums, frac_d_lambda):
    angle_grid_rad = np.deg2rad(angle_grid_deg)
    antenna_index = np.arange(antenna_nums)
    phase_matrix = 1j * 2.0 * np.pi * frac_d_lambda * np.sin(angle_grid_rad)[:, np.newaxis] * antenna_index[np.newaxis, :]
    return np.exp(phase_matrix)


def _compute_pattern_db(beam_matrix, steering_vectors):
    hermitian_beam_matrix = (np.asarray(beam_matrix) + np.asarray(beam_matrix).conj().T) / 2.0
    pattern_linear = np.real(
        np.einsum("an,nm,am->a", steering_vectors.conj(), hermitian_beam_matrix, steering_vectors)
    )
    pattern_linear = np.maximum(pattern_linear, 1e-12)
    pattern_peak = float(np.max(pattern_linear))
    if pattern_peak <= 0.0:
        pattern_peak = 1.0
    pattern_db = 10.0 * np.log10(pattern_linear / pattern_peak)
    return pattern_db, pattern_peak


def _extract_matched_target_indices(uavs_targets_matched_matrix):
    matched_target_indices = -np.ones(uavs_targets_matched_matrix.shape[0], dtype=int)
    row_indices, col_indices = np.where(np.asarray(uavs_targets_matched_matrix) > 0.5)
    for row_idx, col_idx in zip(row_indices, col_indices):
        matched_target_indices[row_idx] = col_idx
    return matched_target_indices


def _compute_target_angle_deg(uav_position, target_position):
    diff_xy = np.asarray(uav_position[:2], dtype=float) - np.asarray(target_position[:2], dtype=float)
    raw_angle_deg = float(np.rad2deg(np.arctan2(diff_xy[1], diff_xy[0])))
    return _wrap_ula_angle_to_principal_range(raw_angle_deg)


def _compute_target_gain_db(beam_matrix, antenna_nums, frac_d_lambda, target_angle_deg, pattern_peak):
    steering_vector = _build_steering_vectors(
        angle_grid_deg=np.asarray([target_angle_deg], dtype=float),
        antenna_nums=antenna_nums,
        frac_d_lambda=frac_d_lambda,
    )[0]
    hermitian_beam_matrix = (np.asarray(beam_matrix) + np.asarray(beam_matrix).conj().T) / 2.0
    target_gain_linear = float(np.real(steering_vector.conj().T @ hermitian_beam_matrix @ steering_vector))
    target_gain_linear = max(target_gain_linear, 1e-12)
    return 10.0 * np.log10(target_gain_linear / max(pattern_peak, 1e-12))


def plot_uav_sensing_beam_patterns(args,
                                   uavs_sen_beams,
                                   uavs_pos,
                                   targets_pos,
                                   uavs_targets_matched_matrix,
                                   save_path=None,
                                   show_plot=True):
    """
    在 penalty_based_cccp_algorithm.py 中调用，绘制 UAV 感知波束图。
    """
    if uavs_sen_beams is None or len(uavs_sen_beams) == 0:
        raise ValueError("uavs_sen_beams must contain at least one beam matrix.")

    angle_grid_deg = np.linspace(-90.0, 90.0, 2001)
    steering_vectors = _build_steering_vectors(
        angle_grid_deg=angle_grid_deg,
        antenna_nums=args.antenna_nums,
        frac_d_lambda=args.frac_d_lambda,
    )
    matched_target_indices = _extract_matched_target_indices(uavs_targets_matched_matrix)
    matched_target_index_set = set(matched_target_indices[matched_target_indices >= 0].tolist())

    fig, (ax_pos, ax) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.0, 1.0]})
    min_pattern_db = 0.0

    unmatched_target_mask = np.array(
        [target_idx not in matched_target_index_set for target_idx in range(targets_pos.shape[0])],
        dtype=bool,
    )
    if np.any(unmatched_target_mask):
        ax_pos.scatter(
            targets_pos[unmatched_target_mask, 0],
            targets_pos[unmatched_target_mask, 1],
            marker="*",
            s=90,
            color=UNMATCHED_TARGET_COLOR,
            edgecolors="none",
            label="Other TARGETs",
            zorder=2,
        )
    ax_pos.scatter(
        0.0,
        0.0,
        marker="o",
        s=70,
        color="red",
        zorder=3,
    )

    for uav_idx, uav_position in enumerate(uavs_pos):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        ax_pos.scatter(
            uav_position[0],
            uav_position[1],
            marker="^",
            s=120,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
        )

    for uav_idx, target_idx in enumerate(matched_target_indices):
        if target_idx < 0:
            continue
        ax_pos.plot(
            [uavs_pos[uav_idx, 0], targets_pos[target_idx, 0]],
            [uavs_pos[uav_idx, 1], targets_pos[target_idx, 1]],
            color="black",
            linewidth=1.2,
            zorder=3,
        )
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        ax_pos.scatter(
            targets_pos[target_idx, 0],
            targets_pos[target_idx, 1],
            marker="*",
            s=180,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=5,
        )

    for uav_idx, beam_matrix in enumerate(uavs_sen_beams):
        color = UAV_COLORS[uav_idx % len(UAV_COLORS)]
        pattern_db, pattern_peak = _compute_pattern_db(beam_matrix, steering_vectors)
        min_pattern_db = min(min_pattern_db, float(np.min(pattern_db)))

        ax.plot(
            angle_grid_deg,
            pattern_db,
            linestyle="-",
            linewidth=2.0,
            color=color,
            label=f"UAV {uav_idx + 1}",
            zorder=2,
        )

        target_idx = matched_target_indices[uav_idx]
        if target_idx >= 0:
            target_angle_deg = _compute_target_angle_deg(uavs_pos[uav_idx], targets_pos[target_idx])
            target_gain_db = _compute_target_gain_db(
                beam_matrix=beam_matrix,
                antenna_nums=args.antenna_nums,
                frac_d_lambda=args.frac_d_lambda,
                target_angle_deg=target_angle_deg,
                pattern_peak=pattern_peak,
            )
            ax.scatter(
                target_angle_deg,
                target_gain_db,
                marker="*",
                s=180,
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                zorder=4,
            )
    sensing_link_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.5,
        linestyle="-",
        label="Links represent UAV and its sensed TARGET",
    )
    uav_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="^",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="UAV",
    )
    matched_target_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="*",
        markersize=12,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Sensed TARGET",
    )
    other_target_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="*",
        markersize=10,
        markerfacecolor=UNMATCHED_TARGET_COLOR,
        markeredgecolor=UNMATCHED_TARGET_COLOR,
        label="Other targets",
    )
    bs_handle = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markersize=8,
        markerfacecolor="red",
        markeredgecolor="red",
        label="BS",
    )

    ax_pos.set_xlabel("x (m)", fontsize=16)
    ax_pos.set_ylabel("y (m)", fontsize=16)
    ax_pos.tick_params(axis="x", which="major", labelsize=16)
    ax_pos.tick_params(axis="y", which="major", labelsize=16)
    ax_pos.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax_pos.legend(
        handles=[sensing_link_handle, uav_handle, matched_target_handle, other_target_handle, bs_handle],
        fontsize=12,
        loc="upper left",
        frameon=True,
        ncol=2,
    )

    ax.set_xlabel("Angle (deg)", fontsize=16)
    ax.set_ylabel("Normalized sensing beampattern (dB)", fontsize=16)
    ax.tick_params(axis="x", which="major", labelsize=16)
    ax.tick_params(axis="y", which="major", labelsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlim(-90, 90)
    y_min = min(-5.0, 5.0 * np.floor(min_pattern_db / 5.0))
    ax.set_ylim(y_min, 60.0)
    ax.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax.legend(fontsize=16, loc="upper right", frameon=True, ncol=2)

    fig.subplots_adjust(hspace=0.32)
    plt.tight_layout()

    # if save_path is not None:
    #     save_dir = os.path.dirname(os.path.abspath(save_path))
    #     if save_dir:
    #         os.makedirs(save_dir, exist_ok=True)
    #     fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig
