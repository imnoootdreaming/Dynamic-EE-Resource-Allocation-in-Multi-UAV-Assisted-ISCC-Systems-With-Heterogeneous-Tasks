"""Shared deterministic ring target geometry for inner and outer environments.

global_ring uses a fixed fraction of the deployment radius. The legacy
sector_ring combines sector containment with a reference movement budget.
These are scenario-design rules, not solver feasibility bounds.
Only NumPy is required; target generation does not consume any random state.
"""

import numpy as np


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if not np.isfinite(number) or number <= 0 or number != int(number):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(number)


def _finite_number(name, value, *, positive=False, nonnegative=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite, got {value!r}") from None
    if not np.isfinite(number) or (positive and number <= 0) or (nonnegative and number < 0):
        requirement = "positive and finite" if positive else "nonnegative and finite" if nonnegative else "finite"
        raise ValueError(f"{name} must be {requirement}, got {value!r}")
    return number


def resolve_equal_target_hold_slots(uavs_num, targets_num, total_time_slots, *, layout="global_ring"):
    """Equal responsibility and complete windows: h = T / (K / I)."""
    uavs_num = _positive_integer("uavs_num", uavs_num)
    targets_num = _positive_integer("targets_num", targets_num)
    total_time_slots = _positive_integer("total_time_slots", total_time_slots)
    if targets_num < uavs_num or targets_num % uavs_num:
        raise ValueError(f"{layout} requires targets_num >= uavs_num and targets_num % uavs_num == 0")
    per_uav = targets_num // uavs_num
    if total_time_slots % per_uav:
        raise ValueError(f"{layout} requires total_time_slots divisible by targets_num / uavs_num")
    return total_time_slots // per_uav


def resolve_global_ring_hold_slots(uavs_num, targets_num, total_time_slots):
    return resolve_equal_target_hold_slots(uavs_num, targets_num, total_time_slots)


def generate_global_ring_targets(uavs_num, targets_num, center, deploy_radius, *,
                                 radius_ratio=0.5, direction="clockwise", target_height=0.0):
    """One global circle; target coordinates and IDs are independent of UAV count.

    Half a target-spacing offset avoids sector boundaries for every divisor of K.
    IDs follow the requested direction globally; each sector's ascending IDs are
    therefore also its visit order. No speed/window bound clips the chosen radius.
    """
    uavs_num = _positive_integer("uavs_num", uavs_num)
    targets_num = _positive_integer("targets_num", targets_num)
    if targets_num < uavs_num or targets_num % uavs_num:
        raise ValueError("global_ring requires targets_num >= uavs_num and targets_num % uavs_num == 0")
    deploy_radius = _finite_number("deploy_radius", deploy_radius, positive=True)
    radius_ratio = _finite_number("radius_ratio", radius_ratio, positive=True)
    if radius_ratio >= 1.0:
        raise ValueError("radius_ratio must lie strictly between 0 and 1")
    if direction not in ("clockwise", "counterclockwise"):
        raise ValueError("direction must be 'clockwise' or 'counterclockwise'")
    target_height = _finite_number("target_height", target_height)
    try:
        center = np.asarray(center, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("center must contain two or three finite coordinates") from None
    if center.shape not in ((2,), (3,)) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two or three finite coordinates")
    base_angles = (np.arange(targets_num) + 0.5) * (2.0 * np.pi / targets_num)
    angles = 2.0 * np.pi - base_angles if direction == "clockwise" else base_angles
    ring_radius = radius_ratio * deploy_radius
    targets = np.empty((targets_num, 3), dtype=float)
    targets[:, :2] = center[:2] + ring_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    targets[:, 2] = target_height
    sectors = np.minimum((angles / (2.0 * np.pi / uavs_num)).astype(np.int64), uavs_num - 1)
    visit_order = [np.flatnonzero(sectors == i).tolist() for i in range(uavs_num)]
    info = {
        "layout": "global_ring", "geometry_contract": "fixed_global_ring_v1",
        "center": [float(center[0]), float(center[1]), target_height],
        "ring_radius": float(ring_radius), "radius_ratio": radius_ratio,
        "deploy_radius": deploy_radius, "targets_per_sector": targets_num // uavs_num,
        "direction": direction, "visit_order": visit_order,
        "first_target_angle": float(angles[0]),
        "adjacent_distance": 0.0 if targets_num == 1 else float(2.0 * ring_radius * np.sin(np.pi / targets_num)),
        "radius_rule": "radius_ratio * deploy_radius; independent of UAV count and window length",
        "target_rng_semantics": "deterministic_ring_preserves_legacy_draws",
    }
    return targets, sectors, info


def generate_sector_ring_targets(uavs_num, targets_num, center, deploy_radius, *,
                                 hold_slots=4, time_slot_duration=0.6,
                                 min_speed=5.0, max_speed=40.0,
                                 space_ratio=1.0 / 3.0, move_ratio=0.85,
                                 direction="clockwise", target_height=0.0):
    """Return (target_positions, target_sector, layout_info).

    Each sector has one complete circle centered on its area centroid. Targets
    are stored as consecutive sector blocks and in the intended visit order.
    The first target points toward the deployment center; for one sector the
    circle is centered there and starts along the positive x axis.

    With n targets per sector, R = min(space_ratio * containment_radius,
    move_ratio * hold_slots * time_slot_duration * reference_speed /
    (2 sin(pi/n))), where reference_speed = (min_speed + max_speed) / 2.
    For n == 1 there is no target transition and no movement radius limit.
    layout_info contains JSON-serializable geometry and budget metadata.
    """
    uavs_num = _positive_integer("uavs_num", uavs_num)
    targets_num = _positive_integer("targets_num", targets_num)
    hold_slots = _positive_integer("hold_slots", hold_slots)
    if targets_num < uavs_num or targets_num % uavs_num != 0:
        raise ValueError("sector_ring requires targets_num >= uavs_num and targets_num % uavs_num == 0")
    n = targets_num // uavs_num
    deploy_radius = _finite_number("deploy_radius", deploy_radius, positive=True)
    time_slot_duration = _finite_number("time_slot_duration", time_slot_duration, positive=True)
    min_speed = _finite_number("min_speed", min_speed, nonnegative=True)
    max_speed = _finite_number("max_speed", max_speed, positive=True)
    if min_speed > max_speed:
        raise ValueError("min_speed must not exceed max_speed")
    space_ratio = _finite_number("space_ratio", space_ratio, positive=True)
    move_ratio = _finite_number("move_ratio", move_ratio, positive=True)
    if space_ratio >= 1.0:
        raise ValueError("space_ratio must lie strictly between 0 and 1 to keep rings inside sector boundaries")
    if move_ratio > 1.0:
        raise ValueError("move_ratio must lie in (0, 1]")
    if direction not in ("clockwise", "counterclockwise"):
        raise ValueError("direction must be 'clockwise' or 'counterclockwise'")
    target_height = _finite_number("target_height", target_height)
    try:
        center = np.asarray(center, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("center must contain two or three finite coordinates") from None
    if center.shape not in ((2,), (3,)) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two or three finite coordinates")
    origin = center[:2]

    if uavs_num == 1:
        centroid_distance = 0.0
        containment_radius = deploy_radius
        center_angles = np.zeros(1, dtype=float)
        start_angles = np.zeros(1, dtype=float)
    else:
        half_width = np.pi / uavs_num
        centroid_distance = 2.0 * deploy_radius * np.sin(half_width) / (3.0 * half_width)
        containment_radius = min(centroid_distance * np.sin(half_width),
                                 deploy_radius - centroid_distance)
        center_angles = (np.arange(uavs_num) + 0.5) * 2.0 * np.pi / uavs_num
        start_angles = np.mod(center_angles + np.pi, 2.0 * np.pi)

    reference_speed = 0.5 * (min_speed + max_speed)
    move_budget = move_ratio * hold_slots * time_slot_duration * reference_speed
    space_radius = space_ratio * containment_radius
    move_radius = None if n == 1 else move_budget / (2.0 * np.sin(np.pi / n))
    ring_radius = space_radius if move_radius is None else min(space_radius, move_radius)
    if not np.isfinite(ring_radius) or ring_radius <= 0 or not np.isfinite(move_budget):
        raise ValueError("sector_ring geometry produced a nonpositive or nonfinite radius/budget")

    circle_centers = np.zeros((uavs_num, 3), dtype=float)
    circle_centers[:, :2] = origin + centroid_distance * np.column_stack(
        (np.cos(center_angles), np.sin(center_angles)))
    circle_centers[:, 2] = target_height
    sign = -1.0 if direction == "clockwise" else 1.0
    angles = start_angles[:, None] + sign * np.arange(n)[None, :] * 2.0 * np.pi / n
    targets = np.repeat(circle_centers[:, None, :], n, axis=1)
    targets[:, :, 0] += ring_radius * np.cos(angles)
    targets[:, :, 1] += ring_radius * np.sin(angles)
    targets = targets.reshape(targets_num, 3)
    target_sector = np.repeat(np.arange(uavs_num, dtype=np.int64), n)
    adjacent_distance = 0.0 if n == 1 else 2.0 * ring_radius * np.sin(np.pi / n)
    layout_info = {
        "layout": "sector_ring",
        "centers": circle_centers.tolist(),
        "centroid_distance": float(centroid_distance),
        "ring_radius": float(ring_radius),
        "contain_radius": float(containment_radius),
        "space_radius": float(space_radius),
        "move_radius": None if move_radius is None else float(move_radius),
        "targets_per_sector": n,
        "adjacent_distance": float(adjacent_distance),
        "move_budget": float(move_budget),
        "reference_speed": float(reference_speed),
        "direction": direction,
        "start_angles": start_angles.tolist(),
        "deploy_radius": deploy_radius,
        "space_ratio": space_ratio,
        "move_ratio": move_ratio,
        "hold_slots": hold_slots,
        "time_slot_duration": time_slot_duration,
        "min_speed": min_speed,
        "max_speed": max_speed,
    }
    return targets, target_sector, layout_info
