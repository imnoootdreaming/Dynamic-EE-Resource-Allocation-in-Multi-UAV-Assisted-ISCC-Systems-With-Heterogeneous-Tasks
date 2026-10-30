"""动作/索引 → 匹配矩阵、接收波束与信道扁平化等映射工具。

从 `environment/my_env.py` 抽离的、与环境状态机无关的纯函数：

- `build_uavs_cus_matched_matrix`：离散 CU 索引动作 → UAV-CU 匹配矩阵 eta
- `build_uav_targets_matched_matrix`：UAV→目标索引 → UAV-目标匹配矩阵
- `build_unit_norm_rec_beam`：幅度/相位 → 逐 UAV 单位范数接收波束 g（‖g_i‖²=1）
- `flatten_complex`：复数信道 → 拼接的实/虚部 float32 向量（观测用）

所需维度参数显式传入，与环境状态机无关。
"""

import numpy as np


def receive_beam_mapping_metadata(mode="absolute", limit_deg=45.0, alpha=1.0,
                                  mvdr_residual=False, residual_radius=0.2,
                                  structured_beam_reference="mvdr"):
    """Validate physical action semantics; missing options retain legacy mapping."""
    if mode not in ("absolute", "phase_residual"):
        raise ValueError("receive_beam_phase_mode must be absolute or phase_residual")
    limit_deg = float(limit_deg)
    if not np.isfinite(limit_deg) or not 0 < limit_deg <= 180:
        raise ValueError("receive_beam_phase_limit_deg must be finite and in (0, 180]")
    if mode == "phase_residual" and float(alpha) != 1.0:
        raise ValueError("phase_residual requires structured_beam_alpha=1")
    if not np.isfinite(float(alpha)):
        raise ValueError("structured_beam_alpha must be finite")
    if not np.isfinite(float(residual_radius)) or not 0 < float(residual_radius) <= 1:
        raise ValueError("mvdr_residual_radius must be finite and in (0,1]")
    if mvdr_residual and mode != "absolute":
        raise ValueError("mvdr_residual requires receive_beam_phase_mode=absolute")
    if structured_beam_reference not in ("mvdr", "matched"):
        raise ValueError("structured_beam_reference must be mvdr or matched")
    if structured_beam_reference == "matched" and (float(alpha) != 0.0 or mvdr_residual):
        raise ValueError("matched reference requires structured_beam_alpha=0 and mvdr_residual disabled")
    return {
        "receive_beam_phase_mode": mode,
        "receive_beam_mapping_version": (
            "matched_unit_norm_v1" if structured_beam_reference == "matched" else
            "mvdr_complex_residual_v1" if mvdr_residual else
            "mvdr_unit_norm_v1" if float(alpha) <= 0.0 else
            "mvdr_aligned_mix_v1" if float(alpha) < 1.0 else
            "target_phase_residual_v1" if mode == "phase_residual" else "absolute_v1"),
        "receive_beam_phase_limit_deg": limit_deg if mode == "phase_residual" else None,
        **({"structured_beam_reference": "matched"} if structured_beam_reference == "matched" else {}),
        **({"mvdr_residual_radius": float(residual_radius)} if mvdr_residual else {}),
    }


def build_current_target_steering(uavs_2_targets_channels, target_indices):
    """Canonical a_r/sqrt(N), shared by observations and residual actions.

    The current model is A=beta*a*a^H, beta>0 and a[0]=1. Normalizing
    A[:,0] avoids the arbitrary global phase of an eigensolver.
    """
    channels = np.asarray(uavs_2_targets_channels, dtype=np.complex128)
    if (channels.ndim != 4 or min(channels.shape) <= 0 or
            channels.shape[2] != channels.shape[3]):
        raise ValueError("current target steering requires sensing channels shaped (I,K,N,N)")
    indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if indices.size != channels.shape[0]:
        raise ValueError("current target steering requires one target index per UAV")
    indices = np.clip(indices, 0, channels.shape[1] - 1)
    selected = channels[np.arange(channels.shape[0]), indices]
    if not np.all(np.isfinite(selected)):
        raise ValueError("current target steering requires finite sensing channels")
    column = selected[:, :, 0]
    scale = np.maximum(np.max(np.abs(column.real), axis=1, keepdims=True),
                       np.max(np.abs(column.imag), axis=1, keepdims=True))
    if np.any(scale == 0):
        raise ValueError("current target steering is undefined for a zero sensing column")
    # Scale real/imag separately before squaring, including subnormal channels.
    scaled = column.real / scale + 1j * (column.imag / scale)
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def build_phase_residual_rec_beam(amplitude_flat, residual_flat, reference):
    """Nonnegative amplitudes and residual radians around a unit steering beam."""
    reference = np.asarray(reference, dtype=np.complex128)
    if (reference.ndim != 2 or not np.all(np.isfinite(reference)) or
            not np.allclose(np.linalg.norm(reference, axis=1), 1.0)):
        raise ValueError("phase_residual requires finite unit reference beams")
    amplitudes = np.asarray(amplitude_flat, dtype=np.float64).reshape(reference.shape)
    residuals = np.asarray(residual_flat, dtype=np.float64).reshape(reference.shape)
    if (not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0) or
            not np.all(np.isfinite(residuals))):
        raise ValueError("phase_residual requires finite nonnegative amplitudes and finite phases")
    direction = amplitudes * reference * np.exp(1j * residuals)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    beams = direction / np.where(norms > 1e-12, norms, 1.0)
    degenerate = norms[:, 0] <= 1e-12
    beams[degenerate] = reference[degenerate]
    return beams


def build_uavs_cus_matched_matrix(discrete_actions, uavs_num, cus_num):
    """将离散 CU 索引动作转换为 UAV-CU 匹配矩阵（每 UAV 行 one-hot）。"""
    discrete_actions = np.clip(np.asarray(discrete_actions, dtype=np.int64), 0, cus_num - 1)
    uavs_cus_matched_matrix = np.zeros((uavs_num, cus_num), dtype=np.float32)
    for uav_idx, cu_idx in enumerate(discrete_actions):
        uavs_cus_matched_matrix[uav_idx, int(cu_idx)] = 1.0
    return uavs_cus_matched_matrix


def build_uav_targets_matched_matrix(target_indices, uavs_num, targets_num):
    """将 UAV→目标索引转换为 UAV-目标匹配矩阵（每 UAV 行 one-hot）。"""
    matched_matrix = np.zeros((uavs_num, targets_num), dtype=np.float32)
    clipped_target_indices = np.clip(
        np.asarray(target_indices, dtype=np.int64),
        0,
        targets_num - 1
    )
    for i, target_idx in enumerate(clipped_target_indices):
        matched_matrix[i, target_idx] = 1.0
    return matched_matrix


def build_unit_norm_rec_beam(amplitude_flat, phase_flat, uavs_num, antenna_nums):
    """由幅度和弧度相位构造单位范数接收波束 g_i（‖g_i‖²=1）。

    :param amplitude_flat: (I*N,) 非负幅度，将被逐 UAV L2 归一化
    :param phase_flat: (I*N,) 相位，单位 rad
    :return: (I, N) complex ndarray，逐 UAV 单位范数
    """
    beams = np.zeros((uavs_num, antenna_nums), dtype=complex)
    for i in range(uavs_num):
        start, end = i * antenna_nums, (i + 1) * antenna_nums
        direction = amplitude_flat[start:end] * np.exp(1j * phase_flat[start:end])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:  # 退化保护：方向近零时回退基向量 e_1
            direction = np.zeros(antenna_nums, dtype=complex)
            direction[0] = 1.0 + 0j
            norm = 1.0
        beams[i] = direction / norm
    return beams


def build_mvdr_residual_rec_beam(reference, real_actions, imag_actions, radius):
    """Bound a complex latent perturbation orthogonally to each unit reference."""
    reference = np.asarray(reference, dtype=np.complex128)
    if (reference.ndim != 2 or not np.all(np.isfinite(reference)) or
            not np.allclose(np.linalg.norm(reference, axis=1), 1.0)):
        raise ValueError("MVDR residual requires finite unit reference beams")
    radius = float(radius)
    if not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError("MVDR residual radius must be in (0,1]")
    real = np.asarray(real_actions, dtype=np.float64).reshape(reference.shape)
    imag = np.asarray(imag_actions, dtype=np.float64).reshape(reference.shape)
    if (not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)) or
            np.any(real < 0) or np.any(real > 1) or np.any(imag < 0) or np.any(imag > 1)):
        raise ValueError("MVDR residual latent actions must be finite in [0,1]")
    latent = (2 * real - 1) + 1j * (2 * imag - 1)
    projected = latent - reference * np.sum(reference.conj() * latent, axis=1, keepdims=True)
    residual = radius * projected / np.maximum(1.0, np.linalg.norm(projected, axis=1, keepdims=True))
    direction = reference + residual
    return direction / np.linalg.norm(direction, axis=1, keepdims=True)


def mix_reference_rec_beam(reference, actor_beam, alpha):
    """Align irrelevant global phase before normalized MVDR/actor mixing."""
    reference = np.asarray(reference, dtype=np.complex128)
    actor_beam = np.asarray(actor_beam, dtype=np.complex128)
    if reference.shape != actor_beam.shape or reference.ndim != 2:
        raise ValueError("beam mix requires matching (UAV, antenna) shapes")
    overlap = np.sum(reference.conj() * actor_beam, axis=1, keepdims=True)
    phase = np.ones_like(overlap)
    np.divide(overlap.conj(), np.abs(overlap), out=phase, where=np.abs(overlap) > 1e-12)
    mixed = (1.0 - alpha) * reference + alpha * actor_beam * phase
    return mixed / np.linalg.norm(mixed, axis=1, keepdims=True)


def build_matched_rec_beam(uavs_2_targets_channels, target_indices):
    """Return per-UAV matched receive beams for the selected sensing targets.

    The result is deterministic given the current sensing channel and target
    schedule.  It is used as a structured initialization/reference for the
    actor's receive-beam head; it does not alter the actor's latent action or
    its PPO log-probability.
    """
    channels = np.asarray(uavs_2_targets_channels, dtype=complex)
    if channels.ndim != 4:
        raise ValueError("uavs_2_targets_channels 应为 (I, K, N, N)")
    I, K, N, N2 = channels.shape
    if N != N2:
        raise ValueError("感知信道矩阵必须为方阵")
    indices = np.clip(np.asarray(target_indices, dtype=np.int64).reshape(-1), 0, K - 1)
    if indices.size != I:
        raise ValueError("target_indices 长度必须等于 UAV 数量")
    beams = np.zeros((I, N), dtype=complex)
    for i, target_idx in enumerate(indices):
        A = channels[i, int(target_idx)]
        hermitian = (A + A.conj().T) / 2.0
        _, vectors = np.linalg.eigh(hermitian)
        beam = vectors[:, -1]
        norm = np.linalg.norm(beam)
        beams[i] = beam / norm if norm > 1e-12 else np.eye(N, dtype=complex)[0]
    return beams


def build_mvdr_rec_beam(uavs_2_targets_channels, target_indices,
                       cus_2_uavs_channels, eta_share, p_cu_power, noise_power):
    """Unit-norm MVDR for the current rank-one sensing model and CU actions.

    ref/ref.tex defines h_CU,UAV as a row, so |g^H h^H|^2 = |g^T h|^2.
    R = sigma^2 I + sum_j eta_ij p_j conj(h_ij) h_ij^T; never replace
    this by h h^H. Normalize the optimal direction to ||g||=1 as required
    by the environment, rather than the distortionless MVDR normalization.
    """
    reference = build_current_target_steering(uavs_2_targets_channels, target_indices)
    h = np.asarray(cus_2_uavs_channels, dtype=np.complex128)
    eta = np.asarray(eta_share, dtype=np.float64)
    power = np.asarray(p_cu_power, dtype=np.float64)
    noise = float(noise_power)
    uavs, antennas = reference.shape
    if h.ndim != 3 or h.shape[0] != uavs or h.shape[2] != antennas:
        raise ValueError("MVDR CU channels must have shape (I,J,N)")
    if eta.shape != h.shape[:2] or power.shape != (h.shape[1],):
        raise ValueError("MVDR matching/power shapes must be (I,J) and (J,)")
    if (not np.isfinite(noise) or noise <= 0 or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(eta)) or np.any(eta < 0)
            or not np.all(np.isfinite(power)) or np.any(power < 0)):
        raise ValueError("MVDR needs finite channels, positive noise and nonnegative weights")
    weights = eta * power[None, :]
    covariance = noise * np.eye(antennas, dtype=np.complex128)[None, :, :] + np.einsum(
        "ij,ijn,ijm->inm", weights, h.conj(), h)
    # Common per-UAV scaling preserves the solution direction and keeps the
    # small SI-valued covariance away from underflow. Physical noise is kept.
    scale = np.max(np.abs(covariance), axis=(1, 2))
    direction = np.linalg.solve(covariance / scale[:, None, None], reference[..., None])[..., 0]
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    if not np.all(np.isfinite(direction)) or np.any(norms <= 0):
        raise ValueError("MVDR returned a nonfinite or zero direction")
    return direction / norms


def flatten_complex(channel):
    """复数信道 → 实部/虚部顺序拼接的 float32 向量。"""
    return np.concatenate([channel.real.flatten(), channel.imag.flatten()]).astype(np.float32)
