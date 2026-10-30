"""Fixed, reproducible per-slot CU tasks; independent of episode/action RNGs."""
import hashlib

import numpy as np


CU_TASK_CONTRACT = "fixed_cu_tasks_fixed_slots_v2"
TASK_RNG_DOMAIN = 0x43555441
TASK_RANGES = {"l_kbit": (150.0, 170.0), "c": (800.0, 1000.0),
               "d": (0.55, 0.65)}


def task_settings(args):
    # Older programmatic callers retain their explicit legacy environment.
    profile = getattr(args, "cu_task_profile", "constant")
    if profile not in ("random_fixed", "constant"):
        raise ValueError("cu_task_profile must be random_fixed or constant")
    duration = getattr(args, "cu_task_slot_duration", 0.65)
    if isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("cu_task_slot_duration must be positive and finite")
    settings = {"profile": profile, "slot_duration": float(duration)}
    for name, defaults in TASK_RANGES.items():
        bounds = [getattr(args, "cu_task_" + name + suffix, default)
                  for suffix, default in zip(("_min", "_max"), defaults)]
        if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in bounds):
            raise ValueError("cu_task_" + name + " bounds must be positive and finite")
        if bounds[0] > bounds[1]:
            raise ValueError("cu_task_" + name + " minimum must not exceed maximum")
        settings[name] = [float(x) for x in bounds]
    if profile == "random_fixed" and settings["d"][1] > duration:
        raise ValueError("cu_task_d_max must not exceed cu_task_slot_duration")
    return settings


class CUTaskSchedule:
    def __init__(self, base_args, args, scene_seed_offset):
        self.settings = task_settings(args)
        slots, users = int(args.total_time_slots), int(base_args.cus_num)
        if slots <= 0 or users <= 0:
            raise ValueError("CU tasks require positive slot and CU counts")
        seed = np.random.SeedSequence(int(base_args.seed),
                                      spawn_key=(TASK_RNG_DOMAIN, int(scene_seed_offset)))
        rng = np.random.default_rng(seed)
        shape = (slots, users)
        if self.settings["profile"] == "random_fixed":
            self.l_bits = rng.uniform(*self.settings["l_kbit"], size=shape) * 1000.0
            self.c_cycles = rng.uniform(*self.settings["c"], size=shape)
            self.d_seconds = rng.uniform(*self.settings["d"], size=shape)
            self.slot_seconds = np.full(slots, self.settings["slot_duration"])
        else:
            self.l_bits = np.full(shape, float(getattr(base_args, "L_cu_task_bits", 170e3)))
            self.c_cycles = np.full(shape, float(base_args.bs_cycles_per_bit))
            self.d_seconds = np.full(shape, float(base_args.cu_max_delay))
            self.slot_seconds = np.full(slots, float(base_args.time_slot_duration))
        if (not np.isfinite(self.slot_seconds).all() or np.any(self.slot_seconds <= 0)
                or np.any(self.d_seconds > self.slot_seconds[:, None] + 1e-12)):
            raise ValueError("every CU deadline must fit inside its positive slot duration")
        self.time_boundaries = np.concatenate(([0.0], np.cumsum(self.slot_seconds)))
        for array in (self.l_bits, self.c_cycles, self.d_seconds,
                      self.slot_seconds, self.time_boundaries):
            array.setflags(write=False)

    def row(self, t):
        # Terminal observation repeats the final task, with no extra task arrival.
        t = min(int(t), len(self.slot_seconds) - 1)
        return self.l_bits[t], self.c_cycles[t], self.d_seconds[t], float(self.slot_seconds[t])

    def metadata(self):
        arrays = {"L_bits": self.l_bits, "C_cycles_per_bit": self.c_cycles,
                  "D_seconds": self.d_seconds, "slot_seconds": self.slot_seconds,
                  "time_boundaries_seconds": self.time_boundaries}
        return {"contract": CU_TASK_CONTRACT, **self.settings,
                "rng_domain": TASK_RNG_DOMAIN, "fixed_across_episodes": True,
                "slot_rule": "fixed_configured_duration" if self.settings["profile"] == "random_fixed"
                             else "base_time_slot_duration",
                "terminal_task_rule": "repeat_last_row_no_arrival",
                "episode_seconds": float(self.time_boundaries[-1]),
                "sha256": {key: hashlib.sha256(value.tobytes()).hexdigest()
                           for key, value in arrays.items()},
                **{key: value.tolist() for key, value in arrays.items()}}
