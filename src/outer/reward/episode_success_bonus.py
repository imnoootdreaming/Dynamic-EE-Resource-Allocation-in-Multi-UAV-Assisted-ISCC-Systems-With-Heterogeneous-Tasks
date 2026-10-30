"""Terminal reward for an episode whose every slot satisfies the environment success flag."""

import math


EPISODE_SUCCESS_BONUS_CONTRACT = "terminal_all_slots_legacy_success_v1"


def validate_episode_success_bonus(value):
    """Return a finite, nonnegative terminal bonus coefficient."""
    if isinstance(value, bool):
        raise ValueError("episode_success_bonus must be nonnegative and finite")
    coefficient = float(value)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("episode_success_bonus must be nonnegative and finite")
    return coefficient


def apply_episode_success_bonus(total_reward, reward, *, slot_success,
                                episode_all_success_before, is_terminal, coefficient):
    """Apply the bonus once, on the terminal transition, iff all slots succeeded.

    The caller owns the episode history.  This function records the complete
    reward accounting in ``reward["components"]`` and keeps ``reward["bs"]``
    aligned with the scalar returned to PPO.
    """
    coefficient = validate_episode_success_bonus(coefficient)
    slot_success = bool(slot_success)
    episode_all_success = bool(episode_all_success_before) and slot_success
    applied = coefficient if bool(is_terminal) and episode_all_success else 0.0
    reward_before_bonus = float(total_reward)
    reward_after_bonus = reward_before_bonus + applied

    components = dict(reward.get("components", {}))
    components.update({
        "slot_success": int(slot_success),
        "episode_all_success": episode_all_success,
        "slot_reward_before_episode_bonus": reward_before_bonus,
        "episode_success_bonus": coefficient,
        "episode_success_bonus_applied": applied,
        "episode_success_bonus_contract": EPISODE_SUCCESS_BONUS_CONTRACT,
    })
    reward["components"] = components
    reward["bs"] = reward_after_bonus
    return reward_after_bonus, episode_all_success, applied
