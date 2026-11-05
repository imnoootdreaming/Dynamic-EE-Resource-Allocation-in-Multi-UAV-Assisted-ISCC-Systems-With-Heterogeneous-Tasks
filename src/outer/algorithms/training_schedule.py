"""Training round limits and the PPO learning-rate floor."""

from itertools import count


def configure_ppo_lr_horizon(args):
    """Use the finite training budget for PPO decay, or episodes for an open-ended run."""
    if args.max_updates > 0:
        if args.episodes != args.max_updates:
            print(f"[training] PPO lr_decay_horizon={args.max_updates} from max_updates; "
                  f"episodes={args.episodes} is ignored for this finite run", flush=True)
        args.episodes = args.max_updates
    return args


def training_rounds(max_updates):
    """Yield zero-based rounds; zero means no training-round limit."""
    if max_updates < 0:
        raise ValueError("max_updates must be nonnegative")
    return range(max_updates) if max_updates else count()


def ppo_lr_multiplier(step, horizon, min_ratio):
    """Match MHBPPO's linear decay and nonzero floor."""
    return max(min_ratio, max(0.0, 1.0 - step / horizon))
