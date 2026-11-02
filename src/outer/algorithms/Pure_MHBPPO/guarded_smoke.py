"""Short-run entry that forbids optimization calls in parent and spawn workers."""
import multiprocessing as mp
import sys
from pathlib import Path

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))

import cvxpy
import mosek.fusion
import reward.cccp_bridge as bridge
import reward.my_reward as old_reward
import fusion_pc3p
from algorithms.Pure_MHBPPO.Pure_MHBPPO_main import get_pure_args, run_training


def forbidden(*args, **kwargs):
    raise AssertionError("optimization solver or old solver-backed reward was called")


# Executed in spawned worker interpreters as well as in the learner.
cvxpy.Problem.solve = forbidden
mosek.fusion.Model.solve = forbidden
bridge.solve_inner_energy = forbidden
bridge.run_pc3p = forbidden
old_reward.penalty_based_cccp = forbidden
old_reward.MyReward.reward_compute = forbidden
fusion_pc3p.run_pc3p_fusion = forbidden


if __name__ == "__main__":
    mp.freeze_support()
    run_training(get_pure_args())
