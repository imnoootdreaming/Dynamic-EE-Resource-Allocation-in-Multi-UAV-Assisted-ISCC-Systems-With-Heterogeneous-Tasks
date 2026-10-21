"""Weight identity must survive differing learner/sampler reduction threads."""
import copy
import math
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from algorithms.MHBPPO.MHBPPO_main import actor_checksum, broadcast_weights, StateNormStats
from algorithms.MHBPPO.MHBPPO_sampler import _actor_checksum, _load_weights_until
from algorithms.MHBPPO.rollout_batch import actor_parameter_checksum, actor_state_signature
from algorithms.MHBPPO.tests.test_parallel_rollout import episode, validate


class ActorChecksumTests(unittest.TestCase):
    def setUp(self):
        self.original_threads = torch.get_num_threads()

    def tearDown(self):
        torch.set_num_threads(self.original_threads)

    def test_cross_thread_checksums_equal_without_changing_actor_or_rng(self):
        generator = torch.Generator().manual_seed(42)
        actor = torch.nn.Module()
        actor.register_parameter("weight", torch.nn.Parameter(torch.randn(128, 1017, generator=generator)))
        actor.register_parameter("bias", torch.nn.Parameter(torch.randn(128, generator=generator)))
        state_before = copy.deepcopy(actor.state_dict())
        rng_before = torch.get_rng_state().clone()
        expected = round(math.fsum(float(x) for p in actor.parameters() for x in p.detach().abs().reshape(-1)), 6)
        observed = []
        for threads in (1, 2, 6):
            torch.set_num_threads(threads)
            observed.extend((actor_checksum(actor), _actor_checksum(actor), actor_parameter_checksum(actor)))
            self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(set(observed), {expected})
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
            self.assertEqual(value.dtype, torch.float32)

    def test_noncontiguous_parameters_have_the_same_canonical_checksum(self):
        values = torch.tensor([[1.25, -2.5], [3.125, -4.75], [5.5, -6.875]]).t()
        self.assertFalse(values.is_contiguous())
        a, b = torch.nn.Module(), torch.nn.Module()
        a.register_parameter("weight", torch.nn.Parameter(values))
        b.register_parameter("weight", torch.nn.Parameter(values.contiguous()))
        self.assertEqual(actor_checksum(a), _actor_checksum(b))
        self.assertEqual(actor_state_signature(a), actor_state_signature(b))

    def test_broadcast_snapshot_remains_identical_after_learner_changes(self):
        learner = torch.nn.Linear(1017, 128)
        worker = copy.deepcopy(learner)
        q = queue.Queue()
        stats = StateNormStats(1017)
        torch.set_num_threads(6)
        checksum = broadcast_weights([q], SimpleNamespace(actor=learner), version=670, state_norm_stats=stats)
        expected_signature = actor_state_signature(learner)
        with torch.no_grad():
            learner.weight.add_(1.)
        torch.set_num_threads(1)
        version, mean, std = _load_weights_until(q, worker, 670, threading.Event(), strict_version=True)
        self.assertEqual(version, 670)
        self.assertEqual(_actor_checksum(worker), checksum)
        self.assertEqual(actor_state_signature(worker), expected_signature)
        self.assertNotEqual(actor_state_signature(learner), expected_signature)
        np.testing.assert_array_equal(mean, stats.mean)
        np.testing.assert_array_equal(std, stats.std)

    def test_checksum_and_version_failures_remain_hard_errors(self):
        for value in (None, float("nan"), float("inf"), 2.002):
            item = episode(0)
            item["weights_checksum"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "checksum mismatch: worker=0"):
                validate(item)
        item = episode(0)
        item["weights_version"] = 1
        with self.assertRaisesRegex(ValueError, "wrong sampling policy version"):
            validate(item)

    def test_changed_weights_with_equal_scalar_sum_still_fail_sha(self):
        from algorithms.MHBPPO.rollout_batch import validate_sync_episode
        a, b = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            a.weight.copy_(torch.tensor([[1., -2.]]))
            b.weight.copy_(-a.weight)
        self.assertEqual(actor_checksum(a), _actor_checksum(b))
        self.assertNotEqual(actor_state_signature(a), actor_state_signature(b))
        item = episode(0)
        item["weights_checksum"] = _actor_checksum(b)
        item["actor_signature"] = actor_state_signature(b)
        with self.assertRaisesRegex(ValueError, "sampling actor SHA mismatch: worker=0"):
            validate_sync_episode(item, expected_round=0, expected_worker_ids=[0], expected_policy_version=0,
                expected_actor_checksum=actor_checksum(a), expected_actor_signature=actor_state_signature(a),
                expected_norm_signature=item["norm_signature"], expected_scene_signature=item["scene_signature"],
                total_time_slots=3)


if __name__ == "__main__":
    unittest.main()
