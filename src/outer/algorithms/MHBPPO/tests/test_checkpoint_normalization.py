"""Disk checkpoint roundtrips must reproduce sampler state normalization."""
import queue
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch

OUTER=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(OUTER))
from algorithms.MHBPPO.MHBPPO_main import StateNormStats,broadcast_weights,save_artifacts
from algorithms.MHBPPO.checkpoint import (save_state_normalization,load_state_normalization,
                                         STATE_NORMALIZATION_FILENAME)


class CheckpointNormalizationTests(unittest.TestCase):
    def test_saved_snapshot_matches_broadcast_and_sampler_transform_exactly(self):
        stats=StateNormStats(3)
        states=np.array([[1.,2.,0.],[3.,6.,0.],[5.,9.,0.]],dtype=np.float32)
        stats.update(states)
        agent=SimpleNamespace(actor=torch.nn.Linear(3,2),critic=torch.nn.Linear(3,1))
        q=queue.Queue();broadcast_weights([q],agent,version=37,state_norm_stats=stats)
        payload=q.get()
        with tempfile.TemporaryDirectory() as directory:
            _,actor_path,_=save_artifacts(directory,1208,37,[.5],[1.],[100.],agent,state_norm_stats=stats)
            norm=load_state_normalization(Path(actor_path).parent/STATE_NORMALIZATION_FILENAME,
                                         expected_dim=3,expected_iteration=37)
            np.testing.assert_array_equal(norm.mean,payload['state_norm_mean'])
            np.testing.assert_array_equal(norm.std,payload['state_norm_std'])
            raw=np.array([2.,4.,0.],dtype=np.float32)
            sampled=(raw-payload['state_norm_mean'])/(np.maximum(payload['state_norm_std'],1e-8)+1e-8)
            np.testing.assert_array_equal(norm.normalize(raw),sampled)
            restored=torch.nn.Linear(3,2)
            restored.load_state_dict(torch.load(actor_path,map_location='cpu',weights_only=True))
            obs=torch.as_tensor(sampled,dtype=torch.float32)
            torch.testing.assert_close(restored(obs),agent.actor(obs),rtol=0,atol=0)
            self.assertEqual(norm.count,stats.count)
            self.assertFalse(norm.mean.flags.writeable)
            self.assertFalse(norm.std.flags.writeable)
            self.assertFalse(Path(actor_path).parent.joinpath(STATE_NORMALIZATION_FILENAME+'.tmp').exists())

    def test_stored_welford_moments_allow_equivalent_stats_update(self):
        stats=StateNormStats(2);stats.update([[1.,2.],[3.,7.]])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/STATE_NORMALIZATION_FILENAME;save_state_normalization(p,stats,8)
            norm=load_state_normalization(p)
            restored=StateNormStats(2)
            restored.mean=norm.mean.copy();restored.m2=norm.m2.copy();restored.count=norm.count
            more=np.array([[9.,4.],[10.,2.]])
            restored.update(more);stats.update(more)
            np.testing.assert_array_equal(restored.mean,stats.mean)
            np.testing.assert_array_equal(restored.m2,stats.m2)
            np.testing.assert_array_equal(restored.std,stats.std)

    def test_initial_and_constant_feature_std_match_training_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/STATE_NORMALIZATION_FILENAME
            for initialized in (False,True):
                stats=StateNormStats(2)
                if initialized:stats.update(np.zeros((4,2)))
                save_state_normalization(p,stats,0)
                norm=load_state_normalization(p)
                np.testing.assert_array_equal(norm.std,np.ones(2) if not initialized else np.full(2,1e-4))
                self.assertTrue(np.all(np.isfinite(norm.normalize([1.,1.]))))
                before=norm.count;norm.normalize(np.ones((5,2)))
                self.assertEqual(norm.count,before)

    def test_dimension_and_iteration_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/STATE_NORMALIZATION_FILENAME
            save_state_normalization(p,StateNormStats(2),12)
            with self.assertRaises(ValueError):load_state_normalization(p,expected_dim=3)
            with self.assertRaises(ValueError):load_state_normalization(p,expected_iteration=13)
            norm=load_state_normalization(p)
            with self.assertRaises(ValueError):norm.normalize(np.zeros(3))

    def test_mismatched_actor_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/STATE_NORMALIZATION_FILENAME;a=Path(directory)/'mhbppo_actor.pth'
            torch.save({'weight':torch.ones(2)},a)
            save_state_normalization(p,StateNormStats(2),2,actor_path=a)
            load_state_normalization(p)
            torch.save({'weight':torch.zeros(2)},a)
            with self.assertRaisesRegex(ValueError,'do not match'):load_state_normalization(p)

    def test_missing_legacy_normalizer_does_not_silently_use_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError,'cannot infer'):
                load_state_normalization(Path(directory)/STATE_NORMALIZATION_FILENAME)

    def test_invalid_stats_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/STATE_NORMALIZATION_FILENAME
            stats=StateNormStats(2);stats.mean[0]=np.nan
            with self.assertRaises(ValueError):save_state_normalization(p,stats,1)
            with self.assertRaises(ValueError):save_state_normalization(p,None,1)


if __name__=='__main__':unittest.main()
