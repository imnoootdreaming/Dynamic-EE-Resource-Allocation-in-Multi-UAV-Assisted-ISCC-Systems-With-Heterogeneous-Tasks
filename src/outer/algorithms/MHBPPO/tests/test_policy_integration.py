"""Configuration round trips and architecture identity at the rollout boundary."""
import contextlib
import copy
import io
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from configs.madrl_args import get_madrl_args
from algorithms.MHBPPO.MHBPPO_agent import HPPO
from algorithms.MHBPPO.MHBPPO_main import broadcast_weights,save_hyperparameters,StateNormStats
from algorithms.MHBPPO.MHBPPO_sampler import _load_latest_weights,_validate_payload_architecture
from algorithms.MHBPPO.policy_initialization import actor_architecture_metadata,convert_agent_to_dedicated


SPLITS={"uav_angles":2,"uav_distances":2,"uav_off_durations":2,"cu_off_powers":4,
        "uav_rec_beam_dir_real":6,"uav_rec_beam_dir_imag":6}


def agent():
    return HPPO(7,16,SPLITS,[4,4],np.zeros(22),np.ones(22),.0003,.0003,.95,.2,.99,5,10000,"cpu")


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_default_settings_and_explicit_conflicts(self):
        args=get_madrl_args([])
        self.assertEqual((args.receive_beam_encoder,args.policy_initialization,args.max_updates),("shared","",0))
        for invalid in (["--max_updates","-1"],
                        ["--policy_initialization","x","--bc_dataset","y"],
                        ["--policy_initialization","x","--no-sync_training"],
                        ["--receive_beam_encoder","dedicated","--structured_beam_alpha",".5"]):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):get_madrl_args(invalid)

    def test_saved_recipe_round_trip_keeps_registered_fields(self):
        args=get_madrl_args(["--receive_beam_encoder","dedicated","--max_updates","10","--policy_initialization","example"])
        learner=agent();learner.policy_initialization_metadata=convert_agent_to_dedicated(learner)
        base=SimpleNamespace(seed=42,_params=SimpleNamespace(a=1),as_dict=lambda:{"seed":42})
        with tempfile.TemporaryDirectory() as temp:
            path=save_hyperparameters(temp,base,args,learner)
            restored=get_madrl_args(["--config",path])
            self.assertEqual(restored.receive_beam_encoder,"dedicated")
            self.assertEqual(restored.max_updates,10)
            self.assertEqual(restored.policy_initialization,"example")
            payload=json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(payload["effective"]["actor_architecture"]["actor_class"],"DedicatedBeamActor")

    def test_dedicated_broadcast_strictly_loads_same_class_and_rejects_shared(self):
        learner=agent();shared=copy.deepcopy(learner.actor)
        convert_agent_to_dedicated(learner)
        destination=copy.deepcopy(learner.actor)
        q=queue.Queue()
        broadcast_weights([q],learner,version=0,state_norm_stats=StateNormStats(7))
        payload=q.get_nowait()
        self.assertEqual(payload["actor_architecture"],actor_architecture_metadata(destination))
        with self.assertRaises(ValueError):_validate_payload_architecture(payload,shared)
        q.put(payload);version,mean,std=_load_latest_weights(q,destination)
        self.assertEqual(version,0)
        for name,tensor in learner.actor.state_dict().items():self.assertTrue(torch.equal(tensor,destination.state_dict()[name]))
        np.testing.assert_array_equal(mean,np.zeros(7));np.testing.assert_array_equal(std,np.ones(7))
        x=torch.zeros(4,7)
        with torch.no_grad():
            sample=destination.sample(x)
            evaluated=destination.evaluate_actions(x,sample[0],sample[1])
        torch.testing.assert_close(sample[2],evaluated[0],atol=4e-5,rtol=2e-5)


if __name__=="__main__":unittest.main()
