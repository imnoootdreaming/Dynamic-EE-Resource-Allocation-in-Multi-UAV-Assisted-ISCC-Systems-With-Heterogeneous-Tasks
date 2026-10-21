"""Private receiver encoder routing, freezing, gradient and probability tests."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.distributions import Beta,kl_divergence

OUTER=Path(__file__).resolve().parents[3]
if str(OUTER) not in sys.path:sys.path.insert(0,str(OUTER))
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.beam_branch import build_branch_actor,branch_dists,train_branch
from algorithms.MHBPPO.beam_pretraining import isolated_torch_rng,teacher_beta_parameters,beam_dists
from algorithms.MHBPPO.checkpoint import SavedStateNormalization
from algorithms.MHBPPO.validate_beam_branch import BeamEvaluationView

SPLITS={"uav_angles":2,"uav_distances":2,"uav_off_durations":2,"cu_off_powers":4,
        "uav_rec_beam_dir_real":6,"uav_rec_beam_dir_imag":6}


def source_actor():
    with isolated_torch_rng(42):return MultiHeadActor(7,16,SPLITS,[4,4])


def check_distribution_equal(test,first,second):
    for name in first[0]:
        for attr in ("concentration1","concentration0"):
            test.assertTrue(torch.equal(getattr(first[0][name],attr),getattr(second[0][name],attr)),name)
    for a,b in zip(first[1],second[1]):test.assertTrue(torch.equal(a.logits,b.logits))


class BeamBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.source=source_actor()
        self.states=torch.tensor(np.random.default_rng(42).normal(size=(16,7)),dtype=torch.float32)

    def test_copied_encoder_initial_policy_and_storage_independence(self):
        rng=torch.get_rng_state().clone()
        branch=build_branch_actor(self.source)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        check_distribution_equal(self,self.source._get_dists(self.states),branch._get_dists(self.states))
        torch.testing.assert_close(branch.beam_features(self.states),self.source._forward_features(self.states),atol=0,rtol=0)
        for name in ("beam_fc1","beam_fc2"):
            private=getattr(branch,name);old=getattr(branch,name[5:])
            self.assertNotEqual(private.weight.data_ptr(),old.weight.data_ptr())
            self.assertTrue(torch.equal(private.weight,old.weight))
        with isolated_torch_rng(987):a=self.source.sample(self.states)
        with isolated_torch_rng(987):b=branch.sample(self.states)
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        with self.assertRaises(RuntimeError):self.source.load_state_dict(branch.state_dict(),strict=True)

    def test_gradients_reach_private_encoder_and_nonbeam_is_unchanged(self):
        branch=build_branch_actor(self.source)
        old={n:x.clone() for n,x in branch.state_dict().items()}
        source_old={n:x.clone() for n,x in self.source.state_dict().items()}
        allowed=lambda name:name.startswith("beam_fc") or any(name.startswith("continuous_heads."+h+".") for h in branch.RECEIVE_BEAM_HEADS)
        params=[]
        for name,param in branch.named_parameters():
            param.requires_grad_(allowed(name))
            if allowed(name):params.append(param)
        self.assertEqual(len(params),12)
        real,imag=branch_dists(branch,self.states)
        target=np.ones((len(self.states),2,3))/np.sqrt(3)
        teacher=teacher_beta_parameters(target,np.zeros_like(target))
        q=Beta(torch.tensor(teacher["alpha"],dtype=torch.float32),torch.tensor(teacher["beta"],dtype=torch.float32))
        p=Beta(torch.cat((real.concentration1,imag.concentration1),1),torch.cat((real.concentration0,imag.concentration0),1))
        optimizer=torch.optim.Adam(params,lr=.01)
        optimizer.zero_grad(set_to_none=True);kl_divergence(q,p).mean().backward()
        self.assertGreater(float(branch.beam_fc1.weight.grad.abs().sum()),0)
        self.assertGreater(float(branch.beam_fc2.weight.grad.abs().sum()),0)
        self.assertIsNone(branch.fc1.weight.grad)
        optimizer.step()
        for name,value in branch.state_dict().items():
            if not allowed(name):self.assertTrue(torch.equal(old[name],value),name)
        for name,value in self.source.state_dict().items():self.assertTrue(torch.equal(source_old[name],value),name)
        first=self.source._get_dists(self.states);second=branch._get_dists(self.states)
        for name in first[0]:
            if name not in branch.RECEIVE_BEAM_HEADS:
                self.assertTrue(torch.equal(first[0][name].concentration1,second[0][name].concentration1))
        for a,b in zip(first[1],second[1]):self.assertTrue(torch.equal(a.logits,b.logits))
        self.assertFalse(torch.equal(branch.beam_fc1.weight,old["beam_fc1.weight"]))

    def test_full_policy_and_beam_only_evaluation_use_same_private_path(self):
        branch=build_branch_actor(self.source)
        with torch.no_grad():branch.beam_fc2.bias.add_(.7)
        full,_=branch._get_dists(self.states)
        direct=branch_dists(branch,self.states)
        view=beam_dists(BeamEvaluationView(branch),self.states)
        for index,name in enumerate(("uav_rec_beam_dir_real","uav_rec_beam_dir_imag")):
            for dist in (direct[index],view[index]):
                self.assertTrue(torch.equal(dist.concentration1,full[name].concentration1))
                self.assertTrue(torch.equal(dist.concentration0,full[name].concentration0))
        self.assertFalse(torch.equal(direct[0].concentration1,beam_dists(self.source,self.states)[0].concentration1))
        with torch.no_grad(),isolated_torch_rng(42):
            sample=branch.sample(self.states)
            evaluated=branch.evaluate_actions(self.states,sample[0],sample[1])
        for a,b in ((sample[2],evaluated[0]),(sample[3],evaluated[2]),(sample[4],evaluated[3])):
            torch.testing.assert_close(a,b,atol=4e-5,rtol=2e-5)

    def test_bounded_training_ignores_test_restores_flags_modes_and_rng(self):
        branch=build_branch_actor(self.source)
        branch.train();branch.fc1.eval()
        branch.fc1.weight.requires_grad_(False)
        flags={n:p.requires_grad for n,p in branch.named_parameters()}
        modes={n:m.training for n,m in branch.named_modules()}
        frozen={n:x.clone() for n,x in self.source.state_dict().items() if not any(n.startswith("continuous_heads."+h+".") for h in self.source.RECEIVE_BEAM_HEADS)}
        raw=np.random.default_rng(42).normal(size=(24,7)).astype(np.float32)
        target=np.ones((24,2,3))/np.sqrt(3)
        data={"raw_states":raw,"teacher_g_real":target,"teacher_g_imag":np.zeros_like(target),"split":np.repeat([0,1,2],8)}
        data["raw_states"][16:]=np.nan;data["teacher_g_real"][16:]=np.nan
        normalizer=SavedStateNormalization(np.zeros(7),np.ones(7),20.,np.full(7,20.),0)
        rng=torch.get_rng_state().clone()
        result=train_branch(branch,data,normalizer,epochs=3,patience=2,batch_size=4,validation_draws=4)
        self.assertFalse(result["test_evaluated"])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(flags,{n:p.requires_grad for n,p in branch.named_parameters()})
        self.assertEqual(modes,{n:m.training for n,m in branch.named_modules()})
        for name,value in frozen.items():self.assertTrue(torch.equal(value,branch.state_dict()[name]))


if __name__=="__main__":unittest.main()
