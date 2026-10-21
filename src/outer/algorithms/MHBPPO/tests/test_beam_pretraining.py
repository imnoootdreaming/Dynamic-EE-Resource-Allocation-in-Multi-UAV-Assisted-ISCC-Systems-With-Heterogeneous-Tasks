"""Receiver supervision contracts; no environment or optimization solver."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

OUTER=Path(__file__).resolve().parents[3]
if str(OUTER) not in sys.path:sys.path.insert(0,str(OUTER))
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.beam_pretraining import (
    BEAM_HEAD_NAMES, beam_dists, build_actor_from_checkpoint, head_only_training,
    isolated_torch_rng, load_dataset, named_beam_parameters, teacher_beta_parameters,
    tensor_hashes, train_receive_beam,
)
from algorithms.MHBPPO.checkpoint import SavedStateNormalization,save_state_normalization
from algorithms.MHBPPO.validate_receive_beam import complex_beams,sensing_metrics


SPLITS={"uav_angles":2,"uav_distances":2,"uav_off_durations":2,"cu_off_powers":4,
        "uav_rec_beam_dir_real":6,"uav_rec_beam_dir_imag":6}


def actor():
    with isolated_torch_rng(42):return MultiHeadActor(7,16,SPLITS,[4,4])


def norm():
    return SavedStateNormalization(np.zeros(7),np.ones(7),20.0,np.full(7,20.0),0)


def arrays():
    rng=np.random.default_rng(42)
    raw=rng.normal(size=(24,7)).astype(np.float32)
    target=np.ones((24,2,3),dtype=complex)/np.sqrt(3)
    return {"raw_states":raw,"teacher_g_real":target.real,"teacher_g_imag":target.imag,
        "split":np.repeat([0,1,2],8),"uav_target_distances":np.full((24,2),100.),
        "physical_cu_power":np.full((24,4),.01),"physical_matching":np.zeros((24,2),int),
        "episode_idx":np.repeat([0,1,2],8),"collection_round":np.repeat([0,1,2],8),
        "worker_id":np.zeros(24,int),"time_slot":np.tile(np.arange(8),3)}


class BeamPretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_teacher_is_finite_representable_and_keeps_layout(self):
        phase=np.array([0.,.4,-.8])
        g=np.exp(1j*phase)[None,None,:]/np.sqrt(3)
        g=np.repeat(g,2,axis=1)
        teacher=teacher_beta_parameters(g.real,g.imag)
        expected=np.concatenate((g.real.reshape(1,-1),g.imag.reshape(1,-1)),axis=1)
        np.testing.assert_allclose(teacher["mean"],(1+.9*expected)/2)
        self.assertTrue(np.all(teacher["alpha"]>1) and np.all(teacher["beta"]>1))
        self.assertTrue(np.all(teacher["alpha"]+teacher["beta"]==64))
        for kwargs in ({"rho":1},{"rho":0},{"kappa":1}):
            with self.assertRaises(ValueError):teacher_beta_parameters(g.real,g.imag,**kwargs)
        q=Beta(torch.tensor(teacher["alpha"]),torch.tensor(teacher["beta"]))
        torch.testing.assert_close(kl_divergence(q,q),torch.zeros_like(q.mean),atol=0,rtol=0)

    def test_signed_teacher_has_gradient_and_updates_only_beam(self):
        model=actor();data=arrays()
        frozen=tensor_hashes(model);before=tensor_hashes(model,beam=True)
        state=torch.tensor(data["raw_states"][:8])
        target=teacher_beta_parameters(data["teacher_g_real"][:8],data["teacher_g_imag"][:8])
        with head_only_training(model) as params:
            real,imag=beam_dists(model,state)
            alpha,beta=torch.tensor(target["alpha"],dtype=torch.float32),torch.tensor(target["beta"],dtype=torch.float32)
            q=Beta(alpha,beta)
            p=Beta(torch.cat((real.concentration1,imag.concentration1),1),torch.cat((real.concentration0,imag.concentration0),1))
            loss=kl_divergence(q,p).mean()
            optimizer=torch.optim.Adam(params,lr=.01)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in params),0)
            self.assertIsNone(model.fc1.weight.grad)
            optimizer.step()
        self.assertEqual(tensor_hashes(model),frozen)
        self.assertNotEqual(tensor_hashes(model,beam=True),before)

    def test_failure_restores_flags_modes_gradients_and_rng(self):
        model=actor();model.train();model.fc1.eval()
        model.fc1.weight.requires_grad_(False)
        sample=next(iter(named_beam_parameters(model).values()))
        sample.grad=torch.full_like(sample,3)
        flags={n:p.requires_grad for n,p in model.named_parameters()}
        modes={n:m.training for n,m in model.named_modules()}
        rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,"sentinel"):
            with isolated_torch_rng(777),head_only_training(model):
                torch.rand(4)
                raise RuntimeError("sentinel")
        self.assertEqual(flags,{n:p.requires_grad for n,p in model.named_parameters()})
        self.assertEqual(modes,{n:m.training for n,m in model.named_modules()})
        torch.testing.assert_close(sample.grad,torch.full_like(sample,3),atol=0,rtol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_fit_does_not_consume_reserved_test_or_modify_normalizer(self):
        model=actor();data=arrays();normalizer=norm()
        # Invalid sentinels are deliberately outside the selected train/val slices.
        data["raw_states"][data["split"]==2]=np.nan
        data["teacher_g_real"][data["split"]==2]=np.nan
        original={key:getattr(normalizer,key).copy() for key in ("mean","std","m2")}
        frozen=tensor_hashes(model);rng=torch.get_rng_state().clone()
        result=train_receive_beam(model,data,normalizer,epochs=3,patience=2,batch_size=4,validation_draws=4)
        self.assertFalse(result["test_evaluated"])
        self.assertTrue(result["nonbeam_parameters_unchanged"])
        self.assertEqual(tensor_hashes(model),frozen)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in original.items():np.testing.assert_array_equal(value,getattr(normalizer,key))

    def test_dataset_hash_and_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.npz";data=arrays()
            np.savez_compressed(path,**data)
            manifest={"dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix(".json").write_text(json.dumps(manifest))
            loaded,_=load_dataset(path)
            self.assertEqual(len(loaded["raw_states"]),24)
            manifest["dataset_sha256"]="0"*64
            path.with_suffix(".json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_dataset(path)
            data["raw_states"][8]=data["raw_states"][0]
            np.savez_compressed(path,**data)
            manifest["dataset_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"duplicate"):load_dataset(path)

    def test_checkpoint_rng_pairing_and_exact_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);model=actor();normalizer=norm()
            torch.save(model.state_dict(),source/"mhbppo_actor.pth")
            (source/"mhbppo_critic.pth").write_bytes(b"unused test fixture; never deserialized")
            save_state_normalization(source/"state_normalization.npz",normalizer,0,actor_path=source/"mhbppo_actor.pth")
            metadata={"iteration":0,"architecture":{"state_dim":7,"hidden_dim":16,"continuous_action_splits":SPLITS,"discrete_action_dims":[4,4],"continuous_dist_type":"beta","structured_beam_alpha":1}}
            (source/"checkpoint_metadata.json").write_text(json.dumps(metadata))
            rng=torch.get_rng_state().clone()
            loaded,paired,_=build_actor_from_checkpoint(source)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(tensor_hashes(loaded),tensor_hashes(model))
            np.testing.assert_array_equal(paired.m2,normalizer.m2)
            payload=torch.load(source/"mhbppo_actor.pth",weights_only=True)
            payload["fc1.bias"][0]+=1
            torch.save(payload,source/"mhbppo_actor.pth")
            with self.assertRaisesRegex(ValueError,"normalization"):build_actor_from_checkpoint(source)

    def test_physical_layout_phase_invariance_and_dot_convention(self):
        raw=np.array([[[1.,0.,0.],[0.,1.,0.]]])+1j*np.array([[[0.,1.,0.],[0.,0.,1.]]])
        raw/=np.linalg.norm(raw,axis=-1,keepdims=True)
        real=torch.tensor((1+.9*raw.real.reshape(1,-1))/2)
        imag=torch.tensor((1+.9*raw.imag.reshape(1,-1))/2)
        g,bad=complex_beams(real,imag,2,3)
        np.testing.assert_allclose(g.numpy(),raw,atol=1e-15)
        self.assertFalse(bad.any())
        h=np.array([[[1+2j,3-1j,1j],[1-1j,2j,3+1j]]])
        params={"sigma_2":.2,"xi_0":1.,"rho_ref":1.,"P_max_uav":1.,"eps_sinr":1.}
        values=sensing_metrics(raw[None],raw,np.full((1,2),2.),h,np.full((1,2),.3),params)
        rotated=sensing_metrics(raw[None]*np.exp(.7j),raw,np.full((1,2),2.),h,np.full((1,2),.3),params)
        scaled=sensing_metrics(raw[None]*(1+2e-7),raw,np.full((1,2),2.),h,np.full((1,2),.3),params)
        np.testing.assert_allclose(values["gain"],1,atol=1e-14)
        expected=.3*np.abs(np.sum(raw*h,axis=-1))**2+.2
        np.testing.assert_allclose(values["gamma"][0],expected,atol=1e-14)
        for key in values:np.testing.assert_allclose(values[key],rotated[key],atol=1e-14)
        for key in values:np.testing.assert_allclose(values[key],scaled[key],atol=1e-14)


if __name__=="__main__":unittest.main()
