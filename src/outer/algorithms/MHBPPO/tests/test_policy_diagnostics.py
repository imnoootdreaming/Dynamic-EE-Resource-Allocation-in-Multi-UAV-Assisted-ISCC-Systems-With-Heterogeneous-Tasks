"""Policy diagnostics are conditional moments and never alter sampled training data."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch
from torch.distributions import Beta,Categorical

OUTER=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(OUTER))
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.MHBPPO_sampler import _choose_action
from algorithms.MHBPPO.policy_diagnostics import (capture_policy_distribution,
    single_state_policy_distribution,summarize_policy_distribution,log_policy_distribution)
from algorithms.MHBPPO.action_logging import ActionLogWriter

PARTS=dict(uav_angles=2,uav_distances=2,uav_off_durations=2,cu_off_powers=3,
           uav_rec_beam_dir_real=4,uav_rec_beam_dir_imag=4)


def actor(kind='beta',alpha=0):
    return MultiHeadActor(7,16,PARTS,[3,3],continuous_dist_type=kind,structured_beam_alpha=alpha)


class PolicyDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_beta_analytic_moments_and_categorical_probabilities(self):
        beta=Beta(torch.tensor([[2.,5.]]),torch.tensor([[5.,2.]]))
        cat=Categorical(probs=torch.tensor([[.1,.2,.7]]))
        stats=single_state_policy_distribution(capture_policy_distribution(
            {'power':beta},[cat],['power'],'beta'))
        np.testing.assert_allclose(stats['continuous']['power']['mean'],[2/7,5/7],rtol=1e-7)
        np.testing.assert_allclose(stats['continuous']['power']['std'],[np.sqrt(10/392)]*2,rtol=1e-7)
        np.testing.assert_array_equal(stats['continuous']['power']['concentration'],[7,7])
        np.testing.assert_allclose(stats['discrete']['cu_matching_0']['probabilities'],[.1,.2,.7],rtol=1e-7)
        self.assertAlmostEqual(stats['discrete']['cu_matching_0']['entropy'],
                               -sum(p*np.log(p) for p in [.1,.2,.7]),places=6)

    def test_optional_capture_preserves_actions_log_probs_rng_and_gradients(self):
        for kind in ('beta','gaussian'):
            for mixing in (0.,.5):
                model=actor(kind,mixing);states=torch.randn(3,7)
                torch.manual_seed(301);model.zero_grad(set_to_none=True)
                before=model.sample(states);before[2].sum().backward()
                grads={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
                rng=torch.get_rng_state().clone()
                torch.manual_seed(301);model.zero_grad(set_to_none=True)
                with patch.object(model,'_get_dists',wraps=model._get_dists) as forward:
                    after=model.sample(states,return_distribution_stats=True)
                self.assertEqual(forward.call_count,1)
                after[2].sum().backward()
                for a,b in zip(before,after[:5]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for n,p in model.named_parameters():
                    if grads[n] is None:self.assertIsNone(p.grad)
                    else:torch.testing.assert_close(grads[n],p.grad,rtol=0,atol=0)
                for head in model.masked_continuous_head_names:
                    self.assertFalse(after[5]['continuous'][head]['active'])
                    self.assertNotIn('std',after[5]['continuous'][head])

    def test_sampler_optional_return_preserves_default_interface(self):
        model=actor();state=np.ones(7)
        torch.manual_seed(41);before=_choose_action(model,state)
        torch.manual_seed(41);after=_choose_action(model,state,return_distribution_stats=True)
        self.assertEqual(len(before),3);self.assertEqual(len(after),4)
        for key in ('continuous','discrete'):np.testing.assert_array_equal(before[0][key],after[0][key])
        self.assertEqual(before[1:],after[1:3])
        self.assertEqual(after[3]['continuous']['cu_off_powers']['space'],'unit_interval')

    def traces(self):
        traces=[]
        for mu in (.1,.9):
            traces.append(dict(clipped_action={'continuous':[mu]},
                policy_distribution=dict(contract='sampling_policy_conditional_moments_v1',distribution='beta',
                    continuous={'power':dict(active=True,dim=1,space='unit_interval',mean=[mu],std=[.05],entropy=[-.3],concentration=[10.])},
                    discrete={'cu_matching_0':dict(probabilities=[.25,.75],entropy=.562335)})))
        return traces

    def test_conditional_std_is_distinct_from_state_and_sample_variation(self):
        summary=summarize_policy_distribution(self.traces(),{'power':1})
        head=summary['continuous']['power']
        self.assertAlmostEqual(head['mean'][0],.5)
        self.assertAlmostEqual(head['std'][0],.05)
        self.assertAlmostEqual(head['state_mean_std'][0],.4)
        self.assertAlmostEqual(head['sampled_action_std'][0],.4)
        self.assertAlmostEqual(head['marginal_std'][0],np.sqrt(.05**2+.4**2))
        self.assertEqual(summary['steps'],2)

    def test_jsonl_and_tensorboard_preserve_component_and_policy_version(self):
        traces=self.traces();summary=summarize_policy_distribution(traces,{'power':1})
        logger=Mock();writer=Mock();log_policy_distribution(logger,writer,summary,12,11)
        self.assertIn('wv=11',logger.info.call_args_list[0].args[0])
        self.assertIn('std=[0.0500]',logger.info.call_args_list[0].args[0])
        self.assertIn(('PolicyDistribution/power/std/dim_0',.05,12),[call.args for call in writer.add_scalar.call_args_list])
        item=dict(sampler_id=0,seed_offset=1,episode_idx=0,weights_version=11,
                  action_traces=traces,transition_dict=dict(continuous_actions=[[.1],[.9]],discrete_actions=[[1],[1]],rewards=[0.,1.]))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'actions.jsonl';aw=ActionLogWriter(str(p),{'run_id':'test'})
            try:aw.write_episode(item,12)
            finally:aw.close()
            records=[json.loads(line) for line in p.read_text().splitlines()][1:]
        self.assertEqual([r['weights_version'] for r in records],[11,11])
        self.assertEqual(records[0]['environment_action']['policy_distribution']['continuous']['power']['std'],[.05])

    def test_legacy_records_and_partial_records_are_explicit(self):
        self.assertFalse(summarize_policy_distribution([{}],{'power':1})['available'])
        with self.assertRaises(ValueError):summarize_policy_distribution(self.traces()+[{}],{'power':1})
        logger=Mock();writer=Mock();log_policy_distribution(logger,writer,{},1,0)
        logger.info.assert_not_called();writer.add_scalar.assert_not_called()

    def test_gaussian_moments_are_labeled_pre_sigmoid(self):
        _,_,_,stats=_choose_action(actor('gaussian'),np.ones(7),return_distribution_stats=True)
        self.assertEqual(stats['continuous']['uav_angles']['space'],'pre_sigmoid')
        self.assertNotIn('concentration',stats['continuous']['uav_angles'])


if __name__=='__main__':unittest.main()
