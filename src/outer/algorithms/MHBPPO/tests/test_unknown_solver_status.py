"""Unknown-only sensing certificates, status provenance, and bridge toggle."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from reward.solver_status import resolve_solver_status, sensing_infeasibility_certificate


def context(margin=0.5):
    return SimpleNamespace(I=1, N=2, P_max_uav=10.0, eps_sinr=1.0,
        Gamma_sinr=np.array([20.0]), G_sen_corr=np.array([np.diag([2*margin, 0.0])]))


class CertificateTests(unittest.TestCase):
    def test_certifies_with_actual_gamma_and_keeps_raw_status(self):
        ctx = context()
        # An unrelated diagnostic gamma is not used.
        ctx.diagnostic_gamma = np.array([1e-20])
        result = resolve_solver_status("ProblemStatus.Unknown", ctx,
            {"optimizer_response_code": 100006, "problem_status": "ProblemStatus.Unknown"})
        self.assertEqual(result["solver_status_kind"], "infeasible")
        self.assertEqual(result["raw_solver_status_kind"], "unknown")
        self.assertEqual(result["optimizer_response_code"], 100006)
        self.assertEqual(result["status_source"], "analytic_c03_c05_c06")
        self.assertEqual(result["infeasibility_certificate"]["failed_uavs"], [0])

    def test_boundary_and_feasible_sensing_remain_unknown(self):
        for margin in (1.0, 1.0-0.5e-6, 1.1):
            with self.subTest(margin=margin):
                result = resolve_solver_status("ProblemStatus.Unknown", context(margin))
                self.assertEqual(result["solver_status_kind"], "unknown")
                self.assertFalse(result["infeasibility_certificate"]["certified"])
        self.assertTrue(sensing_infeasibility_certificate(context(1-2e-6))["certified"])

    def test_non_unknown_never_reads_or_verifies_context(self):
        class Unreadable:
            def __getattribute__(self, name):
                raise AssertionError("non-unknown must not inspect inputs or optimal solution")
        for status, kind in (("ProblemStatus.PrimalAndDualFeasible", "optimal"),
                             ("optimal", "optimal"), ("ProblemStatus.PrimalInfeasible", "infeasible"),
                             ("error: test", "error"), (None, "other")):
            result = resolve_solver_status(status, Unreadable())
            self.assertEqual(result["solver_status_kind"], kind)
            self.assertEqual(result["solver_status"], str(status))
            self.assertNotIn("infeasibility_certificate", result)

    def test_invalid_input_is_inconclusive(self):
        cases = []
        for field, value in (("Gamma_sinr", np.array([np.nan])),
                             ("Gamma_sinr", np.array([0.0])),
                             ("Gamma_sinr", np.array([-1.0])),
                             ("Gamma_sinr", np.array([np.inf])),
                             ("Gamma_sinr", np.array([1e-310])),
                             ("G_sen_corr", np.full((1,2,2), np.nan)),
                             ("G_sen_corr", np.zeros((2,2))),
                             ("P_max_uav", -1.0), ("eps_sinr", np.inf)):
            ctx=context(); setattr(ctx, field, value); cases.append(ctx)
        cases.append(SimpleNamespace())
        for ctx in cases:
            result=resolve_solver_status("ProblemStatus.Unknown",ctx)
            self.assertEqual(result["solver_status_kind"],"unknown")
            self.assertFalse(result["infeasibility_certificate"]["evaluated"])

    def test_complex_channel_and_zero_channel(self):
        ctx=context()
        g=np.array([1,1j])/np.sqrt(2)
        ctx.G_sen_corr=np.array([np.outer(g,g.conj())])
        self.assertAlmostEqual(sensing_infeasibility_certificate(ctx)["min_margin"],0.5)
        ctx.G_sen_corr[:]=0
        self.assertTrue(sensing_infeasibility_certificate(ctx)["certified"])

    def test_scale_and_multiple_uav_do_not_hide_impossibility(self):
        ctx=context()
        ctx.G_sen_corr*=1e-14;ctx.Gamma_sinr*=1e-14
        cert=sensing_infeasibility_certificate(ctx)
        self.assertTrue(cert["certified"])
        self.assertAlmostEqual(cert["min_margin"],0.5)
        ctx=context();ctx.I=2
        ctx.G_sen_corr=np.array([np.diag([4.0,0]),np.diag([0.5,0])])
        ctx.Gamma_sinr=np.array([20.0,20.0])
        self.assertEqual(sensing_infeasibility_certificate(ctx)["failed_uavs"],[1])

    def test_certificate_is_read_only(self):
        ctx=context();beforeG=ctx.G_sen_corr.copy();beforeGamma=ctx.Gamma_sinr.copy()
        sensing_infeasibility_certificate(ctx)
        np.testing.assert_array_equal(ctx.G_sen_corr,beforeG)
        np.testing.assert_array_equal(ctx.Gamma_sinr,beforeGamma)

    def test_eigenvalue_failure_remains_unknown(self):
        with patch("reward.solver_status.np.linalg.eigvalsh",side_effect=np.linalg.LinAlgError):
            self.assertEqual(resolve_solver_status("unknown",context())["solver_status_kind"],"unknown")


class BridgeTests(unittest.TestCase):
    def test_unknown_is_resolved_with_diagnostics_on_and_off(self):
        from configs.base_params import get_base_args
        from reward import cccp_bridge as bridge
        args=get_base_args(I=1,J=1,K=1,N=2)
        kwargs=dict(args=args,uavs_2_cus_channels=np.array([[[10.,0.]]],complex),
            uavs_2_bs_channels=np.array([[[1.,0.]]],complex),cus_2_bs_channels=np.ones((1,1),complex),
            uavs_2_targets_channels=np.array([[np.diag([1.,0.])]],complex),
            uavs_targets_matched_matrix=np.ones((1,1)),uavs_cus_matched_matrix=np.ones((1,1)),
            uavs_pos_pre=np.array([[0.,0.,50.]]),uavs_pos_cur=np.array([[3.,0.,50.]]),
            uavs_off_duration=np.array([.05]),cus_off_power=np.array([.1]),
            cus_entertaining_task_size=np.array([170000.]),uavs_rec_beam_vectors=np.array([[1.,0.]],complex))
        raw={"status":"ProblemStatus.Unknown","W_sen_beam":None,"iterations":1,
             "solver_diagnostics":{"problem_status":"ProblemStatus.Unknown","optimizer_response_code":100006}}
        for enabled in (False, True):
            with patch.object(bridge,"run_pc3p",return_value=raw):
                result=bridge.solve_inner_energy(**kwargs,collect_diagnostics=enabled,diagnostic_candidates=False)
            self.assertTrue(np.isinf(result[0]))
            self.assertEqual(result[-1]["solver_status_kind"],"infeasible")
            self.assertEqual(result[-1]["raw_solver_status_kind"],"unknown")
            self.assertEqual("env_diagnostic" in result[-1], enabled)
            if enabled:
                self.assertEqual(result[-1]["env_diagnostic"]["solver"]["solver_status_kind"],"infeasible")
            self.assertEqual(raw["status"],"ProblemStatus.Unknown")


if __name__ == "__main__":
    unittest.main()
