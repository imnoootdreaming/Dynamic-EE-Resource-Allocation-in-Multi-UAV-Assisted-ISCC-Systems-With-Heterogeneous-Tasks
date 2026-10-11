"""Read-only original constraint diagnostics for the inner CSV entry points."""
import importlib.util
from pathlib import Path
import sys

_ALIAS = "_inner_csv_feasibility_diagnostics"
if _ALIAS not in sys.modules:
    path = Path(__file__).resolve().parents[1] / "outer/reward/feasibility_diagnostics.py"
    spec = importlib.util.spec_from_file_location(_ALIAS, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_ALIAS] = module
_validator = sys.modules[_ALIAS].validate_original_solution

CHECK_FIELDS = ["covariance_passed", "recovered_beams_passed", "rank_passed",
                "physical_solution_verified", "max_normalized_violation", "failed_checks"]

def solution_checks(ctx, result):
    report = _validator(ctx.params, ctx.environment, result)
    covariance, recovered = report["covariance"], report["recovered_beams"]
    return dict(covariance_passed=int(covariance["passed"]),
                recovered_beams_passed=int(recovered["passed"]),
                rank_passed=int(report["rank"]["passed"]),
                physical_solution_verified=int(report["physical_solution_verified"]),
                max_normalized_violation=max(covariance.get("max_normalized_violation", float("nan")),
                                             recovered.get("max_normalized_violation", float("nan"))),
                failed_checks=";".join(["covariance:" + k for k in covariance["failed_checks"]]
                                       + ["recovered:" + k for k in recovered["failed_checks"]]
                                       + ([] if report["rank"]["passed"] else ["rank"])) )
