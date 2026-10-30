"""Log-only energy accounting; no reward or solution acceptance changes."""

import math


ENERGY_STATISTICS_CONTRACT = "finite_and_strict_energy_sums_counts_v1"
ENERGY_COMPONENT_KEYS = (
    "bs_sensing_computation_energy_j", "bs_cu_computation_energy_j",
    "uav_sensing_energy_j", "uav_offloading_energy_j", "uav_flight_energy_j",
    "cu_offloading_energy_j",
)
ENERGY_VALUE_KEYS = ENERGY_COMPONENT_KEYS + ("physical_total_energy_j", "weighted_total_energy_j")
ENERGY_OUTCOME_KEYS = ENERGY_VALUE_KEYS + (
    "energy_contract", "energy_solution_status", "energy_finite_candidate",
    "energy_physical_solution_verified", "energy_strict_feasible",
)
ENERGY_COHORTS = ("finite_candidate", "strict_feasible")


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _empty_statistics():
    return {
        "contract": ENERGY_STATISTICS_CONTRACT,
        "slot_count": 0,
        "strict_verification_unknown_count": 0,
        "cohorts": {
            cohort: {"count": 0, "values": {
                key: {"sum": 0.0, "count": 0, "mean": float("nan")}
                for key in ENERGY_VALUE_KEYS}}
            for cohort in ENERGY_COHORTS},
        # A failed optimization still consumes the already chosen flight energy.
        "all_slots_flight": {"sum": 0.0, "count": 0, "mean": float("nan")},
    }


def _finish(statistics):
    values = [item for cohort in statistics["cohorts"].values()
              for item in cohort["values"].values()]
    values.append(statistics["all_slots_flight"])
    for item in values:
        item["mean"] = item["sum"] / item["count"] if item["count"] else float("nan")
    return statistics


def summarize_energy(traces):
    """Keep candidate/strict denominators explicit, including unavailable fields.

    Finite refers to the returned objective, independently of business success.
    Strict requires the explicit physical-and-outer-check flag from the reward
    diagnostics. Missing/disabled verification is unknown, never counted as pass.
    The legacy objective remains the weighted-value source for exact continuity.
    """
    result = _empty_statistics()
    for trace in traces:
        outcome = trace.get("outcome", {})
        result["slot_count"] += 1
        finite = _finite(outcome.get("obj_fun"))
        strict = outcome.get("energy_strict_feasible")
        if strict is None:
            result["strict_verification_unknown_count"] += 1
        flight = outcome.get("uav_flight_energy_j")
        if _finite(flight):
            result["all_slots_flight"]["sum"] += float(flight)
            result["all_slots_flight"]["count"] += 1
        for cohort, included in (("finite_candidate", finite),
                                 ("strict_feasible", finite and strict is True)):
            if not included:
                continue
            stats = result["cohorts"][cohort]
            stats["count"] += 1
            for key in ENERGY_VALUE_KEYS:
                value = outcome.get("obj_fun") if key == "weighted_total_energy_j" else outcome.get(key)
                if _finite(value):
                    stats["values"][key]["sum"] += float(value)
                    stats["values"][key]["count"] += 1
    return _finish(result)


def merge_energy_statistics(summaries):
    """Merge workers by sums/counts, not by averaging per-worker means."""
    result = _empty_statistics()
    for summary in summaries:
        if summary.get("contract") != ENERGY_STATISTICS_CONTRACT:
            raise ValueError("incompatible energy statistics contract")
        result["slot_count"] += int(summary["slot_count"])
        result["strict_verification_unknown_count"] += int(summary["strict_verification_unknown_count"])
        for cohort in ENERGY_COHORTS:
            target = result["cohorts"][cohort]
            source = summary["cohorts"][cohort]
            target["count"] += int(source["count"])
            for key in ENERGY_VALUE_KEYS:
                target["values"][key]["sum"] += float(source["values"][key]["sum"])
                target["values"][key]["count"] += int(source["values"][key]["count"])
        for field in ("sum", "count"):
            result["all_slots_flight"][field] += summary["all_slots_flight"][field]
    return _finish(result)


def energy_scalar_metrics(statistics):
    """Stable flat schema shared by training_updates.csv and TensorBoard."""
    result = {"energy_slot_count": statistics["slot_count"],
              "energy_strict_verification_unknown_count": statistics["strict_verification_unknown_count"]}
    for cohort in ENERGY_COHORTS:
        values = statistics["cohorts"][cohort]
        result[f"energy_{cohort}_count"] = values["count"]
        for key in ENERGY_VALUE_KEYS:
            for field in ("mean", "count", "sum"):
                result[f"energy_{cohort}_{key}_{field}"] = values["values"][key][field]
    for field in ("mean", "count", "sum"):
        result[f"energy_all_slots_uav_flight_energy_j_{field}"] = statistics["all_slots_flight"][field]
    return result


def log_energy_statistics(writer, statistics, iteration):
    for key, value in energy_scalar_metrics(statistics).items():
        writer.add_scalar("Energy/" + key.removeprefix("energy_"), value, iteration)
