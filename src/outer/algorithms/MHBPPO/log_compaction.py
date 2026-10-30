"""Lossy diagnostic projections only; never apply to PPO transitions or rewards."""

LOG_COMPACTION_CONTRACT = "physical_steps_and_episode_policy_summary_v1"


def validate_logging_settings(args):
    if args.log_detail_level not in ("compact", "full"):
        raise ValueError("log_detail_level must be compact or full")
    for name in ("log_rotate_iterations", "log_rotate_max_bytes", "log_rotate_backup_segments",
                 "log_min_free_bytes", "log_policy_vector_interval"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")


def _certificate(value):
    if not isinstance(value, dict):
        return value
    keys = ("evaluated", "certified", "scope", "reason", "relative_tolerance",
            "failed_uavs", "min_margin", "margin", "status")
    return {key: value[key] for key in keys if key in value}


def _trim_certificates(record):
    return {key: _certificate(value) if key in ("presolve_certificate", "infeasibility_certificate")
            else value for key, value in record.items()}


def compact_action_trace(trace, *, keep_policy=False):
    """Keep executed beams/trajectory/actions and outcomes; drop alternative encodings."""
    keys = ("captured_at", "time_slot", "cu_tasks", "physical_action", "structured_beam_alpha",
            "receive_beam_diagnostics", "outcome")
    result = {key: trace[key] for key in keys if key in trace}
    if "outcome" in result:
        result["outcome"] = _trim_certificates(result["outcome"])
    # Policy summaries still need raw samples and per-state moments inside the worker.
    if keep_policy:
        for key in ("clipped_action", "policy_distribution"):
            if key in trace:
                result[key] = trace[key]
    return result


def _compact_checks(record):
    if not isinstance(record, dict):
        return record
    result = dict(record)
    if isinstance(record.get("checks"), dict):
        checks = {}
        for name, check in record["checks"].items():
            # Failing/unknown checks retain numerical details for diagnosis.
            if check.get("passed") is True:
                check = {key: value for key, value in check.items()
                         if key not in ("raw_residual", "normalized_residual", "scale")}
            checks[name] = check
        result["checks"] = checks
    return result


def compact_env_trace(trace):
    """Preserve feasibility, physical validity and sensing numbers for every slot."""
    result = {key: value for key, value in trace.items() if key != "problem_snapshot"}
    if "sensing" in result:
        sensing = {key: value for key, value in result["sensing"].items()
                   if key != "interference_by_cu_w"}
        if isinstance(sensing.get("candidates"), dict):
            sensing["candidates"] = {key: value for key, value in sensing["candidates"].items()
                                     if key not in ("current_power_margin", "min_power_margin")}
        result["sensing"] = sensing
    for key in ("solver", "reward_components"):
        if key in result:
            result[key] = _trim_certificates(result[key])
    if "outer_action" in result:
        result["outer_action"] = _compact_checks(result["outer_action"])
    if "validation" in result:
        result["validation"] = {key: _compact_checks(value)
                                for key, value in result["validation"].items()}
    return result


def compact_episode_payload(item):
    """Called only after worker summaries: reduce queue copies, preserving transitions."""
    result = dict(item)
    result["log_detail_level"] = "compact"
    result["action_traces"] = [compact_action_trace(trace) for trace in item["action_traces"]]
    result["diagnostic_traces"] = [compact_env_trace(trace) for trace in item.get("diagnostic_traces", [])]
    return result
