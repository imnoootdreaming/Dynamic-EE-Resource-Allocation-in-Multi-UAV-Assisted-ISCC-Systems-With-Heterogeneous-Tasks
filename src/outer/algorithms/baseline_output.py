"""Output and scene contracts shared by legacy baseline trainers."""
import functools
import hashlib
import json
import os
import shutil
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.MHBPPO.action_logging import ActionLogWriter, JsonlRotation, _json_value
from algorithms.MHBPPO.log_compaction import compact_action_trace, compact_env_trace
from algorithms.MHBPPO.energy_statistics import summarize_energy, energy_scalar_metrics
from algorithms.background_logging import BACKGROUND_LOG_ENV


class _BackgroundLogSink:
    """Baseline messages are already printed to the shell's unified run log."""

    def write(self, text):
        return len(text)

    def flush(self):
        pass


def training_log_stream(path):
    if os.environ.get(BACKGROUND_LOG_ENV):
        return nullcontext(_BackgroundLogSink())
    return Path(path).open("w", encoding="utf-8")


def log_training_failure(path, traceback_text):
    if not os.environ.get(BACKGROUND_LOG_ENV):
        with Path(path).open("a", encoding="utf-8") as failure_log:
            failure_log.write("[ERROR] training interrupted or failed:\n" + traceback_text + "\n")


def fixed_scene(args):
    return not any(getattr(args, key, False) for key in (
        'randomize_layout_per_episode', 'randomize_cu_traj_per_episode', 'randomize_nlos_per_episode'))


def check_scene(expected, actual, args):
    if fixed_scene(args) and actual['signature'] != expected['signature']:
        raise RuntimeError('worker exogenous scene differs from startup scene')


def capture(env, reward, total, objective, success, done, args):
    action = getattr(env, 'last_action_trace', None)
    diagnostic = getattr(env, 'last_env_diagnostic', None)
    if action is not None:
        action = {**action, 'outcome': {**reward.get('components', {}),
                  'total_reward': float(total), 'raw_bs_reward': float(reward['bs']),
                  'obj_fun': float(objective), 'success': int(success), 'done': bool(done)}}
        if getattr(args, 'log_detail_level', 'full') == 'compact' and 'physical_action' in action:
            action = compact_action_trace(action)
    if diagnostic and getattr(args, 'log_detail_level', 'full') == 'compact' and 'outer_action' in diagnostic:
        diagnostic = compact_env_trace(diagnostic)
    return action, diagnostic


def aggregate_metrics(pairs):
    actions = [action for action, _ in pairs if action is not None]
    pure = bool(actions and 'physical' in actions[0])
    physical = [action['physical'] for action in actions] if pure else []
    if pure:
        actions = [{'outcome': {'obj_fun': a['physical'].get('candidate_energy'),
                   'energy_strict_feasible': a['physical'].get('feasible'),
                   'uav_flight_energy_j': (a['physical'].get('energy') or {}).get('flight_j')}} for a in actions]
    metrics = energy_scalar_metrics(summarize_energy(actions))
    metrics['energy_interpretation'] = 'direct_candidate' if pure else 'inner_solver_candidate'
    diagnostics = [diag for _, diag in pairs if isinstance(diag, dict) and isinstance(diag.get('solver'), dict)]
    metrics['diagnostic_slots'] = len(diagnostics)
    times = [diag['solver'].get('solve_seconds') for diag in diagnostics]
    times = [float(t) for t in times if isinstance(t, (int, float)) and np.isfinite(t)]
    metrics['solver_seconds'] = sum(times) if times else ''
    metrics['solver_seconds_slots'] = len(times)
    outcomes = [diag.get('validation', {}).get('physical_solution_verified') for diag in diagnostics]
    known = [v for v in outcomes if isinstance(v, (bool, np.bool_))]
    metrics['physical_verification_known_slots'] = len(known)
    metrics['physical_verification_passed_slots'] = sum(bool(v) for v in known)
    checks = [item.get('passed') for p in physical for item in (p.get('checks') or {}).values()]
    checks = [v for v in checks if isinstance(v, (bool, np.bool_))]
    metrics['direct_constraint_checks_known'] = len(checks)
    metrics['direct_constraint_checks_passed'] = sum(bool(v) for v in checks)
    return metrics


METRIC_COLUMNS = list(aggregate_metrics([]))


def log_scalar_metrics(writer, metrics, iteration):
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            writer.add_scalar('Diagnostics/' + key, value, iteration)


def setup_run(outer, name, slug, args, probe, output_dir=None):
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f') + '_' + uuid.uuid4().hex[:8]
    background_log = os.environ.get(BACKGROUND_LOG_ENV) if output_dir is None else None
    if background_log:
        run_id = Path(background_log).parent.name
    if output_dir is not None:
        ckpt = Path(output_dir)
        log_dir, tb_dir = ckpt / 'logs', ckpt / 'tensorboard'
        run_id = ckpt.name
    else:
        ckpt = outer / 'ckpt' / name / run_id
        # The shared parser's default is specific to MHBPPO. Respect explicit custom paths.
        configured = Path(args.log_file)
        default = Path('logs/MHBPPO/mhbppo_train.log')
        if background_log:
            log_dir = Path(background_log).parent
        elif configured == default:
            log_dir = outer / 'logs' / name / run_id
        else:
            log_dir = (configured if configured.is_absolute() else outer / configured).parent / run_id
        tb_dir = outer / 'runs' / slug / run_id
    ckpt.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=bool(background_log))
    tb_dir.mkdir(parents=True, exist_ok=False)
    log_name = (Path(background_log).name if background_log else
                f'{name.lower()}_train.log' if Path(args.log_file) == Path('logs/MHBPPO/mhbppo_train.log') else Path(args.log_file).name)
    signature = probe.get_exogenous_scene_signature()
    scene = {'target_positions': probe.init_targets_pos.copy(), 'uav_positions': probe.init_uavs_pos.copy(),
             'target_sector': probe.target_sector.copy(), 'target_layout': probe.target_layout_info,
             'center': np.asarray(probe.base_args.center).copy(), 'deploy_radius': float(probe.base_args.radius),
             'bs_position': np.asarray(probe.base_args.bs_pos).copy()}
    from algorithms.MHBPPO.MHBPPO_main import save_environment_plot
    save_environment_plot(str(ckpt), scene)
    scene_file = {'algorithm': name, 'run_id': run_id, 'initial_exogenous_scene': signature,
                  'base_args': probe.base_args.as_dict(),
                  'inner_parameter_source': (vars(probe.base_args._params).copy()
                                             if getattr(probe.base_args, '_params', None) is not None else None),
                  'initial_cu_positions': probe.init_cus_pos, 'plot_scene': scene,
                  'cu_tasks': probe.cu_tasks.metadata(), 'fixed_scene': fixed_scene(args),
                  'randomization': {key: bool(getattr(args, key, False)) for key in (
                      'randomize_layout_per_episode', 'randomize_cu_traj_per_episode', 'randomize_nlos_per_episode')}}
    (ckpt / 'scene_signature.json').write_text(json.dumps(_json_value(scene_file), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    paths = {'run_id': run_id, 'checkpoint_dir': str(ckpt.resolve()), 'log_dir': str(log_dir.resolve()),
             'tensorboard_dir': str(tb_dir.resolve()), 'training_log': str((log_dir / log_name).resolve())}
    print(f'[{name}] checkpoint={ckpt.resolve()} | logs={log_dir.resolve()} | tensorboard={tb_dir.resolve()} | scene={signature["signature"]}', flush=True)
    return ckpt, log_dir, tb_dir, paths, signature


def atomic_checkpoint(function):
    """Publish a completed snapshot once; the marker is written before the atomic rename."""
    @functools.wraps(function)
    def save(output, iteration, *args, **kwargs):
        output = Path(output)
        destination = output / f'iter_{iteration}'
        if destination.exists():
            if (destination / 'COMPLETE').is_file():
                return
            raise RuntimeError(f'incomplete checkpoint already exists: {destination}')
        staging = output / ('.checkpoint_' + uuid.uuid4().hex)
        staging.mkdir()
        try:
            shutil.copyfile(output / 'training_updates.csv', staging / 'training_updates.csv')
            function(staging, iteration, *args, **kwargs)
            snapshot = staging / f'iter_{iteration}'
            metadata_file = snapshot / 'checkpoint_metadata.json'
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            metadata['file_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in snapshot.iterdir() if p.is_file() and p.name != metadata_file.name}
            metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            (snapshot / 'COMPLETE').write_text('complete\n', encoding='utf-8')
            os.replace(snapshot, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return save


class TraceLogs:
    def __init__(self, directory, name, metadata, args):
        self.args = args
        self.directory = Path(directory)
        self.paused = False
        self.rotation = JsonlRotation(args.log_rotate_iterations, args.log_rotate_max_bytes, args.log_rotate_backup_segments)
        self.action = ActionLogWriter(str(self.directory / f'{name}_actions.jsonl'), metadata)
        self.env = ActionLogWriter(str(self.directory / f'{name}_env.jsonl'), metadata) if args.env_diagnostics else None

    def begin_round(self, iteration):
        paused = bool(self.args.log_min_free_bytes and shutil.disk_usage(self.directory).free < self.args.log_min_free_bytes)
        writers = [w for w in (self.action, self.env) if w]
        index = 0
        if paused and not self.paused and self.rotation.active:
            index = self.rotation.force(iteration)
        elif not paused and self.rotation.due(iteration, [w.size for w in writers]):
            index = self.rotation.segment
        if index:
            for w in writers:
                w.rotate(index)
                w.prune(index + 1 - self.rotation.keep_segments)
        event = None
        if paused != self.paused:
            event = '[baseline] detailed logging ' + ('paused: low disk space' if paused else 'resumed')
            print(event, flush=True)
        self.paused = paused
        return event

    def record(self, identity, action, diagnostic):
        if self.paused:
            return
        if self.args.log_detail_level == 'compact' and action and 'physical_action' in action:
            action = compact_action_trace(action)
        self.action._write({'record_type': 'action', 'run_id': self.action.run_id, **identity, 'environment_action': action})
        if self.env:
            if diagnostic and self.args.log_detail_level == 'compact' and 'outer_action' in diagnostic:
                diagnostic = compact_env_trace(diagnostic)
            self.env._write({'record_type': 'env_diagnostic', 'run_id': self.action.run_id, **identity, 'diagnostic': diagnostic})

    def flush(self):
        self.action._file.flush()
        if self.env:
            self.env._file.flush()

    def close(self):
        self.action.close()
        if self.env:
            self.env.close()
