# N2 run frozen training trend and throughput analysis

Boundary: update 5804, 464320 transitions, 11608 episodes. Last complete UPDATE: 2026-10-19 09:07:01. Fixed seed 42 / scene offset 0 / fixed shared exogenous scene.

The training learns in aggregate and recent completion still improves, but absolute success remains low. Q decreases over the long run and rises slightly between the two latest adjacent 500-update windows. This is an on-policy training curve conclusion, without a frozen-policy or independent-scene evaluation.

| Window | Completion % | Optimal % | Reward | Base reward | Q | Steps/s wall |
|---|---:|---:|---:|---:|---:|---:|
| first100 (1–100) | 0.36250 | 0.67500 | -2.041899 | -1.541100 | 0.500799 | 8.979 |
| first500 (1–500) | 1.27000 | 1.54000 | -1.786547 | -1.333234 | 0.453313 | 8.780 |
| first1000 (1–1000) | 2.07000 | 2.22250 | -1.595042 | -1.172052 | 0.422991 | 8.604 |
| previous1000 (3805–4804) | 5.57875 | 5.60875 | -1.185512 | -0.899033 | 0.286479 | 8.126 |
| last1000 (4805–5804) | 6.68500 | 6.70000 | -1.147550 | -0.876760 | 0.270790 | 7.969 |
| previous500 (4805–5304) | 6.28750 | 6.31000 | -1.155570 | -0.886307 | 0.269264 | 8.109 |
| last500 (5305–5804) | 7.08250 | 7.09000 | -1.139529 | -0.867213 | 0.272316 | 7.834 |
| last100 (5705–5804) | 7.40000 | 7.42500 | -1.131806 | -0.859940 | 0.271867 | 6.963 |
| all (1–5804) | 4.57831 | 4.61406 | -1.276888 | -0.956475 | 0.320413 | 8.217 |

## Non-overlapping 500-update windows

| Updates | Completion % | Optimal % | Reward | Base reward | Q | Wall s/update |
|---|---:|---:|---:|---:|---:|---:|
| 1–500 | 1.27000 | 1.54000 | -1.786547 | -1.333234 | 0.453313 | 9.112 |
| 501–1000 | 2.87000 | 2.90500 | -1.403537 | -1.010869 | 0.392668 | 9.484 |
| 1001–1500 | 3.18500 | 3.18750 | -1.312987 | -0.959711 | 0.353276 | 9.616 |
| 1501–2000 | 3.52750 | 3.53000 | -1.270616 | -0.946058 | 0.324558 | 9.718 |
| 2001–2500 | 4.33000 | 4.33250 | -1.241931 | -0.928285 | 0.313646 | 9.756 |
| 2501–3000 | 4.58250 | 4.58750 | -1.222284 | -0.921370 | 0.300915 | 9.784 |
| 3001–3500 | 5.31250 | 5.31750 | -1.196453 | -0.906909 | 0.289544 | 9.832 |
| 3501–4000 | 5.69250 | 5.70750 | -1.194102 | -0.898442 | 0.295661 | 9.794 |
| 4001–4500 | 5.76000 | 5.79500 | -1.179318 | -0.895682 | 0.283636 | 9.832 |
| 4501–5000 | 5.62000 | 5.64750 | -1.175601 | -0.898879 | 0.276723 | 9.882 |
| 5001–5500 | 6.45000 | 6.46000 | -1.150260 | -0.881113 | 0.269147 | 9.866 |
| 5501–5804 | 7.47533 | 7.48355 | -1.132364 | -0.858894 | 0.273470 | 10.418 |

## Timing (additive measurements and overlap)

| Window | Wall seconds | Collection wait % | PPO % | Detailed JSONL logging % | Unmeasured % |
|---|---:|---:|---:|---:|---:|
| all | 56505 | 93.94 | 2.70 | 1.58 | 1.77 |
| first500 | 4556 | 93.68 | 2.88 | 1.61 | 1.83 |
| previous500 | 4933 | 94.01 | 2.65 | 1.59 | 1.75 |
| last500 | 5106 | 94.02 | 2.70 | 1.57 | 1.72 |
| last100 | 1149 | 93.97 | 2.85 | 1.48 | 1.71 |

Worker max collection time overlaps collection wait, so its percentage must never be added. CSV/text/TensorBoard/checkpoint and broadcast overhead are in the unmeasured residual; detailed JSONL logging_seconds alone does not capture all I/O.

## Integrity and anomalies

- 5804 UPDATE / 11608 ENV rows match the frozen boundary. Checksums ok 5804/5804; policy versions and stale=1 follow the synchronous contract.
- Solver counts: {'optimal': 21424, 'infeasible': 442892, 'unknown': 4, 'error': 0, 'other': 0}. Optimal minus completion: 166 transitions.
- Reward = base_reward − Q maximum numeric discrepancy: 5.55e-16.
- Prefix warning/error/critical or checksum mismatch lines: 0; log events: 10. See trends.json for exact evidence.

## Scope limits

- Frozen CSV is authoritative; live log is streamed only through matching update prefix.
- All success rates are transition rates, not probability that an entire 40-slot episode finishes.
- Log timestamp resolution is one second; wall-time cost sums omit broadcast, CSV, text/TensorBoard, checkpoint and other overhead.
- worker_collection_seconds_max is an overlapping diagnostic, never added to collection wait/PPO/logging.
- Text diagnostics/policy arrays are rounded; CSV reward/Q are full precision.
- last500 and previous500 are adjacent fixed trailing windows; best500 is descriptive and selected retrospectively.
- No frozen-policy evaluation, new-seed validation, or controlled ablation was run. Training improvement does not establish deployment success or isolate causality.
