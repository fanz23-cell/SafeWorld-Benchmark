# SAFEWORLD Benchmark Tasks

This directory implements the benchmark task layer for SAFEWORLD-BENCH L1-L4. It is not the SAFEWORLD main algorithm, not LPPM, not CEGAR, not world-model training, and not baseline policy training.

The benchmark tasks keep the paper formulas unchanged and only instantiate them in the four Safety Gymnasium environments:

- `SafetyPointGoal1-v0`
- `SafetyPointGoal2-v0`
- `SafetyCarGoal1-v0`
- `SafetyPointButton1-v0`

## What Is Implemented

- Task configs for L1-L4
- Atomic proposition extractors
- Rollout runner with `action_source="random"`
- Paper-spec-specific boolean evaluators
- Single-task and batch-suite scripts
- JSON and CSV result saving
- Screenshot saving for key steps

## What Is Not Implemented

- SAFEWORLD latent monitor
- Transfer calibrator
- LPPM
- DreamerV3 or TD-MPC2 integration
- Baseline policy training
- Full general STL or LTL parser

## Grounding Notes

- `hazard` uses native hazard geometry from Safety Gymnasium.
- `goal` in goal environments uses native `task.goal_achieved`.
- `goal` in Button1 uses native `task.goal_achieved`, and the target-button distance is also logged.
- `fast` uses confirmed thresholds:
  - Point environments: `0.35`
  - Car environments: `0.30`
- `near_obs` currently grounds to nearest vase distance with:
  - `near_obs := distance(agent, nearest_vase_center) < 0.30`
  - This is explicit because `hazard` is already reserved for the hazard AP.
- `A/B/C` are explicit config-level geometric zones, not hidden in code.

## Placeholder Status

- `E4_L4_HumanCaution_Button` is placeholder-only.
- `near_human` is not paper-faithfully grounded in stock `SafetyPointButton1-v0`.
- The task config, AP interface, and evaluator interface exist, but the task is excluded from default batch runs.
- Results and summaries mark it as `placeholder`, `needs_user_confirmation`, and `not paper-faithfully grounded yet`.

## Task Status Categories

- `fully_runnable`: grounded and enabled in default suites
- `placeholder`: interface exists, but default batch running is disabled
- `needs_manual_review`: reserved for future partial-grounding cases

## Scripts

- Inspect native environment objects:

```bash
python scripts/inspect_env_objects.py
```

- Run one task:

```bash
python scripts/run_single_task.py --task_id E1_L1_HazardAvoid --seed 0
```

- Run one or more levels:

```bash
python scripts/run_level_suite.py --level 1
python scripts/run_level_suite.py --level 1 2 3 4
```

## Output Layout

Single-task artifacts are saved under:

```text
outputs/<task_id>/seed_<seed>/
```

This includes:

- `task_config_snapshot.json`
- `trace.json`
- `result.json`
- key frames such as `frame_000.png`, midpoint, final, and violation frame when relevant

Batch suites are saved under:

```text
outputs/suite_runs/levels_<...>/seed_<seed>/
```

This includes:

- `summary.csv`
- `summary.json`
- `suite_manifest.json`

## Future Integration

The runner interface is intentionally narrow:

```python
run_task(task_config, action_source="random", seed=0, render=False)
```

This is where future `policy` and `world_model` action sources can be attached without changing the task definitions or AP evaluator layer.
