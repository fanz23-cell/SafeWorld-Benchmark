# Goal2 DreamerV3 Comparison Analysis

Run config:

```bash
python run_goal2_benchmark.py --n 30 --horizon 150 --output results_goal2_safeworld.json
python run_goal2_action_baselines.py --n 30 --horizon 150 --candidates 8 --plan-horizon 3 --output results_goal2_action_baselines.json
```

## Side-by-side results

SafeWorld reports verification outcomes. SafeDreamer-MPC and Shielding-MPC are action-generating DreamerV3 baselines, not AP post-processing. Their columns are empirical compliance rates, not formal warrants.

| L | Spec | SafeWorld | rho_net | SafeDreamer-MPC | Shielding-MPC |
|---:|---|---|---:|---:|---:|
| 1 | stl_hazard_avoidance | WARRANT | +0.181 | 100.0% | 100.0% |
| 1 | ltl_hazard_avoidance | STL_MARGIN | +0.002 | 96.7% | 100.0% |
| 1 | stl_speed_limit | WARRANT | +0.526 | 100.0% | 100.0% |
| 1 | ltl_speed_limit | STL_MARGIN | +0.526 | 100.0% | 100.0% |
| 2 | stl_safe_goal_reach | VIOLATION | -0.721 | 0.0% | 0.0% |
| 2 | ltl_safe_goal | VIOLATION | -0.363 | 0.0% | 0.0% |
| 2 | ltl_safe_slow_goal | VIOLATION | -0.363 | 0.0% | 0.0% |
| 3 | stl_sequential_zones | N/A | -- | N/A | N/A |
| 3 | ltl_sequential_goals | N/A | -- | N/A | N/A |
| 3 | ltl_three_stage | N/A | -- | N/A | N/A |
| 4 | stl_obstacle_response | VIOLATION | -1.560 | N/A | 0.0% |
| 4 | ltl_hazard_response | STL_MARGIN | +0.045 | N/A | 100.0% |
| 5 | stl_bounded_patrol | N/A | -- | N/A | N/A |
| 5 | ltl_patrol | N/A | -- | N/A | N/A |
| 5 | stl_safe_dual_patrol | N/A | -- | N/A | N/A |
| 5 | ltl_dual_patrol | N/A | -- | N/A | N/A |
| 6 | ltl_safe_reactive_goal | VIOLATION | -0.638 | 0.0% | N/A |
| 6 | ltl_safe_patrol | N/A | -- | N/A | N/A |
| 7 | ltl_human_caution | STL_MARGIN | -0.024 | N/A | N/A |
| 7 | ltl_conditional_speed | N/A | -- | N/A | N/A |
| 7 | ltl_conditional_proximity | STL_MARGIN | +0.141 | N/A | N/A |
| 8 | ltl_full_mission | N/A | -- | N/A | N/A |
| 8 | stl_full_mission | N/A | -- | N/A | N/A |

## What changed after fixing the baseline issue

The earlier `run_goal2_baselines.py` was only an AP-trajectory compliance runner. The new `run_goal2_action_baselines.py` makes both baselines generate actions in the learned model:

- `safedreamer_mpc`: random-shooting Lagrangian MPC using the trained RSSM, reward head, cost head, goal-distance head, and hazard/speed margins.
- `shielding_mpc`: one-step model-predictive shield around oracle proposed actions; unsafe proposed actions are replaced by candidate fallback actions predicted to satisfy simple safety constraints.

This fixes the specific "AP post-processing" weakness. It still does not make the results official SafeDreamer training or formal shield synthesis, because the checkpoint has no actor/critic parameters and the current Python environment cannot instantiate Safety-Gymnasium.

## High-level takeaway

The current DreamerV3 integration gives a better pilot comparison after switching to action-level baselines, but it is still not strong enough as the final main experimental result for the paper.

What it supports:

- SafeWorld can produce formal-style verdicts on the APs the trained world model actually predicts.
- On L1, SafeWorld gives two WARRANT results and two positive-margin results, while the action baselines also satisfy the easy safety tasks.
- On L4 `ltl_hazard_response`, SafeWorld and Shielding both show positive behavior; SafeWorld additionally reports calibrated margins.
- The comparison clearly demonstrates expressiveness limitations: SafeDreamer is only meaningful on L1-L2 here, Shielding only on L1-L4, while later levels require temporal/semantic APs those baselines or the current model cannot handle.

What prevents this from being final-paper main evidence:

- Many benchmark levels are N/A because the trained DreamerV3 model has no `zone_a`, `zone_b`, `zone_c`, or `carrying` heads. This makes L3, L5, L8, and part of L6 impossible to evaluate fairly.
- The new SafeDreamer/Shielding baselines generate actions, but they are still local model-predictive approximations, not a retrained SafeDreamer actor and not an exact shield synthesized over a formal abstraction.
- L2 and L6 violations are dominated by the strict `goal_dist < -0.2` threshold and decoder/calibration mismatch, so the current table does not yet show SafeWorld outperforming baselines on reachability.
- L7 depends on synthetic `human_distance` and is marked PARTIAL, so it should not be treated as fully grounded evidence.
- The runs use 30 rollouts and one seed. A final paper table should use multiple seeds and confidence intervals.

Recommended use in the paper:

- Use this as a pilot/diagnostic table or appendix result.
- Do not present it as the final headline experimental comparison.
- For the final main result, either retrain the world model with missing AP heads and run multi-seed evaluation, or narrow the claim to the subset of supported APs/specs and explicitly label baselines as empirical expressiveness baselines.
