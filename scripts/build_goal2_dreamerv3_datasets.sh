#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Check Goal2 benchmark tasks"
python - <<'PY'
from data_generation.generate_goal2_master_dataset import GOAL2_TASK_IDS
from benchmark.task_registry import get_task_config
for task_id in GOAL2_TASK_IDS:
    task = get_task_config(task_id)
    print(task.task_id, task.env_id, task.level, task.grounding_status)
PY

echo "[2/6] Check external/dreamerv3"
test -d external/dreamerv3
python - <<'PY'
from pathlib import Path
print((Path("external/dreamerv3") / "dreamerv3" / "main.py").resolve())
PY

echo "[3/6] Generate Goal2 pilot master dataset"
python scripts/generate_goal2_pilot_dataset.py --allow-partial

echo "[4/6] Show pilot summary"
python - <<'PY'
import json
from pathlib import Path
path = Path("datasets/goal2_master/dataset_summary_master.json")
print(json.loads(path.read_text(encoding="utf-8"))["total_episodes"])
PY

echo "[5/6] Export mixed and success-only subsets"
python scripts/export_goal2_subsets.py

echo "[6/6] Export DreamerV3-compatible datasets"
python scripts/export_goal2_for_dreamerv3.py --source-root datasets/goal2_master --output-root datasets/goal2_master_dreamerv3
python scripts/export_goal2_for_dreamerv3.py --source-root datasets/goal2_mixed_70_20_10 --output-root datasets/goal2_mixed_70_20_10_dreamerv3
python scripts/export_goal2_for_dreamerv3.py --source-root datasets/goal2_success_only --output-root datasets/goal2_success_only_dreamerv3

echo "Goal2 DreamerV3 pilot build complete."
