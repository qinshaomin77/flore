"""Evaluate E2 movement Max-Pressure on Kunshan with three workers."""
from __future__ import annotations

import importlib.util
import multiprocessing
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = HERE

# Load the shared parallel runner under a stable module name while keeping the
# real-world directory first on sys.path. Its imports therefore resolve to this
# project's config/env/logger/network_parser/controller implementations. The
# stable name is also required when Windows spawn unpickles worker functions.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUNNER_MODULE_NAME = "_kunshan_maxpressure_parallel_runner"
runner_spec = importlib.util.spec_from_file_location(
    RUNNER_MODULE_NAME,
    GRID / "_maxpressure_runner.py",
)
if runner_spec is None or runner_spec.loader is None:
    raise ImportError("Unable to load the shared Max-Pressure evaluator")
runner = importlib.util.module_from_spec(runner_spec)
sys.modules[RUNNER_MODULE_NAME] = runner
runner_spec.loader.exec_module(runner)

runner.ROOT = HERE.parents[1]
runner.DEFAULT_WORKERS = 3
runner.DEFAULT_SAVE_PHYSICAL_LANE_STEP = True
runner.DEFAULT_OUTPUT_ROOT = (
    "results/evaluation/kunshan/"
    "kunshan_freight_enhanced_maxpressure"
)
runner.DEFAULTS = {
    "sumocfg": (
        "data/kunshan/"
        "kunshan_freight_enhanced.sumocfg"
    ),
    "net": "data/kunshan/kunshan.net.xml",
    "add": "data/kunshan/maxpressure_e2.add.xml",
    "parse_add": "data/kunshan/kunshan.add.xml",
    "groups": "data/kunshan/intersection_groups.json",
    "map": "data/kunshan/maxpressure_detector_map.json",
    "clamp_detector_geometry": True,
}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    runner.main()
