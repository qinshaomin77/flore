"""Regression checks for the portable FLORE-TS scenario workflow."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import train
import evaluate_windowed as window
from config import get_config

class FloreTSTests(unittest.TestCase):
    def test_configuration_contract(self):
        cfg = get_config(str(ROOT / "configs/grid36/demand/flore_ts_scenarios.yaml"))
        self.assertEqual(cfg.network.lane_input_dim, 4)
        self.assertFalse(cfg.dual_head.enabled)
        for key in ("enabled_reward", "use_truck_count_state", "use_nox_risk_state", "log_lane_risk"):
            self.assertFalse(getattr(cfg.emission_risk, key))
        self.assertEqual(cfg.env.episode_duration, 4200)
        self.assertEqual(cfg.ddqn.total_episodes, 160)
        self.assertEqual(cfg.ddqn.epsilon_decay_steps, 50400)

    def test_shared_routes_and_reproducible_shuffle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "shared.rou.xml").write_text("<routes/>")
            for i in range(5):
                (root / f"episode_{i}.sumocfg").write_text('<configuration><input><route-files value="shared.rou.xml"/></input></configuration>')
            kw = dict(rounds=3, low_exploration_rounds=1, global_seed=42, steps_per_episode=420, shuffle=True)
            first, decay = train.build_scenario_training_plan(folder, **kw)
            second, _ = train.build_scenario_training_plan(folder, **kw)
            self.assertEqual(first, second)
            self.assertEqual(decay, 4200)
            for start in (0, 5, 10):
                self.assertEqual(len({x["scenario"] for x in first[start:start+5]}), 5)

    def test_pure_traffic_training_needs_no_threshold_file(self):
        cfg = get_config(str(ROOT / "configs/grid36/demand/flore_ts_scenarios.yaml"))
        with patch.object(sys, "argv", ["train", "--thresholds-json", "missing_for_test.json"]):
            args = train.parse_args()
        cfg = train.apply_overrides(cfg, args)
        self.assertFalse(cfg.emission_risk.enabled_reward)

    def test_evaluation_diagnostics_preserve_policy_contract(self):
        with patch.object(sys, "argv", ["window", "--controller", "rl", "--case-name", "FLORE-TS", "--config", str(ROOT / "configs/grid36/demand/flore_ts_scenarios.yaml"), "--save-risk-service-log"]):
            args = window.parse_args()
        with tempfile.TemporaryDirectory() as folder:
            cfg = window.prepare_config(args, output_root=Path(folder))
        self.assertTrue(cfg.emission_risk.log_lane_risk)
        self.assertEqual(cfg.network.lane_input_dim, 4)
        self.assertFalse(cfg.emission_risk.enabled_reward)
        self.assertFalse(cfg.emission_risk.use_nox_risk_state)
        self.assertFalse(cfg.emission_risk.use_truck_count_state)

    def test_manifest_strategy_aliases(self):
        for alias in ("FLORE-TS_D20", "m0_traffic_D20"):
            self.assertEqual(window._manifest_policy(alias), "m0_traffic")
        self.assertEqual(window._manifest_policy("FLORE-TO_D40"), "traffic_only")
        self.assertEqual(window._manifest_policy("FLORE_D60"), "multi_objective")

if __name__ == "__main__":
    unittest.main()
