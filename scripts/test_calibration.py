"""Test calibration sample selection, per-demand separation and P80 rounding."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
class CalibrationTests(unittest.TestCase):
    def test_quantile_filters_and_demands_are_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for demand,multiplier in [('D20',1),('D40',10)]:
                path=root/demand/'physical/episode_0001/lane_step.csv'
                path.parent.mkdir(parents=True)
                with path.open('w',encoding='utf-8',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(['sim_time','lane_nox_mg'])
                    writer.writerows([(0,99999),(300,0),(300,-1),(300,'nan'),(300,'inf'),
                                      *[(310+i,multiplier*x) for i,x in enumerate([1,2,3,4,5])]])
            subprocess.run([sys.executable,str(ROOT/'scripts/calibrate_nox_max.py'),
                            '--run-dir',str(root),'--output-dir',str(root/'out')],check=True,capture_output=True)
            for demand,expected in [('D20',5),('D40',42)]:
                payload=json.loads((root/'out'/f'{demand}.json').read_text())
                self.assertEqual(payload['NOx_max'],expected)
                self.assertEqual(payload['calibration_meta']['sample_count'],5)

if __name__=='__main__': unittest.main()
