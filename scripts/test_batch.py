"""Exercise dependencies, successful resume and invalidation after config changes."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

class BatchTests(unittest.TestCase):
    def test_dependency_resume_and_config_change(self):
        file=Path(__file__).with_name('run_experiments.py')
        spec=importlib.util.spec_from_file_location('batch_under_test',file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'data').mkdir();(root/'configs').mkdir()
            (root/'data/manifest.json').write_text('{}')
            config=root/'configs/base.yaml';config.write_text('seed: 42\n')
            manifest={'suites':{'test':[
                {'name':'train','args':['train','--config','configs/base.yaml']},
                {'name':'eval','checkpoint_from':'train','args':['evaluate','--config','configs/base.yaml']}]}}
            (root/'configs/experiments.yaml').write_text(yaml.safe_dump(manifest))
            module.ROOT=root
            commands=[]
            def launch(command,**kwargs):
                commands.append(command)
                output=Path(command[command.index('--log-root')+1]);output.mkdir(parents=True)
                if command[2]=='train':
                    (output/'checkpoints').mkdir();(output/'checkpoints/checkpoint_best.pt').write_bytes(b'model')
                else:
                    checkpoint=Path(command[command.index('--checkpoint')+1])
                    self.assertEqual(checkpoint.read_bytes(),b'model')
                return SimpleNamespace(returncode=0)
            with patch.object(sys,'argv',['run_experiments.py','--manifest','configs/experiments.yaml','--suite','test','--resume',str(root/'batch')]), \
                 patch.object(module.subprocess,'run',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                module.main();self.assertEqual(len(commands),2)
                module.main();self.assertEqual(len(commands),2)
                config.write_text('seed: 142\n')
                module.main();self.assertEqual(len(commands),4)

if __name__=='__main__': unittest.main()
