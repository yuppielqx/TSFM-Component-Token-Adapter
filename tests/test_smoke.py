"""Offline checks for the retained training, checkpoint, and forecasting paths."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import T5Config

from data_provider.data_factory import data_provider
from models.chronos_bolt.ChronosBoltPrototype import ChronosBoltWithRQVAEPrototypes
from models.chronos_bolt.ComponentTemporalTokenAdapter import ComponentTemporalTokenAdapter
from models.chronos_bolt.utils import left_pad_and_stack_1D
from utils.decomposition import decompose_series


ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    config = T5Config(
        d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=1,
        num_decoder_layers=1, dropout_rate=0.0, decoder_start_token_id=0,
    )
    config.chronos_config = dict(
        context_length=512, prediction_length=64, input_patch_size=16,
        input_patch_stride=16, quantiles=[.1, .5, .9], use_reg_token=True,
    )
    return config


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_help_without_optional_training_packages(self):
        for name in (
            "train_component_prototypes.py", "train_component_token_adapter.py",
            "eval_component_token_adapter_rollout.py", "scripts/download_backbone.py",
        ):
            with self.subTest(entrypoint=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / name), "--help"], cwd=ROOT,
                    capture_output=True, text=True, timeout=45,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
        self.assertNotIn("autogluon", sys.modules)
        self.assertNotIn("gluonts", sys.modules)

    def test_input_decomposition_and_padding(self):
        values = torch.arange(1024, dtype=torch.float32).reshape(2, 512) / 100
        components = decompose_series(values, period=24)
        torch.testing.assert_close(
            components["trend"] + components["seasonal"] + components["residual"], values,
        )
        padded = left_pad_and_stack_1D([torch.tensor([1.]), torch.tensor([2., 3.])])
        self.assertEqual(tuple(padded.shape), (2, 2))
        self.assertTrue(torch.isnan(padded[0, 0]))
        self.assertEqual(padded[0, 1].item(), 1.)

    def test_dataset_train_only_scaling_and_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            length = 14400
            trend = np.arange(length, dtype=float)
            frame = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=length, freq="h")})
            for index, name in enumerate(("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")):
                frame[name] = trend * (index + 1) + index
            frame.to_csv(Path(directory) / "ETTh2.csv", index=False)
            common = dict(
                model_id="offline", root_path=directory, data_path="ETTh2.csv",
                seq_len=512, pred_len=96, label_len=48, features="M", target="OT",
                embed="timeF", freq="h", percent=100, max_len=-1, train_all=False,
                return_feature_id=False, batch_size=4, num_workers=0,
            )
            for kind, train_size, test_size in (("ett_h", 8640, 2880), ("custom", 10080, 2880)):
                args = SimpleNamespace(data=kind, **common)
                dataset, loader = data_provider(args, "test")
                self.assertAlmostEqual(dataset.scaler.mean_[0], (train_size - 1)/2)
                self.assertEqual(len(dataset), (test_size - 96 + 1) * 7)
                self.assertEqual(tuple(next(iter(loader))[0].shape), (4, 512, 1))
                # The non-timeF branch must also support modern pandas.
                args.embed = "fixed"
                _, loader = data_provider(args, "val")
                self.assertEqual(tuple(next(iter(loader))[0].shape), (4, 512, 1))

    def test_stage_one_backward_keeps_backbone_frozen(self):
        torch.manual_seed(2021)
        model = ChronosBoltWithRQVAEPrototypes(tiny_config(), n_proto=8, n_rq_stages=2)
        output = model(torch.randn(2, 512))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.residual_quantizer.codebooks[0].weight.grad)
        self.assertIsNotNone(model.reconstruction_heads[0].weight.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))

    def test_launcher_selects_ett_protocol_and_requested_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_python = directory / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\nfrom pathlib import Path\n"
                "with open(os.environ['AUDIT_COMMANDS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "if sys.argv[1]=='train_component_token_adapter.py':\n"
                " p=Path(sys.argv[sys.argv.index('--checkpoint_dir')+1]);p.mkdir(parents=True,exist_ok=True);(p/'adapter_checkpoint.pt').touch()\n"
            )
            fake_python.chmod(0o755)
            log = directory / "commands.jsonl"
            result = subprocess.run(
                ["bash", str(ROOT / "script/run_chronos_bolt.sh")], cwd=ROOT,
                env={**os.environ, "PYTHON_BIN": str(fake_python), "DATASET": "ETTh2",
                     "HORIZONS": "96 192", "CHECKPOINT_ROOT": str(directory / "weights"),
                     "RESULTS_ROOT": str(directory / "results"), "AUDIT_COMMANDS": str(log)},
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertEqual(command[command.index("--data") + 1], "ett_h")
                self.assertEqual(command[command.index("--components") + 1], "trend seasonal")
            self.assertEqual(commands[0][commands[0].index("--pred_len") + 1], "64")
            self.assertEqual(commands[1][commands[1].index("--pred_len") + 1], "192")
            self.assertEqual(commands[1][commands[1].index("--horizons") + 1], "96 192")

    def test_adapter_checkpoint_and_two_round_rollout(self):
        torch.manual_seed(2021)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for component in ("trend", "seasonal"):
                path = Path(directory) / f"{component}.pt"
                torch.save({f"codebook_{level}": torch.randn(8, 32) for level in range(2)}, path)
                paths[component] = str(path)
            model = ComponentTemporalTokenAdapter(
                tiny_config(), paths, components=("trend", "seasonal"),
                n_proto=8, n_rq_stages=2, token_dropout=0.,
            )
            context = torch.randn(2, 512)
            output = model(context, target=torch.randn(2, 64))
            output.loss.backward()
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in model.component_quantizers.parameters()))
            self.assertIsNotNone(model.fusion_gate.weight.grad)
            adapter_path = Path(directory) / "adapter.pt"
            torch.save(model.adapter_state_dict(), adapter_path)
            state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            self.assertFalse(any(key.startswith("encoder.") for key in state))
            model.load_adapter_state_dict(state)
            model.eval()
            evaluator = importlib.import_module("eval_component_token_adapter_rollout")
            with patch.object(model, "_component_token_condition", wraps=model._component_token_condition) as calls:
                prediction = evaluator.rolling_forecast(model, context, 96)
            self.assertEqual(tuple(prediction.shape), (2, 96))
            self.assertEqual(calls.call_count, 2)
            with torch.inference_mode():
                native = model(context=context).quantile_preds[:, 1]
            torch.testing.assert_close(prediction[:, :64], native, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
