import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

import models
from data import get_loaders
from runner import get_device, get_memory_usage_mb
from train import get_run_list, get_task2_model_name, load_config, make_result_row


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_config_has_required_parts(self):
        self.assertIn("DATASETS", self.config)
        self.assertIn("MODELS", self.config)
        self.assertIn("task2", self.config)
        self.assertIn("task3", self.config)

    def test_run_lists_for_task1_and_task2(self):
        task1_single = get_run_list(self.config, "task1", run_all=False)
        task1_all = get_run_list(self.config, "task1", run_all=True)
        task2_all = get_run_list(self.config, "task2", run_all=True)

        self.assertEqual(task1_single, [(self.config["DATA"], self.config["MODEL"], None)])
        self.assertIn(("cells", "AlexNet", None), task1_all)
        self.assertIn(("orgs", "ResNet18", None), task1_all)
        self.assertIn(("cells", "AlexNet", "Baseline"), task2_all)
        self.assertIn(("cells", "AlexNet", "Light"), task2_all)
        self.assertEqual(get_task2_model_name("AlexNet", "Light"), "Light_AlexNet")

    def test_models_return_correct_shape(self):
        dataset_config = self.config["DATASETS"]["cells"]
        x = torch.randn(2, dataset_config["channels"], 64, 64)

        for model_name in ["AlexNet", "VGG16", "ResNet18", "Light_AlexNet"]:
            model_class = getattr(models, model_name)
            model = model_class(
                in_channels=dataset_config["channels"],
                num_classes=dataset_config["num_classes"],
            )
            model.eval()
            with torch.no_grad():
                output = model(x)

            self.assertEqual(output.shape, (2, dataset_config["num_classes"]))

    def test_data_loader_can_read_pt_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            torch.save(
                {
                    "train_images": torch.randn(10, 3, 64, 64),
                    "train_labels": torch.randint(0, 8, (10,)),
                    "test_images": torch.randn(4, 3, 64, 64),
                    "test_labels": torch.randint(0, 8, (4,)),
                },
                data_path / "cells.pt",
            )

            train_loader, val_loader, test_loader = get_loaders(
                "cells",
                data_path,
                batch_size=2,
                val_split=0.2,
                seed=42,
            )

            self.assertEqual(len(train_loader.dataset), 8)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(len(test_loader.dataset), 4)

            images, labels = next(iter(train_loader))
            self.assertEqual(images.shape, (2, 3, 64, 64))
            self.assertEqual(labels.shape[0], 2)

    def test_result_row_has_green_metrics(self):
        metrics = {
            "num_params": 100,
            "total_params": 120,
            "training_time": 1.5,
            "peak_train_memory": None,
            "train_memory_type": "not_available_on_cpu",
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "macro_f1": 0.65,
            "inference_time": 0.2,
            "peak_inference_memory": None,
            "inference_memory_type": "not_available_on_cpu",
            "inference_latency_per_sample": 0.01,
        }
        row = make_result_row("cells", "Light_AlexNet", "Light", self.config, metrics)

        self.assertEqual(row["mode"], "Light")
        self.assertEqual(row["accuracy"], 80.0)
        self.assertIn("training_time_seconds", row)
        self.assertIn("inference_latency_per_sample", row)
        self.assertIn("training_memory_type", row)

    def test_device_and_cpu_memory_fallback(self):
        device = get_device()
        memory_mb, memory_type = get_memory_usage_mb(torch.device("cpu"))

        self.assertIn(device.type, {"cuda", "mps", "cpu"})
        self.assertIsNone(memory_mb)
        self.assertEqual(memory_type, "not_available_on_cpu")


if __name__ == "__main__":
    unittest.main()
