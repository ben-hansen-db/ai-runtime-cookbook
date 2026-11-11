"""CPU-only regression checks for the requested vLLM actor pool.

Run with: python3 -m unittest discover -s <recipe-directory> -p 'test_*.py'
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch


def load_recipe():
    """Import without installing Ray, pyarrow, or downloading model dependencies."""
    modules = {name: ModuleType(name) for name in (
        "pyarrow", "pyarrow.parquet", "ray", "ray.data", "ray.data.llm", "huggingface_hub"
    )}
    modules["pyarrow"].parquet = modules["pyarrow.parquet"]
    modules["ray.data.llm"].build_processor = MagicMock()
    modules["ray.data.llm"].vLLMEngineProcessorConfig = MagicMock()
    modules["huggingface_hub"].hf_hub_download = MagicMock()
    spec = importlib.util.spec_from_file_location(
        "multimodal_batch_inference", Path(__file__).with_name("batch_inference.py")
    )
    recipe = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(recipe)
    return recipe


class StopBeforeInference(Exception):
    pass


class ReplicaPoolTest(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe()
        self.recipe.ray.init = MagicMock()
        self.recipe.ray.cluster_resources = MagicMock()
        self.recipe.load_images = MagicMock()
        self.recipe.check_output_volume = MagicMock()
        self.recipe.build_processor.side_effect = StopBeforeInference

    def test_every_allocated_gpu_gets_a_replica(self):
        # In Ray 2.56.1, an integer is a 1..n autoscaling pool. Equal bounds
        # ensure the eight-GPU workload starts eight replicas, even for short input.
        for gpu_count in (1, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                self.recipe.ray.cluster_resources.return_value = {"GPU": gpu_count}
                with patch.object(self.recipe.os, "makedirs"), patch("builtins.print"):
                    with self.assertRaises(StopBeforeInference):
                        self.recipe.main()
                settings = self.recipe.vLLMEngineProcessorConfig.call_args[1]
                self.assertEqual(settings["concurrency"], (gpu_count, gpu_count))
                self.assertEqual(settings["engine_kwargs"]["tensor_parallel_size"], 1)

    def test_no_gpu_fails_before_data_or_model_loading(self):
        for resources in ({}, {"GPU": 0}):
            with self.subTest(resources=resources):
                self.recipe.ray.cluster_resources.return_value = resources
                with patch.object(self.recipe.os, "makedirs"):
                    with self.assertRaisesRegex(RuntimeError, "No GPUs found"):
                        self.recipe.main()
        self.recipe.load_images.assert_not_called()
        self.recipe.vLLMEngineProcessorConfig.assert_not_called()
        self.recipe.build_processor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
