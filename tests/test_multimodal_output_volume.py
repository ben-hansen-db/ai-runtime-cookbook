"""Exercise output preflight without importing Ray, vLLM, or downloading models."""

import ast
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / (
    "multimodal/ray-data-batch-inference-qwen3-vl-8b-vllm/batch_inference.py"
)
tree = ast.parse(SOURCE.read_text())
# Load only the preflight and its caller so these tests run with the standard
# library even when the recipe's GPU dependencies are unavailable locally.
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
    "check_output_volume", "prepare_output_directory", "main",
}]
namespace = {"os": os, "logging": logging, "tempfile": tempfile}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)


class OutputVolumeTest(unittest.TestCase):
    def test_missing_volume_is_error_with_recovery_steps(self):
        with patch("os.path.isdir", return_value=False), self.assertLogs(level="ERROR") as logs:
            with self.assertRaises(RuntimeError) as error:
                namespace["check_output_volume"]("/Volumes/main/default/air_examples/results")
        message = str(error.exception)
        for text in ("main.default.air_examples", "CREATE VOLUME", "grant access", "OUTPUT_PATH", "rerun"):
            self.assertIn(text, message)
        self.assertIn(message, logs.output[0])

    def test_missing_volume_stops_before_directory_creation(self):
        with patch("os.path.isdir", return_value=False), patch("os.makedirs") as mkdir:
            with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError):
                namespace["prepare_output_directory"](
                    "/Volumes/main/default/missing/results", "/Volumes/main/default/missing/results/run"
                )
        mkdir.assert_not_called()

    def test_writable_directory_succeeds_without_leaving_probe_file(self):
        with tempfile.TemporaryDirectory() as root:
            run_path = os.path.join(root, "results", "run")
            namespace["prepare_output_directory"](root, run_path)
            self.assertTrue(os.path.isdir(run_path))
            self.assertEqual(os.listdir(run_path), [])
            # Reusing an existing writable run directory also succeeds.
            namespace["prepare_output_directory"](root, run_path)

    def test_directory_creation_failure_retains_cause_and_guidance(self):
        cause = PermissionError("Permission denied")
        with patch("os.makedirs", side_effect=cause):
            with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError) as error:
                namespace["prepare_output_directory"]("/tmp/results", "/tmp/results/run")
        self.assertIs(error.exception.__cause__, cause)
        self.assertIn("Cannot write", str(error.exception))
        self.assertIn("volume permissions", str(error.exception))
        self.assertIn("/tmp/results/run", str(error.exception))
        self.assertIn("workload.yaml", str(error.exception))

    def test_existing_directory_without_write_access_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("tempfile.TemporaryFile", side_effect=PermissionError("Permission denied")):
                with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "Cannot write"):
                    namespace["prepare_output_directory"](root, root)

    def test_missing_volume_fails_before_ray_initialization(self):
        ray = Mock()
        with patch.dict(namespace, {
            "OUTPUT_PATH": "/Volumes/main/default/missing/results",
            "RUN_OUTPUT_PATH": "/Volumes/main/default/missing/results/run",
            "ray": ray,
        }), patch("os.path.isdir", return_value=False):
            with self.assertLogs(level="ERROR"), self.assertRaises(RuntimeError):
                namespace["main"]()
        ray.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
