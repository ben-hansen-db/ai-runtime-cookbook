"""Run with python -m unittest discover -s . -p 'test_*.py'; no GPU or downloads."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class HubHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = types.SimpleNamespace(status_code=status_code)


class ModelAccessTests(unittest.TestCase):
    def setUp(self):
        self.ray = types.ModuleType("ray")
        self.ray.init = Mock()
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.get_hf_file_metadata = Mock()
        self.hub.hf_hub_url = lambda repository, filename: f"https://huggingface.co/{repository}/resolve/main/{filename}"
        hub_utils = types.ModuleType("huggingface_hub.utils")
        hub_utils.HfHubHTTPError = HubHTTPError
        self.torch = types.ModuleType("torch")
        self.torch.load = Mock()
        self.original_load = self.torch.load
        self.torch.device = Mock(return_value="cuda")
        whisper = types.ModuleType("faster_whisper")
        whisper.BatchedInferencePipeline = Mock()
        whisper.WhisperModel = Mock()
        self.pipeline = Mock()
        pyannote = types.ModuleType("pyannote")
        pyannote_audio = types.ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = self.pipeline
        self.modules = patch.dict(sys.modules, {
            "ray": self.ray,
            "huggingface_hub": self.hub,
            "huggingface_hub.utils": hub_utils,
            "torch": self.torch,
            "faster_whisper": whisper,
            "pyannote": pyannote,
            "pyannote.audio": pyannote_audio,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            "transcribe_diarize", Path(__file__).with_name("transcribe_diarize.py")
        )
        self.recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.recipe)
        # This value never leaves a mock; no real credential is needed.
        self.recipe.HF_TOKEN = "test-token"

    def assert_recovery_message(self, error, repository):
        message = str(error)
        self.assertIn(repository, message)
        self.assertIn("https://huggingface.co/pyannote/speaker-diarization-3.1", message)
        self.assertIn("https://huggingface.co/pyannote/segmentation-3.0", message)
        self.assertIn("accept the terms", message)
        self.assertIn("read token", message)
        self.assertIn("secrets.HF_TOKEN", message)
        self.assertIn("rerun databricks air run", message)
        self.assertNotIn("test-token", message)

    def test_checks_both_repositories(self):
        self.recipe.check_model_access()
        self.assertEqual(self.hub.get_hf_file_metadata.call_args_list, [
            unittest.mock.call(self.hub.hf_hub_url(self.recipe.DIARIZATION_MODEL, "config.yaml"), token="test-token"),
            unittest.mock.call(self.hub.hf_hub_url(self.recipe.SEGMENTATION_MODEL, "config.yaml"), token="test-token"),
        ])

    def test_missing_token_fails_before_ray_or_downloads(self):
        self.recipe.HF_TOKEN = None
        with self.assertRaises(RuntimeError) as caught:
            self.recipe.main()
        self.assert_recovery_message(caught.exception, self.recipe.DIARIZATION_MODEL)
        self.hub.get_hf_file_metadata.assert_not_called()
        self.ray.init.assert_not_called()

    def test_access_failure_identifies_repository_before_ray_starts(self):
        for status in (401, 403, 404):
            for repository in (self.recipe.DIARIZATION_MODEL, self.recipe.SEGMENTATION_MODEL):
                with self.subTest(status=status, repository=repository):
                    error = HubHTTPError(status)
                    self.hub.get_hf_file_metadata.side_effect = (
                        [error] if repository == self.recipe.DIARIZATION_MODEL else ["config", error]
                    )
                    with self.assertRaises(RuntimeError) as caught:
                        self.recipe.main()
                    self.assert_recovery_message(caught.exception, repository)
                    self.assertIs(caught.exception.__cause__, error)
                    self.ray.init.assert_not_called()

    def test_server_and_local_errors_are_not_mislabeled(self):
        for error in (HubHTTPError(500), OSError("disk full")):
            with self.subTest(error=error):
                self.hub.get_hf_file_metadata.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    self.recipe.check_model_access()
                self.assertIs(caught.exception, error)

    def test_none_pipeline_has_actionable_error_and_restores_torch_load(self):
        self.pipeline.from_pretrained.return_value = None
        with self.assertRaises(RuntimeError) as caught:
            self.recipe.WhisperDiarizer()
        self.assert_recovery_message(caught.exception, self.recipe.DIARIZATION_MODEL)
        self.assertIs(self.torch.load, self.original_load)
        self.torch.device.assert_not_called()

    def test_actor_access_exception_has_recovery_and_restores_torch_load(self):
        self.pipeline.from_pretrained.side_effect = HubHTTPError(403)
        with self.assertRaises(RuntimeError) as caught:
            self.recipe.WhisperDiarizer()
        self.assert_recovery_message(caught.exception, self.recipe.SEGMENTATION_MODEL)
        self.assertIs(self.torch.load, self.original_load)

    def test_actor_unrelated_exception_is_preserved_and_restores_torch_load(self):
        error = RuntimeError("CUDA out of memory")
        self.pipeline.from_pretrained.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.recipe.WhisperDiarizer()
        self.assertIs(caught.exception, error)
        self.assertIs(self.torch.load, self.original_load)

    def test_success_moves_pipeline_to_cuda_after_restoring_torch_load(self):
        diarizer = self.pipeline.from_pretrained.return_value
        diarizer.to.side_effect = lambda device: self.assertIs(self.torch.load, self.original_load)
        self.recipe.WhisperDiarizer()
        self.pipeline.from_pretrained.assert_called_once_with(
            self.recipe.DIARIZATION_MODEL, use_auth_token="test-token"
        )
        diarizer.to.assert_called_once_with("cuda")
        self.assertIs(self.torch.load, self.original_load)


if __name__ == "__main__":
    unittest.main()
