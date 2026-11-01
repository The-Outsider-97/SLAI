"""Model hub resolver contract tests: no actual network or model downloads."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.model_hub import ModelHubError, resolve_model_bundle, resolve_model_file


class ModelHubResolverTests(unittest.TestCase):
    def test_local_mode_never_imports_or_downloads(self):
        with patch.dict(os.environ, {"SLAI_MODEL_SOURCE": "local"}):
            self.assertEqual(
                resolve_model_file("local/checkpoint.pt", "language/lantra/model.pt"),
                Path("local/checkpoint.pt"),
            )
            self.assertEqual(
                resolve_model_bundle("local/ranker", "knowledge/relevance_ranker", ("manifest.json", "model.pt")),
                Path("local/ranker"),
            )

    def test_hub_file_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "model.pt"
            artifact.write_bytes(b"test")
            mock_hub = types.ModuleType("huggingface_hub")
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                return str(artifact)
            mock_hub.hf_hub_download = download
            with patch.dict(sys.modules, {"huggingface_hub": mock_hub}), patch.dict(
                os.environ, {"SLAI_MODEL_SOURCE": "hub", "SLAI_MODEL_REVISION": "abc123", "SLAI_MODEL_OFFLINE": "1"}
            ):
                self.assertEqual(resolve_model_file("missing.pt", "learning/dqn/StandardDQN.pt"), artifact)
            self.assertEqual(calls[0]["revision"], "abc123")
            self.assertTrue(calls[0]["local_files_only"])

    def test_prefer_hub_keeps_local_fallback(self):
        mock_hub = types.ModuleType("huggingface_hub")
        def missing(**kwargs):
            raise OSError("unavailable")
        mock_hub.hf_hub_download = missing
        with patch.dict(sys.modules, {"huggingface_hub": mock_hub}), patch.dict(
            os.environ, {"SLAI_MODEL_SOURCE": "prefer_hub"}
        ):
            self.assertEqual(resolve_model_file("cached.pt", "safety/cyber/mlp_http_risk.pt"), Path("cached.pt"))

    def test_required_hub_does_not_silently_fallback(self):
        mock_hub = types.ModuleType("huggingface_hub")
        def missing(**kwargs):
            raise OSError("offline")
        mock_hub.hf_hub_download = missing
        with patch.dict(sys.modules, {"huggingface_hub": mock_hub}), patch.dict(
            os.environ, {"SLAI_MODEL_SOURCE": "hub"}
        ):
            with self.assertRaises(ModelHubError):
                resolve_model_file("cached.pt", "learning/dqn/StandardDQN.pt")

    def test_bundle_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "knowledge" / "relevance_ranker"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{}", encoding="utf-8")
            (directory / "model.pt").write_bytes(b"test")
            mock_hub = types.ModuleType("huggingface_hub")
            mock_hub.snapshot_download = lambda **kwargs: temp
            with patch.dict(sys.modules, {"huggingface_hub": mock_hub}), patch.dict(
                os.environ, {"SLAI_MODEL_SOURCE": "hub"}
            ):
                self.assertEqual(
                    resolve_model_bundle("unused", "knowledge/relevance_ranker", ("manifest.json", "model.pt")),
                    directory,
                )

    def test_rejects_traversal(self):
        with self.assertRaises(ValueError):
            resolve_model_file("unused", "../escape.pt")
        with self.assertRaises(ValueError):
            resolve_model_bundle("unused", "knowledge/relevance_ranker", ("../model.pt",))


if __name__ == "__main__":
    unittest.main()
