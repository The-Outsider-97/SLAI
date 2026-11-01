"""Checkpoint compatibility and release-baseline regressions for SLAILM."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from train_slailm import (
    Example,
    _data_batch_loss,
    audit_data,
    checkpoint_manifest,
    compare_architectures,
    main,
    selection_objective,
)


class TrainingRecoveryTests(unittest.TestCase):
    def test_dialogue_selection_does_not_mix_contrastive_loss(self) -> None:
        metrics = {
            "aggregate_objective": 6.2,
            "tasks": {"dialogue": {"objective": 2.1}, "embedding": {"objective": 0.3}},
        }
        self.assertEqual(selection_objective(metrics, "dialogue"), 2.1)
        self.assertEqual(selection_objective(metrics, "embedding"), 0.3)
        with self.assertRaisesRegex(ValueError, "no examples"):
            selection_objective(metrics, "translation")

    def test_data_audit_retains_all_seven_lantra_tasks(self) -> None:
        files = sorted(Path("data/processed/lantra/supervised").glob("*.jsonl"))
        audit = audit_data(files, require_all_tasks=True)
        self.assertTrue(audit["fingerprint"])
        self.assertGreater(audit["records"], 0)
        self.assertEqual(len({key.split("/")[0] for key in audit["counts"]}), 7)

    def test_checkpoint_manifest_and_shape_comparison(self) -> None:
        import torch  # type: ignore

        with tempfile.TemporaryDirectory() as directory:
            small = Path(directory) / "small.pt"
            large = Path(directory) / "large.pt"
            torch.save({"state_dict": {"a": torch.zeros(2, 2)},
                        "base_config": {"d_model": 2, "nhead": 1},
                        "metadata": {"model_id": "baseline"}}, small)
            torch.save({"state_dict": {"a": torch.zeros(4, 2)},
                        "base_config": {"d_model": 4, "nhead": 1},
                        "metadata": {"model_id": "candidate"}}, large)
            baseline = checkpoint_manifest(torch, small)
            candidate = checkpoint_manifest(torch, large)
            report = compare_architectures(baseline, candidate)
            self.assertEqual(len(baseline["sha256"]), 64)
            self.assertEqual(baseline["parameter_elements"], 4)
            self.assertEqual(candidate["parameter_elements"], 8)
            self.assertEqual(report["exact_shape_coverage"], 0)
            self.assertIn("d_model", report["architecture_differences"])

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--inspect-checkpoints", str(small), str(large)]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(len(result["checkpoints"]), 2)

    def test_benchmark_refuses_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--benchmark-checkpoints", str(missing)]), 1)

    def test_language_forward_uses_consistent_boolean_causal_mask(self) -> None:
        import torch  # type: ignore
        from src.agents.language.modules.language_transformer import LanguageTransformer

        model = LanguageTransformer(
            src_vocab_size=64, tgt_vocab_size=64,
            d_model=32, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=64,
            max_len=32, max_position_embeddings=32,
            dropout=0.0, record_history=False,
            enable_memory_integration=False,
            tie_embeddings=False, tie_output_projection=False,
        )
        source = torch.tensor([[4, 5, 2, 0], [4, 7, 8, 2]])
        target = torch.tensor([[1, 12, 2], [1, 13, 2]])
        output = model.language_forward(
            source, target,
            src_key_padding_mask=source.eq(0),
            memory_key_padding_mask=source.eq(0),
            return_dict=True,
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 3, 64))

    def test_sequence_and_contrastive_training_losses_backpropagate(self) -> None:
        import torch  # type: ignore
        from src.agents.language.modules.language_transformer import LanguageTransformer

        class TinyTokenizer:
            pad_token_id = 0

            def encode(self, value: str, *, max_length: int, **_: object) -> dict[str, list[int]]:
                tokens = [1] + [4 + (ord(char) % 60) for char in value][:max_length - 2] + [2]
                return {"input_ids": tokens}

        model = LanguageTransformer(
            src_vocab_size=64, tgt_vocab_size=64,
            d_model=32, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=64,
            max_len=32, max_position_embeddings=32,
            dropout=0.0, record_history=False,
            enable_memory_integration=False,
            tie_embeddings=False, tie_output_projection=False,
        )
        examples = [
            Example("dialogue", "train", "user: hello", "Hello!", "", "g1", "r1"),
            Example("embedding", "train", "sky", "blue sky", "black dog", "g2", "r2"),
        ]
        loss = _data_batch_loss(torch, model, TinyTokenizer(), examples,
                                src_limit=32, tgt_limit=24,
                                device=torch.device("cpu"), contrastive_temperature=0.1)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
