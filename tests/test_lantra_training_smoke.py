"""Torch smoke tests for the batched LANTRA trainer and constrained decoder.

These need PyTorch (``py -m unittest tests.test_lantra_training_smoke -v``) and are skipped otherwise.
They use a tiny randomly initialised model and a fake whitespace tokenizer, so they run in seconds on CPU
and check the properties that matter for correctness:

* a padded batch gives the same per-example loss as each example alone (masks are right);
* batched decoding equals single decoding under source padding;
* ``no_repeat_ngram_size`` makes ``meta meta meta`` loops impossible;
* a few optimizer steps reduce the loss on a memorisable batch;
* pairwise (embedding / reranking) and generation-health paths run end to end.
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
import zlib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - environment without PyTorch
    torch = None  # type: ignore[assignment]


class FakeTokenizer:
    pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 3
    vocab = list(range(64))

    def encode(self, text, **kwargs):
        ids = [4 + zlib.crc32(word.encode()) % 50 for word in str(text).lower().split()]
        limit = kwargs.get("max_length") or 10**6
        return {"input_ids": ([self.bos_token_id, *ids, self.eos_token_id])[:limit]}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        out = []
        for value in values:
            if value == self.eos_token_id:
                break
            if skip_special_tokens and value in special:
                continue
            out.append(f"t{value}")
        return " ".join(out)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BatchedTrainerSmoke(unittest.TestCase):
    def setUp(self):
        import train_lantra as t
        from src.agents.language.modules.language_transformer import LanguageTransformer

        self.t = t
        torch.manual_seed(0)
        self.model = LanguageTransformer(
            src_vocab_size=64, tgt_vocab_size=64, d_model=32, nhead=4, num_encoder_layers=1,
            num_decoder_layers=1, dim_feedforward=64, dropout=0.0, max_position_embeddings=64,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, label_smoothing=0.0,
            enable_memory_integration=False, batch_first=True,
        )
        self.model.eval()
        self.tok = FakeTokenizer()
        self.config = dataclasses.replace(
            t.config_from_args(t.parse_args(["--device", "cpu"])),
            batch_size=4, source_max_length=32, target_max_length=16, embedding_max_length=32,
            raw_max_length=48, amp="off", label_smoothing=0.0,
        )
        self.short = ([1, 10, 11, 2], [1, 20, 21, 2])
        self.long = ([1, 10, 11, 12, 13, 14, 15, 2], [1, 20, 21, 22, 23, 24, 2])

    def test_padded_batch_matches_single_examples(self):
        t = self.t
        with torch.no_grad():
            both = t.seq2seq_batch_losses(self.model, 0, [self.short, self.long], "cpu", self.config)
            first = t.seq2seq_batch_losses(self.model, 0, [self.short], "cpu", self.config)
            second = t.seq2seq_batch_losses(self.model, 0, [self.long], "cpu", self.config)
        self.assertTrue(torch.allclose(both, torch.cat([first, second]), atol=1e-4), (both, first, second))

    def test_generate_batch_padding_invariance_and_no_loops(self):
        src_short = torch.tensor([[1, 10, 11, 2]])
        src_long = torch.tensor([[1, 10, 11, 12, 13, 14, 15, 2]])
        padded = torch.tensor([[1, 10, 11, 2, 0, 0, 0, 0], [1, 10, 11, 12, 13, 14, 15, 2]])
        mask = padded.eq(0)
        kwargs = dict(max_len=12, min_length=2, repetition_penalty=1.15, no_repeat_ngram_size=3)
        alone = self.model.generate_batch(src_short, **kwargs)[0].tolist()
        batched = self.model.generate_batch(padded, src_key_padding_mask=mask, **kwargs)
        self.assertEqual(batched[0].tolist()[: len(alone)], alone)
        single_long = self.model.generate_batch(src_long, **kwargs)[0].tolist()
        self.assertEqual(batched[1].tolist()[: len(single_long)], single_long)
        for row in batched.tolist():
            body = row[1:]
            trigrams = [tuple(body[i:i + 3]) for i in range(len(body) - 2) if 2 not in body[i:i + 3]]
            self.assertEqual(len(trigrams), len(set(trigrams)), row)

    def test_loss_decreases_on_memorisable_batch(self):
        t = self.t
        pairs = [self.short, self.long, ([1, 30, 31, 2], [1, 40, 41, 42, 2]), ([1, 33, 34, 35, 2], [1, 44, 2])]
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        before = float(t.seq2seq_batch_losses(self.model, 0, pairs, "cpu", self.config).mean())
        for _ in range(60):
            optimizer.zero_grad()
            t.seq2seq_batch_losses(self.model, 0, pairs, "cpu", self.config).mean().backward()
            optimizer.step()
        after = float(t.seq2seq_batch_losses(self.model, 0, pairs, "cpu", self.config).mean())
        self.assertLess(after, before * 0.6, (before, after))

    def test_pairwise_and_example_batches(self):
        t = self.t
        examples = [
            t.TrainingExample(example_id="e", task="embedding", split="train", anchor="a b c", positive="a b", negative="z y x"),
            t.TrainingExample(example_id="r", task="reranking", split="train", anchor="q w", positive="q w e", negative="n m"),
        ]
        losses, aux = t.pairwise_batch_losses(self.model, self.tok, examples, self.config, "cpu")
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(all("pairwise_correct" in item for item in aux))
        dialogues = [t.TrainingExample(example_id=f"d{i}", task="dialogue", split="train", source="user: hi there", target="hello friend") for i in range(3)]
        for batch in t.example_micro_batches(dialogues + examples, self.config):
            batch_losses, _ = t.example_batch_losses(self.model, self.tok, batch, self.config, "cpu")
            self.assertTrue(torch.isfinite(batch_losses).all())

    def test_predictions_and_health_probe(self):
        t = self.t
        sources = [f"task: dialogue\nconversation:\nuser: question number {i}\nassistant:" for i in range(6)]
        replies = t.generate_predictions(self.model, self.tok, sources, [True] * 6, self.config, "cpu")
        self.assertEqual(len(replies), 6)
        context = t.HealthContext(tuple(sources), tuple(f"question number {i}" for i in range(6)), tuple("f" for _ in range(6)))
        report = t.probe_generation_health(self.model, self.tok, context, self.config, "cpu")
        self.assertIn("penalty", report)
        self.assertEqual(len(report["samples"]), 6)

    def test_raw_pretraining_pairs_train_a_step(self):
        t = self.t
        import random

        pairs = t.raw_training_pairs(self.tok, [" ".join(f"w{i}" for i in range(30))] * 6, self.config, random.Random(0))
        self.assertTrue(pairs)
        self.model.train()
        loss = t.seq2seq_batch_losses(self.model, 0, pairs, "cpu", self.config).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in self.model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
