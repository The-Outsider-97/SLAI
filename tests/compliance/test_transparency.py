"""EU AI Act Art. 50: disclosure, signed machine-readable marking, fail-closed release."""
from __future__ import annotations

import struct
import unittest
import zlib

from src.agents.governance.risk_classification import AISystemProfile
from src.agents.governance.transparency import (
    LIMITATIONS, TransparencyError, TransparencyMarker, interaction_disclosure, required_measures, visible_label,
)

KEY = b"t" * 40


def tiny_png() -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def profile(**kw):
    return AISystemProfile(system_id="sys-1", name="SLAI Writer", intended_purpose="drafting", is_gpai_model=False, **kw)


class MeasureSelection(unittest.TestCase):
    def codes(self, **kw):
        return {m.code for m in required_measures(profile(**kw))}

    def test_chat_needs_disclosure_unless_obvious(self):
        self.assertIn("interaction_disclosure", self.codes(interacts_with_natural_persons=True))
        self.assertNotIn("interaction_disclosure", self.codes(interacts_with_natural_persons=True, interaction_obvious_to_reasonable_person=True))

    def test_synthetic_content_needs_marking_unless_assistive_only(self):
        self.assertIn("machine_readable_marking", self.codes(generates_synthetic_content=True))
        self.assertNotIn("machine_readable_marking", self.codes(generates_synthetic_content=True, synthetic_content_only_assistive_editing=True))

    def test_deepfake_emotion_and_public_interest(self):
        self.assertIn("deepfake_visible_label", self.codes(deepfake_content=True))
        self.assertIn("exposed_person_notice", self.codes(emotion_recognition=True))
        self.assertIn("public_interest_text_label", self.codes(public_interest_text=True, generates_synthetic_content=True))

    def test_nothing_required_for_plain_system(self):
        self.assertEqual(self.codes(), set())


class MarkingTests(unittest.TestCase):
    def setUp(self):
        self.m = TransparencyMarker(KEY, clock=lambda: 1_800_000_000)

    def test_text_manifest_verifies_and_detects_tampering(self):
        r = self.m.release(profile(generates_synthetic_content=True), "Hello world", model="lantra-1")
        data = b"Hello world"
        self.assertTrue(self.m.verify(data, r.manifest))
        self.assertFalse(self.m.verify(b"Hello worle", r.manifest))
        forged = {**r.manifest, "model": "other"}
        self.assertFalse(self.m.verify(data, forged))
        self.assertFalse(TransparencyMarker(b"x" * 40).verify(data, r.manifest))

    def test_png_embedding_roundtrip_and_tamper(self):
        png = tiny_png()
        r = self.m.release(profile(generates_synthetic_content=True), png, model="img-1", kind="png")
        self.assertNotEqual(r.content, png)
        embedded = self.m.extract_png(r.content)
        self.assertEqual(embedded, r.manifest)
        self.assertTrue(self.m.verify(self.m.png_content_hash_source(r.content), embedded))
        self.assertTrue(r.content.startswith(b"\x89PNG"))
        tampered = bytearray(r.content)
        tampered[40] ^= 0xFF
        self.assertFalse(self.m.verify(self.m.png_content_hash_source(bytes(tampered)), embedded))

    def test_missing_key_fails_closed_for_marking(self):
        with self.assertRaises(TransparencyError):
            TransparencyMarker(None).release(profile(generates_synthetic_content=True), "x", model="m")

    def test_unmarkable_content_fails_closed(self):
        with self.assertRaises(TransparencyError):
            self.m.release(profile(generates_synthetic_content=True), {"obj": 1}, model="m", kind="object")

    def test_no_key_needed_when_no_marking_required(self):
        r = TransparencyMarker(None).release(profile(interacts_with_natural_persons=True), "hi", model="m", language="nl")
        self.assertIsNone(r.manifest)
        self.assertIn("AI-systeem", r.disclosures[0])

    def test_deepfake_gets_visible_label_and_limitations_are_reported(self):
        r = self.m.release(profile(deepfake_content=True, generates_synthetic_content=True), b"\x00\x01", model="v", kind="video")
        self.assertIn("artificially generated", r.visible_label)
        self.assertIn("deepfake_visible_label", r.measures_applied)
        self.assertEqual(r.limitations, LIMITATIONS)
        self.assertTrue(any("watermark" in l for l in LIMITATIONS))

    def test_texts(self):
        self.assertIn("AI system", interaction_disclosure("X"))
        self.assertIn("kunstmatig", visible_label("deepfake", "nl"))


if __name__ == "__main__":
    unittest.main()
