import unittest
from datetime import date

from src.agents.governance.risk_classification import RISK_DATES, AISystemProfile, RiskClassifier, RiskTier

C = RiskClassifier()


def ids(c):
    return {o.obligation_id for o in c.obligations}


class RiskClassificationTests(unittest.TestCase):
    def test_game_ai_and_math_are_minimal_not_high_risk(self):
        for purpose in ("tactical game AI (GangUP)", "mathematical calculation", "generic CAD file conversion"):
            c = C.classify(AISystemProfile("s", "s", intended_purpose=purpose, is_gpai_model=False))
            self.assertEqual(c.tier, RiskTier.MINIMAL, purpose)
            self.assertEqual(c.obligations, ())

    def test_autonomy_alone_does_not_make_a_system_high_risk(self):
        c = C.classify(AISystemProfile("s", "autonomous agent", intended_purpose="autonomous task planning", is_gpai_model=False))
        self.assertNotEqual(c.tier, RiskTier.HIGH_RISK)

    def test_annex_iii_areas_are_high_risk_with_omnibus_dates(self):
        for area in ("employment", "essential_services", "education", "law_enforcement", "biometrics"):
            c = C.classify(AISystemProfile("s", "s", intended_purpose="x", annex_iii_areas=(area,), is_gpai_model=False))
            self.assertEqual(c.tier, RiskTier.HIGH_RISK, area)
            self.assertEqual({o.applies_from for o in c.obligations if o.obligation_id.startswith("HR-")}, {RISK_DATES["high_risk_annex_iii"]})
            self.assertEqual(RISK_DATES["high_risk_annex_iii"], date(2027, 12, 2))

    def test_annex_i_requires_safety_component_and_third_party_conformity(self):
        only_component = C.classify(AISystemProfile("s", "s", intended_purpose="x", annex_i_safety_component=True, is_gpai_model=False))
        self.assertNotEqual(only_component.tier, RiskTier.HIGH_RISK)
        both = C.classify(AISystemProfile("s", "medical device AI", intended_purpose="x", annex_i_safety_component=True, annex_i_third_party_conformity=True, is_gpai_model=False))
        self.assertEqual(both.tier, RiskTier.HIGH_RISK)
        self.assertEqual({o.applies_from for o in both.obligations if o.obligation_id == "HR-RISK"}, {date(2028, 8, 2)})

    def test_art6_3_derogation_needs_documentation_and_never_applies_to_profiling(self):
        base = dict(system_id="s", name="s", intended_purpose="x", annex_iii_areas=("employment",), is_gpai_model=False)
        derog = C.classify(AISystemProfile(**base, art6_3_exception_claims=("preparatory_task",), art6_3_assessment_ref="ass-1"))
        self.assertNotEqual(derog.tier, RiskTier.HIGH_RISK)
        undocumented = C.classify(AISystemProfile(**base, art6_3_exception_claims=("preparatory_task",)))
        self.assertEqual(undocumented.tier, RiskTier.HIGH_RISK)
        profiling = C.classify(AISystemProfile(**base, art6_3_exception_claims=("preparatory_task",), art6_3_assessment_ref="ass-1", performs_profiling_of_persons=True))
        self.assertEqual(profiling.tier, RiskTier.HIGH_RISK)

    def test_art50_transparency_obligations_and_preexisting_marking_grace(self):
        chat = C.classify(AISystemProfile("s", "assistant", intended_purpose="chat", interacts_with_natural_persons=True, is_gpai_model=False))
        self.assertEqual(chat.tier, RiskTier.TRANSPARENCY)
        self.assertIn("T50-1", ids(chat))
        new_gen = C.classify(AISystemProfile("s", "gen", intended_purpose="gen", generates_synthetic_content=True, is_gpai_model=False))
        old_gen = C.classify(AISystemProfile("s", "gen", intended_purpose="gen", generates_synthetic_content=True, on_market_before_2026_08_02=True, is_gpai_model=False))
        self.assertEqual(next(o for o in new_gen.obligations if o.obligation_id == "T50-2").applies_from, date(2026, 8, 2))
        self.assertEqual(next(o for o in old_gen.obligations if o.obligation_id == "T50-2").applies_from, date(2026, 12, 2))
        assistive = C.classify(AISystemProfile("s", "spellcheck", intended_purpose="edit", generates_synthetic_content=True, synthetic_content_only_assistive_editing=True, is_gpai_model=False))
        self.assertNotIn("T50-2", ids(assistive))

    def test_obvious_ai_interaction_needs_no_art50_1_notice(self):
        c = C.classify(AISystemProfile("s", "s", intended_purpose="x", interacts_with_natural_persons=True, interaction_obvious_to_reasonable_person=True, is_gpai_model=False))
        self.assertNotIn("T50-1", ids(c))

    def test_gpai_is_never_inferred_from_size_and_unknowns_are_reported(self):
        c = C.classify(AISystemProfile("lantra", "LANTRA", intended_purpose="language model", training_compute_flops=1e26))
        self.assertNotIn("gpai", c.labels)
        self.assertTrue(any(q.startswith("gpai_status_unassessed") for q in c.open_questions))

    def test_gpai_systemic_by_compute_and_open_source_documentation_exemption(self):
        sysr = C.classify(AISystemProfile("m", "m", intended_purpose="x", is_gpai_model=True, gpai_generality_evidence_ref="eval-1", training_compute_flops=2e25, free_open_source_licence=True))
        self.assertIn("gpai_systemic_risk", sysr.labels)
        self.assertIn("GPAI-DOC", ids(sysr))  # exemption lost when systemic
        small = C.classify(AISystemProfile("m", "m", intended_purpose="x", is_gpai_model=True, gpai_generality_evidence_ref="eval-1", training_compute_flops=1e23, free_open_source_licence=True))
        self.assertNotIn("gpai_systemic_risk", small.labels)
        self.assertNotIn("GPAI-DOC", ids(small))
        self.assertIn("GPAI-COPYRIGHT", ids(small))
        self.assertIn("GPAI-SUMMARY", ids(small))

    def test_prohibited_beats_everything(self):
        c = C.classify(AISystemProfile("s", "s", intended_purpose="x", practices=("social_scoring",), annex_iii_areas=("employment",)))
        self.assertEqual(c.tier, RiskTier.PROHIBITED)

    def test_ruleset_is_versioned_and_states_its_verification_status(self):
        d = C.classify(AISystemProfile("s", "s", intended_purpose="x", is_gpai_model=False)).to_dict()
        self.assertIn("2026/1744", d["ruleset_version"])
        self.assertIn("not yet confirmed", d["ruleset_verification"])


if __name__ == "__main__":
    unittest.main()
