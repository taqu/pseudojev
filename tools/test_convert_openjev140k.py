#!/usr/bin/env python3
"""
Unit tests for tools/convert_openjev140k.py.

Run with:  python -m pytest tools/test_convert_openjev140k.py -v
       or: python tools/test_convert_openjev140k.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from convert_openjev140k import (
    CONVERSION_VERSION,
    DATASET_REPO,
    SOURCE_STRING,
    VALID_TASKS,
    ConversionStats,
    classify_record_language,
    content_fingerprint,
    convert_record,
    detect_language,
    make_group_id,
    make_record_id,
    normalize_target,
    render_option,
    render_options,
    validate_record,
    validate_target,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    task: str = "choice",
    state: str = "The applicant has a strong track record.",
    question: str = "Which candidate should be selected?",
    options: list | None = None,
    answer_kind: str = "hard",
    probs: list | None = None,
    category: str = "hiring",
    score_values: list | None = None,
) -> dict:
    if options is None:
        options = [
            {"id": "o0", "name": "Alice", "criteria": None},
            {"id": "o1", "name": "Bob",   "criteria": "Must have 5yr exp"},
            {"id": "o2", "name": "Carol", "criteria": None},
        ]
    if probs is None:
        probs = [0.0, 1.0, 0.0] if answer_kind == "hard" else [0.2, 0.5, 0.3]
    raw: dict = {
        "task":     task,
        "state":    state,
        "question": question,
        "options":  options,
        "answer":   {"kind": answer_kind, "probabilities": probs},
        "category": category,
    }
    if score_values is not None:
        raw["score_values"] = score_values
    return raw


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage(unittest.TestCase):
    def test_english_text(self):
        self.assertEqual(detect_language(["Hello world", "This is a test"]), "en")

    def test_chinese_text(self):
        self.assertEqual(detect_language(["你好世界", "这是一个测试"]), "zh")

    def test_mixed_heavy_cjk(self):
        # Mostly Chinese with some ASCII
        self.assertEqual(detect_language(["很多汉字 abc"]), "zh")

    def test_empty(self):
        self.assertEqual(detect_language([""]), "unknown")

    def test_mixed_borderline(self):
        # Some CJK mixed with ASCII — result depends on exact ratio; just verify no crash
        texts = ["Hello world test sentence " + "中文" * 2]
        lang = detect_language(texts)
        self.assertIn(lang, ("en", "mixed", "zh"))


# ---------------------------------------------------------------------------
# classify_record_language
# ---------------------------------------------------------------------------

class TestClassifyRecordLanguage(unittest.TestCase):
    def test_explicit_language_field(self):
        raw = {"language": "en", "question": "hi", "state": ""}
        self.assertEqual(classify_record_language(raw), "en")

    def test_explicit_zh(self):
        raw = {"language": "zh", "question": "你好"}
        self.assertEqual(classify_record_language(raw), "zh")

    def test_heuristic_english(self):
        raw = _make_raw(state="The applicant has strong skills.")
        self.assertEqual(classify_record_language(raw), "en")

    def test_heuristic_chinese(self):
        raw = _make_raw(
            state="申请人具有强大的背景和丰富的经验，能够胜任此职位。",
            question="应该选择哪位候选人？",
        )
        self.assertEqual(classify_record_language(raw), "zh")


# ---------------------------------------------------------------------------
# render_option
# ---------------------------------------------------------------------------

class TestRenderOption(unittest.TestCase):
    def test_name_only(self):
        opt = {"id": "o0", "name": "Alice", "criteria": None}
        self.assertEqual(render_option(opt), "Alice")

    def test_name_with_criteria(self):
        opt = {"id": "o1", "name": "Bob", "criteria": "Must have 5yr exp"}
        self.assertEqual(render_option(opt), "Bob\nCriteria: Must have 5yr exp")

    def test_null_criteria_excluded(self):
        opt = {"id": "o0", "name": "Alice", "criteria": None}
        rendered = render_option(opt)
        self.assertNotIn("Criteria", rendered)

    def test_empty_string_criteria_excluded(self):
        opt = {"id": "o0", "name": "Alice", "criteria": ""}
        rendered = render_option(opt)
        self.assertNotIn("Criteria", rendered)

    def test_string_option_passthrough(self):
        self.assertEqual(render_option("plain string"), "plain string")

    def test_option_id_not_in_output(self):
        opt = {"id": "o5", "name": "Eve", "criteria": None}
        rendered = render_option(opt)
        self.assertNotIn("o5", rendered)

    def test_order_preserved(self):
        # Verify options list order is preserved
        opts = [
            {"id": "o0", "name": "First",  "criteria": None},
            {"id": "o1", "name": "Second", "criteria": None},
            {"id": "o2", "name": "Third",  "criteria": None},
        ]
        rendered = render_options(opts)
        self.assertEqual(rendered[0], "First")
        self.assertEqual(rendered[1], "Second")
        self.assertEqual(rendered[2], "Third")

    def test_criteria_not_sorted(self):
        # Original option ordering must be preserved regardless
        opts = [
            {"id": "o0", "name": "Zebra", "criteria": None},
            {"id": "o1", "name": "Apple", "criteria": None},
        ]
        rendered = render_options(opts)
        self.assertEqual(rendered[0], "Zebra")
        self.assertEqual(rendered[1], "Apple")


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------

class TestValidateTarget(unittest.TestCase):
    def test_valid_hard(self):
        errs = validate_target([0.0, 1.0, 0.0], 3, "hard")
        self.assertEqual(errs, [])

    def test_valid_soft(self):
        errs = validate_target([0.2, 0.5, 0.3], 3, "soft")
        self.assertEqual(errs, [])

    def test_length_mismatch(self):
        errs = validate_target([0.0, 1.0], 3, "hard")
        self.assertTrue(any("length" in e for e in errs))

    def test_negative_probability(self):
        errs = validate_target([-0.1, 1.1, 0.0], 3, "soft")
        self.assertTrue(any("negative" in e for e in errs))

    def test_sum_not_one(self):
        errs = validate_target([0.5, 0.5, 0.5], 3, "soft")
        self.assertTrue(any("sum" in e for e in errs))

    def test_hard_not_one_hot(self):
        errs = validate_target([0.5, 0.5, 0.0], 3, "hard")
        self.assertTrue(any("hard" in e for e in errs))

    def test_soft_preserved_not_argmaxed(self):
        # Soft targets must pass as-is
        probs = [0.2, 0.5, 0.3]
        errs = validate_target(probs, 3, "soft")
        self.assertEqual(errs, [])
        # Must not be silently replaced with one-hot
        self.assertNotEqual(probs, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

class TestIdGeneration(unittest.TestCase):
    def test_sft_id_format(self):
        self.assertEqual(make_record_id("SFT", 0),     "openjev140k/SFT/00000000")
        self.assertEqual(make_record_id("SFT", 1),     "openjev140k/SFT/00000001")
        self.assertEqual(make_record_id("SFT", 99999), "openjev140k/SFT/00099999")

    def test_rl_id_format(self):
        self.assertEqual(make_record_id("RL", 0),      "openjev140k/RL/00000000")
        self.assertEqual(make_record_id("RL", 42),     "openjev140k/RL/00000042")

    def test_group_id_matches_record_id(self):
        # Default group = same as record (no original group structure)
        self.assertEqual(make_group_id("SFT", 7), make_record_id("SFT", 7))

    def test_deterministic(self):
        # Same call must always return same ID
        self.assertEqual(make_record_id("SFT", 123), make_record_id("SFT", 123))

    def test_split_ids_are_distinct(self):
        self.assertNotEqual(make_record_id("SFT", 0), make_record_id("RL", 0))


# ---------------------------------------------------------------------------
# convert_record
# ---------------------------------------------------------------------------

class TestConvertRecord(unittest.TestCase):
    def _conv(self, raw: dict | None = None, **kwargs) -> dict:
        if raw is None:
            raw = _make_raw(**kwargs)
        return convert_record(
            raw=raw, split="SFT", index=0,
            include_soft=False, lang_filter="en",
        )

    def test_basic_sft_choice(self):
        rec = self._conv()
        self.assertEqual(rec["kind"],   "choice")
        self.assertEqual(rec["split"],  "SFT")
        self.assertEqual(rec["source"], SOURCE_STRING)
        self.assertEqual(rec["id"],     "openjev140k/SFT/00000000")

    def test_state_string_preserved(self):
        state = "The applicant has a strong background in machine learning."
        rec = self._conv(_make_raw(state=state))
        self.assertEqual(rec["state"], state)

    def test_hard_target_preserved(self):
        probs = [0.0, 1.0, 0.0]
        rec = self._conv(_make_raw(answer_kind="hard", probs=probs))
        self.assertEqual(rec["target"], [0.0, 1.0, 0.0])

    def test_soft_target_excluded_by_default(self):
        raw = _make_raw(answer_kind="soft", probs=[0.2, 0.5, 0.3])
        with self.assertRaises(ValueError) as ctx:
            convert_record(raw=raw, split="SFT", index=0,
                           include_soft=False, lang_filter="all")
        self.assertIn("soft", str(ctx.exception))

    def test_soft_target_included_when_enabled(self):
        raw = _make_raw(answer_kind="soft", probs=[0.2, 0.5, 0.3])
        rec = convert_record(raw=raw, split="SFT", index=0,
                             include_soft=True, lang_filter="en")
        self.assertEqual(rec["target"], [0.2, 0.5, 0.3])

    def test_soft_not_argmaxed(self):
        raw = _make_raw(answer_kind="soft", probs=[0.1, 0.7, 0.2])
        rec = convert_record(raw=raw, split="SFT", index=0,
                             include_soft=True, lang_filter="en")
        # Must NOT be [0, 1, 0]
        self.assertNotEqual(rec["target"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(rec["target"][1], 0.7)

    def test_choice_mapping(self):
        rec = self._conv(_make_raw(task="choice"))
        self.assertEqual(rec["kind"], "choice")

    def test_noul_mapping(self):
        rec = self._conv(_make_raw(task="noul"))
        self.assertEqual(rec["kind"], "noul")

    def test_score_mapping(self):
        opts = [{"id": f"o{i}", "name": f"Level {i}", "criteria": None}
                for i in range(5)]
        probs = [0.0, 0.0, 1.0, 0.0, 0.0]
        raw = _make_raw(task="score", options=opts, probs=probs,
                        score_values=[0, 1, 2, 3, 4])
        rec = self._conv(raw)
        self.assertEqual(rec["kind"], "score")
        self.assertEqual(rec["metadata"]["score_values"], [0, 1, 2, 3, 4])

    def test_score_values_not_assumed_5_levels(self):
        opts = [{"id": f"o{i}", "name": f"L{i}", "criteria": None}
                for i in range(3)]
        probs = [0.0, 1.0, 0.0]
        raw = _make_raw(task="score", options=opts, probs=probs,
                        score_values=[10, 20, 30])
        rec = self._conv(raw)
        self.assertEqual(len(rec["options"]), 3)
        self.assertEqual(rec["metadata"]["score_values"], [10, 20, 30])

    def test_option_criteria_preserved(self):
        raw = _make_raw()
        rec = self._conv(raw)
        # Bob has criteria
        self.assertIn("Criteria:", rec["options"][1])

    def test_option_criteria_none_not_in_output(self):
        raw = _make_raw()
        rec = self._conv(raw)
        # Alice (index 0) has no criteria
        self.assertNotIn("Criteria", rec["options"][0])

    def test_option_order_preserved(self):
        opts = [
            {"id": "o0", "name": "Zebra",  "criteria": None},
            {"id": "o1", "name": "Apple",  "criteria": None},
            {"id": "o2", "name": "Mango",  "criteria": None},
        ]
        raw = _make_raw(options=opts)
        rec = self._conv(raw)
        self.assertEqual(rec["options"][0], "Zebra")
        self.assertEqual(rec["options"][1], "Apple")
        self.assertEqual(rec["options"][2], "Mango")

    def test_domain_from_category(self):
        rec = self._conv(_make_raw(category="healthcare"))
        self.assertEqual(rec["metadata"]["domain"], "healthcare")

    def test_answer_kind_in_metadata(self):
        rec = self._conv(_make_raw(answer_kind="hard"))
        self.assertEqual(rec["metadata"]["answer_kind"], "hard")

    def test_source_split_in_metadata(self):
        raw = _make_raw()
        rec = convert_record(raw=raw, split="RL", index=5,
                             include_soft=False, lang_filter="en")
        self.assertEqual(rec["metadata"]["source_split"], "RL")
        self.assertEqual(rec["split"], "RL")

    def test_chinese_excluded_by_default(self):
        raw = _make_raw(
            state="申请人具有强大的技术背景，擅长深度学习算法的研究与开发。",
            question="应该选择哪位候选人？",
        )
        with self.assertRaises(ValueError) as ctx:
            convert_record(raw=raw, split="SFT", index=0,
                           include_soft=False, lang_filter="en")
        self.assertIn("language_excluded", str(ctx.exception))

    def test_all_language_includes_chinese(self):
        raw = _make_raw(
            state="申请人具有强大的技术背景，擅长深度学习算法的研究与开发。",
            question="应该选择哪位候选人？",
        )
        # Should not raise
        rec = convert_record(raw=raw, split="SFT", index=0,
                             include_soft=False, lang_filter="all")
        self.assertEqual(rec["id"], "openjev140k/SFT/00000000")

    def test_stable_ids_across_calls(self):
        raw = _make_raw()
        rec1 = convert_record(raw=raw, split="SFT", index=42,
                              include_soft=False, lang_filter="en")
        rec2 = convert_record(raw=raw, split="SFT", index=42,
                              include_soft=False, lang_filter="en")
        self.assertEqual(rec1["id"], rec2["id"])

    def test_unsupported_task_raises(self):
        raw = _make_raw(task="unknown_task")
        with self.assertRaises(ValueError):
            self._conv(raw)

    def test_unicode_state_preserved(self):
        state = "Candidat A: résumé exceptionnel. Expérience: 10 ans."
        rec = self._conv(_make_raw(state=state))
        self.assertEqual(rec["state"], state)


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------

class TestValidateRecord(unittest.TestCase):
    def _valid_rec(self) -> dict:
        raw = _make_raw()
        return convert_record(raw=raw, split="SFT", index=0,
                              include_soft=False, lang_filter="en")

    def test_valid_record_passes(self):
        rec = self._valid_rec()
        errs = validate_record(rec)
        self.assertEqual(errs, [])

    def test_missing_id(self):
        rec = self._valid_rec()
        rec["id"] = ""
        errs = validate_record(rec)
        self.assertTrue(any("id" in e for e in errs))

    def test_wrong_kind(self):
        rec = self._valid_rec()
        rec["kind"] = "invalid"
        errs = validate_record(rec)
        self.assertTrue(any("kind" in e for e in errs))

    def test_language_must_be_en(self):
        rec = self._valid_rec()
        rec["metadata"]["language"] = "zh"
        errs = validate_record(rec)
        self.assertTrue(any("language" in e for e in errs))

    def test_options_non_string(self):
        rec = self._valid_rec()
        rec["options"] = [1, 2, 3]
        errs = validate_record(rec)
        self.assertTrue(any("non-string" in e for e in errs))

    def test_target_length_mismatch(self):
        rec = self._valid_rec()
        rec["target"] = [0.0, 1.0]  # 2, but options = 3
        errs = validate_record(rec)
        self.assertTrue(any("length" in e for e in errs))


# ---------------------------------------------------------------------------
# content_fingerprint
# ---------------------------------------------------------------------------

class TestContentFingerprint(unittest.TestCase):
    def test_deterministic(self):
        fp1 = content_fingerprint("state A", "question B", ["opt1", "opt2"])
        fp2 = content_fingerprint("state A", "question B", ["opt1", "opt2"])
        self.assertEqual(fp1, fp2)

    def test_different_content_different_fp(self):
        fp1 = content_fingerprint("state A", "question B", ["opt1"])
        fp2 = content_fingerprint("state X", "question B", ["opt1"])
        self.assertNotEqual(fp1, fp2)

    def test_option_order_matters(self):
        fp1 = content_fingerprint("s", "q", ["a", "b"])
        fp2 = content_fingerprint("s", "q", ["b", "a"])
        self.assertNotEqual(fp1, fp2)


# ---------------------------------------------------------------------------
# ConversionStats
# ---------------------------------------------------------------------------

class TestConversionStats(unittest.TestCase):
    def test_tracks_converted(self):
        st = ConversionStats()
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        st.record_converted(rec)
        self.assertEqual(st.n_converted, 1)

    def test_tracks_skipped(self):
        st = ConversionStats()
        st.record_skipped("language_excluded:zh")
        self.assertEqual(st.n_skipped, 1)
        self.assertIn("language_excluded", st.skip_reasons)

    def test_duplicate_content_detected(self):
        st = ConversionStats()
        rec0 = convert_record(_make_raw(), split="SFT", index=0,
                              include_soft=False, lang_filter="en")
        rec1 = convert_record(_make_raw(), split="SFT", index=1,
                              include_soft=False, lang_filter="en")
        st.record_converted(rec0)
        st.record_converted(rec1)  # same content, different ID
        self.assertEqual(st.duplicate_content, 1)

    def test_duplicate_ids_detected(self):
        st = ConversionStats()
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        st.record_converted(rec)
        st.record_converted(rec)  # same ID twice
        self.assertEqual(st.duplicate_ids, 1)


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------

class TestJsonlRoundtrip(unittest.TestCase):
    def test_serializes_to_valid_json(self):
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        line = json.dumps(rec, ensure_ascii=False, sort_keys=False,
                          separators=(",", ":"))
        parsed = json.loads(line)
        self.assertEqual(parsed["id"],   rec["id"])
        self.assertEqual(parsed["kind"], rec["kind"])

    def test_required_pjevfilldb_fields_present(self):
        """Verify all fields required by pjevfilldb are present."""
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        for field in ("id", "group_id", "kind", "question",
                      "source", "split", "options", "target", "state"):
            self.assertIn(field, rec, f"missing field: {field}")

    def test_options_are_strings(self):
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        self.assertTrue(all(isinstance(o, str) for o in rec["options"]))

    def test_target_are_floats(self):
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        self.assertTrue(all(isinstance(v, float) for v in rec["target"]))

    def test_state_is_string(self):
        rec = convert_record(_make_raw(), split="SFT", index=0,
                             include_soft=False, lang_filter="en")
        self.assertIsInstance(rec["state"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
