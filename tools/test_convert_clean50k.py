#!/usr/bin/env python3
"""
Unit tests for tools/convert_clean50k.py.

Run with:  python -m pytest tools/test_convert_clean50k.py -v
       or: python tools/test_convert_clean50k.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# Make convert_clean50k importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))

from convert_clean50k import (
    CONVERSION_VERSION,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_SPLIT,
    FALLBACK_QUESTION,
    SOURCE_STRING,
    ConversionStats,
    _check_no_leakage,
    _stable_json,
    content_hash,
    convert,
    convert_record,
    discover_schema,
    make_one_hot,
    normalize_state,
    render_option,
    render_options,
    validate_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REVISION = "test_rev_abc123"

def _sample_raw(
    n_options: int = 3,
    correct: int = 0,
    question: str | None = "Pick an action",
    state: object = None,
    uid: str = "rec_001",
    group: str | None = None,
) -> dict:
    """Build a minimal source record in a plausible CLEAN-50K schema."""
    options = [f"action_{i}" for i in range(n_options)]
    raw: dict = {
        "id":           uid,
        "question":     question,
        "options":      options,
        "answer_index": correct,
        "state":        state if state is not None else {"context": "test"},
    }
    if group:
        raw["group_id"] = group
    return raw

def _sample_mapping() -> dict[str, str]:
    return {
        "id":       "id",
        "group_id": "group_id",
        "question": "question",
        "options":  "options",
        "target":   "answer_index",
        "state":    "state",
    }

# ---------------------------------------------------------------------------
# make_one_hot
# ---------------------------------------------------------------------------

class TestMakeOneHot(unittest.TestCase):

    def test_first_option_correct(self):
        t = make_one_hot(0, 3)
        self.assertEqual(t, [1.0, 0.0, 0.0])

    def test_middle_option_correct(self):
        t = make_one_hot(2, 5)
        self.assertEqual(t, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_last_option_correct(self):
        t = make_one_hot(4, 5)
        self.assertEqual(t, [0.0, 0.0, 0.0, 0.0, 1.0])

    def test_two_options(self):
        self.assertEqual(make_one_hot(1, 2), [0.0, 1.0])
        self.assertEqual(make_one_hot(0, 2), [1.0, 0.0])

    def test_invalid_negative_index(self):
        with self.assertRaises(ValueError):
            make_one_hot(-1, 3)

    def test_out_of_range_index(self):
        with self.assertRaises(ValueError):
            make_one_hot(3, 3)

    def test_noninteger_index(self):
        with self.assertRaises(ValueError):
            make_one_hot("abc", 3)

    def test_float_index_converts(self):
        # Float that is an exact integer should convert OK
        t = make_one_hot(2.0, 4)
        self.assertEqual(t, [0.0, 0.0, 1.0, 0.0])

    def test_exactly_one_1(self):
        for n in range(2, 20):
            t = make_one_hot(n - 1, n)
            self.assertEqual(sum(t), 1.0)
            self.assertEqual(t[n - 1], 1.0)

    def test_large_option_set(self):
        t = make_one_hot(17, 20)
        self.assertEqual(len(t), 20)
        self.assertEqual(t[17], 1.0)
        self.assertEqual(sum(t), 1.0)


# ---------------------------------------------------------------------------
# render_option / render_options
# ---------------------------------------------------------------------------

class TestRenderOption(unittest.TestCase):

    def test_plain_string(self):
        self.assertEqual(render_option("search"), "search")

    def test_dict_with_standard_keys(self):
        opt = {"type": "tool", "name": "search_documents",
               "description": "Search docs.", "parameters": {"q": {"type": "str"}}}
        rendered = render_option(opt)
        self.assertIn("Type: tool", rendered)
        self.assertIn("Name: search_documents", rendered)
        self.assertIn("Description: Search docs.", rendered)
        self.assertIn("Parameters:", rendered)

    def test_dict_extra_keys(self):
        opt = {"name": "action_a", "custom_field": "xyz"}
        rendered = render_option(opt)
        self.assertIn("Name: action_a", rendered)
        self.assertIn("custom_field", rendered)

    def test_nested_parameter_schema(self):
        opt = {
            "name": "call_api",
            "parameters": {
                "url":    {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
            }
        }
        rendered = render_option(opt)
        self.assertIn("Parameters:", rendered)
        # Nested JSON must be deterministically serialized
        rendered2 = render_option(opt)
        self.assertEqual(rendered, rendered2)

    def test_nonempty_result(self):
        self.assertTrue(render_option("x"))
        self.assertTrue(render_option({"name": "a"}))

    def test_renders_list_as_json(self):
        rendered = render_option([1, 2, 3])
        self.assertIn("1", rendered)

    def test_options_list_two(self):
        opts = render_options(["opt_a", "opt_b"])
        self.assertEqual(opts, ["opt_a", "opt_b"])

    def test_options_list_five(self):
        raw = [{"name": f"action_{i}"} for i in range(5)]
        opts = render_options(raw)
        self.assertEqual(len(opts), 5)
        for o in opts:
            self.assertIsInstance(o, str)
            self.assertTrue(o)

    def test_options_more_than_16(self):
        raw = [f"option_{i}" for i in range(20)]
        opts = render_options(raw)
        self.assertEqual(len(opts), 20)

    def test_stable_ordering(self):
        raw = [{"name": "a", "type": "t"}, {"name": "b", "type": "t"}]
        opts1 = render_options(raw)
        opts2 = render_options(raw)
        self.assertEqual(opts1, opts2)

    def test_non_list_raises(self):
        with self.assertRaises(ValueError):
            render_options("not a list")


# ---------------------------------------------------------------------------
# normalize_state
# ---------------------------------------------------------------------------

class TestNormalizeState(unittest.TestCase):

    def test_dict_preserved(self):
        s = {"key": "value", "nested": {"x": 1}}
        self.assertEqual(normalize_state(s), s)

    def test_forbidden_keys_stripped(self):
        s = {"context": "abc", "answer_index": 2, "target_index": 1,
             "correct_option": 0, "correct_index": 2}
        result = normalize_state(s)
        self.assertNotIn("answer_index", result)
        self.assertNotIn("target_index", result)
        self.assertNotIn("correct_option", result)
        self.assertNotIn("correct_index", result)
        self.assertIn("context", result)

    def test_json_string_parsed(self):
        inner = {"goal": "find document"}
        s = json.dumps(inner)
        result = normalize_state(s)
        self.assertEqual(result, inner)

    def test_plain_string_preserved(self):
        s = "The agent is navigating a web page."
        self.assertEqual(normalize_state(s), s)

    def test_multiline_string(self):
        s = "Line 1\nLine 2\nLine 3"
        self.assertEqual(normalize_state(s), s)

    def test_empty_history(self):
        s = {"history": [], "goal": "find file"}
        result = normalize_state(s)
        self.assertEqual(result["history"], [])

    def test_none_returns_empty_dict(self):
        self.assertEqual(normalize_state(None), {})

    def test_nested_json_string_parsed(self):
        inner = {"a": {"b": 1}}
        s = json.dumps(inner)
        result = normalize_state(s)
        self.assertEqual(result, inner)


# ---------------------------------------------------------------------------
# discover_schema
# ---------------------------------------------------------------------------

class TestDiscoverSchema(unittest.TestCase):

    def test_standard_fields(self):
        sample = {
            "id": "x", "question": "q", "options": [],
            "answer_index": 0, "state": {}
        }
        m = discover_schema(sample)
        self.assertEqual(m["id"], "id")
        self.assertEqual(m["question"], "question")
        self.assertEqual(m["options"], "options")
        self.assertEqual(m["target"], "answer_index")
        self.assertEqual(m["state"], "state")

    def test_alternative_names(self):
        sample = {
            "decision_id": "x", "choices": ["a"], "correct": 0, "context": {}
        }
        m = discover_schema(sample)
        self.assertEqual(m.get("id"), "decision_id")
        self.assertEqual(m.get("options"), "choices")
        self.assertEqual(m.get("target"), "correct")
        self.assertEqual(m.get("state"), "context")

    def test_group_id_detected(self):
        sample = {"id": "x", "trajectory_id": "traj/1",
                  "options": [], "answer_index": 0, "state": {}}
        m = discover_schema(sample)
        self.assertEqual(m.get("group_id"), "trajectory_id")


# ---------------------------------------------------------------------------
# convert_record
# ---------------------------------------------------------------------------

class TestConvertRecord(unittest.TestCase):

    def _convert(self, raw: dict, mapping: dict | None = None) -> dict:
        if mapping is None:
            mapping = discover_schema(raw)
        return convert_record(raw, mapping, REVISION)

    def test_two_option_first_correct(self):
        raw = _sample_raw(n_options=2, correct=0)
        rec = self._convert(raw)
        self.assertEqual(rec["target"], [1.0, 0.0])
        self.assertEqual(len(rec["options"]), 2)
        self.assertEqual(rec["kind"], "choice")

    def test_five_option_middle_correct(self):
        raw = _sample_raw(n_options=5, correct=2)
        rec = self._convert(raw)
        self.assertEqual(rec["target"], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_last_option_correct(self):
        raw = _sample_raw(n_options=4, correct=3)
        rec = self._convert(raw)
        self.assertEqual(rec["target"][3], 1.0)

    def test_more_than_16_options(self):
        raw = _sample_raw(n_options=20, correct=15)
        rec = self._convert(raw)
        self.assertEqual(len(rec["options"]), 20)
        self.assertEqual(rec["target"][15], 1.0)

    def test_id_namespaced(self):
        raw = _sample_raw(uid="abc123")
        rec = self._convert(raw)
        self.assertEqual(rec["id"], "clean50k/abc123")

    def test_group_id_namespaced(self):
        raw = _sample_raw(uid="x", group="traj/42")
        rec = self._convert(raw)
        self.assertEqual(rec["group_id"], "clean50k/traj/42")

    def test_group_id_fallback_to_id(self):
        raw = _sample_raw(uid="xyz")
        raw.pop("group_id", None)
        mapping = discover_schema(raw)
        mapping.pop("group_id", None)
        rec = convert_record(raw, mapping, REVISION)
        self.assertEqual(rec["group_id"], rec["id"])

    def test_question_preserved(self):
        raw = _sample_raw(question="Which action fits?")
        rec = self._convert(raw)
        self.assertEqual(rec["question"], "Which action fits?")

    def test_question_fallback(self):
        raw = _sample_raw(question=None)
        raw.pop("question", None)
        mapping = discover_schema(raw)
        mapping.pop("question", None)
        rec = convert_record(raw, mapping, REVISION)
        self.assertEqual(rec["question"], FALLBACK_QUESTION)

    def test_kind_always_choice(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["kind"], "choice")

    def test_split_is_validation(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["split"], "validation")

    def test_metadata_language_en(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["language"], "en")

    def test_metadata_source_split_train(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["source_split"], DATASET_SPLIT)

    def test_metadata_evaluation_role(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["evaluation_role"],
                         "external_retrieval_queries")

    def test_metadata_dataset_fields(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["dataset"], DATASET_REPO)
        self.assertEqual(rec["metadata"]["dataset_config"], DATASET_CONFIG)

    def test_metadata_original_id(self):
        raw = _sample_raw(uid="orig_007")
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["original_id"], "orig_007")

    def test_metadata_revision(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["metadata"]["dataset_revision"], REVISION)

    def test_source_field(self):
        raw = _sample_raw()
        rec = self._convert(raw)
        self.assertEqual(rec["source"], SOURCE_STRING)

    def test_state_is_dict(self):
        raw = _sample_raw(state={"goal": "find file", "history": []})
        rec = self._convert(raw)
        self.assertIsInstance(rec["state"], dict)
        self.assertEqual(rec["state"]["history"], [])

    def test_multiline_state(self):
        raw = _sample_raw(state={"desc": "line1\nline2\nline3"})
        rec = self._convert(raw)
        self.assertIn("\n", rec["state"]["desc"])

    def test_nested_parameter_schemas(self):
        opts = [
            {"name": "call", "parameters": {"a": {"type": "string", "enum": ["x"]}}},
            {"name": "skip"},
        ]
        raw = _sample_raw(n_options=2, correct=0)
        raw["options"] = opts
        mapping = discover_schema(raw)
        rec = convert_record(raw, mapping, REVISION)
        # Parameters must be serialized in options
        self.assertIn("Parameters:", rec["options"][0])

    def test_empty_history_in_state(self):
        raw = _sample_raw(state={"history": [], "goal": "navigate"})
        rec = self._convert(raw)
        self.assertEqual(rec["state"]["history"], [])

    def test_missing_optional_metadata_null_fields(self):
        raw = _sample_raw()
        raw.pop("group_id", None)
        mapping = discover_schema(raw)
        rec = convert_record(raw, mapping, REVISION)
        self.assertIn("original_id", rec["metadata"])

    def test_null_optional_fields(self):
        raw = _sample_raw()
        raw["group_id"] = None
        rec = self._convert(raw)
        # group_id None should fall back to record ID
        self.assertEqual(rec["group_id"], rec["id"])

    def test_option_order_preserved(self):
        opts = ["z_action", "a_action", "m_action"]
        raw = {"id": "x", "question": "q", "options": opts,
               "answer_index": 1, "state": {}}
        mapping = discover_schema(raw)
        rec = convert_record(raw, mapping, REVISION)
        self.assertEqual(rec["options"], ["z_action", "a_action", "m_action"])
        self.assertEqual(rec["target"][1], 1.0)

    def test_no_answer_leakage_in_state(self):
        # Correct index must NOT appear in state under suspicious key names
        raw = _sample_raw(n_options=3, correct=1)
        rec = self._convert(raw)
        if isinstance(rec["state"], dict):
            for k in ("answer_index", "correct_index", "label"):
                self.assertNotIn(k, rec["state"])


# ---------------------------------------------------------------------------
# No target leakage
# ---------------------------------------------------------------------------

class TestNoLeakage(unittest.TestCase):

    def test_leakage_raises(self):
        with self.assertRaises(ValueError):
            _check_no_leakage(
                state={"answer_index": 2},
                question="q",
                options=["a", "b", "c"],
                target=[0.0, 0.0, 1.0],
            )

    def test_no_leakage_innocent_int(self):
        # An integer that happens to equal the correct index but under a
        # non-suspicious key must not raise.
        _check_no_leakage(
            state={"retry_count": 2},
            question="q",
            options=["a", "b", "c"],
            target=[0.0, 0.0, 1.0],
        )

    def test_no_leakage_clean_state(self):
        _check_no_leakage(
            state={"history": [], "goal": "file"},
            question="Pick one",
            options=["a", "b"],
            target=[1.0, 0.0],
        )


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------

class TestValidateRecord(unittest.TestCase):

    def _good_record(self, n: int = 3, correct: int = 0) -> dict:
        raw = _sample_raw(n_options=n, correct=correct)
        mapping = discover_schema(raw)
        return convert_record(raw, mapping, REVISION)

    def test_valid_record_no_errors(self):
        rec = self._good_record()
        self.assertEqual(validate_record(rec), [])

    def test_empty_id_fails(self):
        rec = self._good_record()
        rec["id"] = ""
        errors = validate_record(rec)
        self.assertTrue(any("id" in e for e in errors))

    def test_wrong_kind_fails(self):
        rec = self._good_record()
        rec["kind"] = "score"
        errors = validate_record(rec)
        self.assertTrue(any("kind" in e for e in errors))

    def test_target_length_mismatch_fails(self):
        rec = self._good_record(n=3)
        rec["target"] = [1.0, 0.0]  # wrong length
        errors = validate_record(rec)
        self.assertTrue(any("target" in e for e in errors))

    def test_two_ones_in_target_fails(self):
        rec = self._good_record(n=3)
        rec["target"] = [1.0, 1.0, 0.0]
        errors = validate_record(rec)
        self.assertTrue(any("ones" in e for e in errors))

    def test_no_ones_in_target_fails(self):
        rec = self._good_record(n=3)
        rec["target"] = [0.0, 0.0, 0.0]
        errors = validate_record(rec)
        self.assertTrue(any("ones" in e for e in errors))

    def test_one_option_fails(self):
        rec = self._good_record(n=2)
        rec["options"] = ["only_one"]
        rec["target"] = [1.0]
        errors = validate_record(rec)
        self.assertTrue(any("options" in e for e in errors))

    def test_empty_question_fails(self):
        rec = self._good_record()
        rec["question"] = ""
        errors = validate_record(rec)
        self.assertTrue(any("question" in e for e in errors))

    def test_one_hot_target_validity(self):
        for n in range(2, 10):
            for correct in range(n):
                rec = self._good_record(n=n, correct=correct)
                errors = validate_record(rec)
                self.assertEqual(errors, [],
                                 f"n={n} correct={correct}: {errors}")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_stable_json(self):
        d = {"b": 1, "a": 2, "c": {"z": 3, "y": 4}}
        j1 = _stable_json(d)
        j2 = _stable_json(d)
        self.assertEqual(j1, j2)

    def test_content_hash_deterministic(self):
        raw = _sample_raw(n_options=3, correct=1)
        h1 = content_hash(raw)
        h2 = content_hash(raw)
        self.assertEqual(h1, h2)

    def test_convert_record_deterministic(self):
        raw = _sample_raw(n_options=3, correct=1, uid="det_test")
        mapping = discover_schema(raw)
        r1 = convert_record(raw, mapping, REVISION)
        r2 = convert_record(raw, mapping, REVISION)
        self.assertEqual(
            json.dumps(r1, ensure_ascii=False, sort_keys=False),
            json.dumps(r2, ensure_ascii=False, sort_keys=False),
        )

    def test_serialized_output_deterministic(self):
        raw = _sample_raw(n_options=4, correct=2)
        mapping = discover_schema(raw)
        rec = convert_record(raw, mapping, REVISION)
        s1 = json.dumps(rec, ensure_ascii=False, sort_keys=False,
                        separators=(",", ":"))
        s2 = json.dumps(rec, ensure_ascii=False, sort_keys=False,
                        separators=(",", ":"))
        self.assertEqual(s1, s2)


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------

class TestJsonlRoundTrip(unittest.TestCase):

    def test_roundtrip(self):
        records = [_sample_raw(n_options=i + 2, correct=0, uid=f"r{i}")
                   for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out.jsonl")
            stats_path = os.path.join(tmpdir, "out.stats.json")
            # Override stats path inside convert() by monkeypatching is complex;
            # write a wrapper.
            mapping = discover_schema(records[0])

            with open(out, "w", encoding="utf-8", newline="\n") as f:
                for raw in records:
                    rec = convert_record(raw, mapping, REVISION)
                    f.write(json.dumps(rec, ensure_ascii=False,
                                       separators=(",", ":")) + "\n")

            with open(out, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 5)
            for line in lines:
                rec = json.loads(line)
                errors = validate_record(rec)
                self.assertEqual(errors, [])

    def test_unique_ids(self):
        records = [_sample_raw(n_options=2, correct=0, uid=f"uid_{i}")
                   for i in range(10)]
        mapping = discover_schema(records[0])
        ids = [convert_record(r, mapping, REVISION)["id"] for r in records]
        self.assertEqual(len(set(ids)), len(ids))


# ---------------------------------------------------------------------------
# Integration: convert() function
# ---------------------------------------------------------------------------

class TestConvertFunction(unittest.TestCase):

    def _make_records(self, n: int) -> list[dict]:
        return [_sample_raw(n_options=3, correct=i % 3, uid=f"conv_{i}")
                for i in range(n)]

    def test_convert_writes_correct_count(self):
        records = self._make_records(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            ret = convert(
                records=records,
                output_path=out,
                stats_path=stats,
                limit=0,
                validate_only=False,
                strict=False,
                verbose=False,
                dataset_revision=REVISION,
            )
            self.assertEqual(ret, 0)
            with open(out) as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 10)

    def test_convert_limit(self):
        records = self._make_records(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=5, validate_only=False, strict=False,
                    verbose=False, dataset_revision=REVISION)
            with open(out) as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 5)

    def test_convert_validate_only_no_file(self):
        records = self._make_records(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=True, strict=False,
                    verbose=False, dataset_revision=REVISION)
            self.assertFalse(os.path.exists(out))

    def test_convert_stats_written(self):
        records = self._make_records(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=False, strict=False,
                    verbose=False, dataset_revision=REVISION)
            self.assertTrue(os.path.exists(stats))
            with open(stats) as f:
                d = json.load(f)
            self.assertEqual(d["records_converted"], 5)
            self.assertEqual(d["records_read"], 5)

    def test_convert_invalid_record_skipped_nonstrict(self):
        records = self._make_records(4)
        # Poison one record with an out-of-range target
        records[1]["answer_index"] = 99
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=False, strict=False,
                    verbose=False, dataset_revision=REVISION)
            with open(out) as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 3)

    def test_convert_strict_stops_on_first_error(self):
        records = self._make_records(4)
        records[0]["answer_index"] = 99  # poison first record
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            ret = convert(records=records, output_path=out, stats_path=stats,
                          limit=0, validate_only=False, strict=True,
                          verbose=False, dataset_revision=REVISION)
            self.assertNotEqual(ret, 0)

    def test_all_output_records_valid(self):
        records = self._make_records(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=False, strict=True,
                    verbose=False, dataset_revision=REVISION)
            with open(out) as f:
                for line in f:
                    rec = json.loads(line)
                    errors = validate_record(rec)
                    self.assertEqual(errors, [])

    def test_r0_required_fields_present(self):
        """Every output record must have all fields required by parse_exemplar()."""
        required = {"id", "group_id", "kind", "source", "split",
                    "question", "options", "target", "state"}
        records = self._make_records(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=False, strict=True,
                    verbose=False, dataset_revision=REVISION)
            with open(out) as f:
                for line in f:
                    rec = json.loads(line)
                    missing = required - rec.keys()
                    self.assertEqual(missing, set(),
                                     f"Missing R0 fields: {missing}")

    def test_language_en_in_metadata(self):
        """R0 skips records where metadata.language != 'en'."""
        records = self._make_records(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out   = os.path.join(tmpdir, "out.jsonl")
            stats = os.path.join(tmpdir, "out.stats.json")
            convert(records=records, output_path=out, stats_path=stats,
                    limit=0, validate_only=False, strict=True,
                    verbose=False, dataset_revision=REVISION)
            with open(out) as f:
                for line in f:
                    rec = json.loads(line)
                    self.assertEqual(rec.get("metadata", {}).get("language"), "en")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
