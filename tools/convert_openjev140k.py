#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "huggingface_hub>=0.23",
# ]
# ///
"""
Convert shenjunhao/OpenJevData-140k to Open Jev exemplar JSONL
and invoke the native pjevfilldb builder.

Two-stage pipeline:
  1. Python: download, validate, convert → JSONL + manifest
  2. Native pjevfilldb: embed + store → models/examplar.db

Serialization policy:
  - UTF-8, ensure_ascii=False
  - One JSON object per line
  - Keys in fixed declaration order (sort_keys=False)
  - Nested dicts with sort_keys=True for determinism
  - No trailing whitespace or BOM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

DATASET_REPO       = "shenjunhao/OpenJevData-140k"
SOURCE_STRING      = DATASET_REPO
CONVERSION_VERSION = 1

VALID_SPLITS   = {"SFT", "RL"}
VALID_KINDS    = {"hard", "soft"}
VALID_TASKS    = {"choice", "noul", "score"}

# CJK Unicode block ranges (BMP only; supplementary planes skipped to avoid
# regex parsing ambiguity with \uXXXX 4-digit escapes).
_CJK_RE = re.compile(
    "[\u4e00-\u9fff"    # CJK Unified Ideographs
    "\u3400-\u4dbf"     # CJK Extension A
    "\uf900-\ufaff"     # CJK Compatibility Ideographs
    "\u3000-\u303f"     # CJK Symbols and Punctuation
    "]"
)
# ASCII alphabetic
_ALPHA_RE = re.compile(r"[A-Za-z]")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(texts: list[str]) -> str:
    """
    Lightweight deterministic language classifier.

    Policy:
    - If combined text has CJK fraction > 0.15 of (CJK + ASCII-alpha) chars → "zh"
    - If CJK fraction <= 0.05 → "en"
    - Otherwise → "mixed"

    This is conservative: uncertain/mixed records are classified as "mixed"
    and excluded by the default English-only filter.
    """
    combined = " ".join(t for t in texts if isinstance(t, str))
    n_cjk   = len(_CJK_RE.findall(combined))
    n_alpha = len(_ALPHA_RE.findall(combined))
    total   = n_cjk + n_alpha
    if total == 0:
        return "unknown"
    ratio = n_cjk / total
    if ratio > 0.15:
        return "zh"
    if ratio <= 0.05:
        return "en"
    return "mixed"


def classify_record_language(raw: dict) -> str:
    """
    Detect language for a source record.

    Preference order:
    1. Explicit `language` field in record.
    2. Explicit `language` inside nested `answer` dict.
    3. Character-based detection on state + question + option names.
    """
    # Explicit field
    lang = raw.get("language")
    if isinstance(lang, str) and lang:
        return lang.lower()

    # Nested answer language
    answer = raw.get("answer")
    if isinstance(answer, dict):
        lang = answer.get("language")
        if isinstance(lang, str) and lang:
            return lang.lower()

    # Heuristic
    texts: list[str] = []
    state = raw.get("state", "")
    if isinstance(state, str):
        texts.append(state)
    question = raw.get("question", "")
    if isinstance(question, str):
        texts.append(question)
    options = raw.get("options", [])
    if isinstance(options, list):
        for opt in options:
            if isinstance(opt, dict):
                name = opt.get("name", "")
                if isinstance(name, str):
                    texts.append(name)
                crit = opt.get("criteria", "")
                if isinstance(crit, str):
                    texts.append(crit)

    return detect_language(texts)

# ---------------------------------------------------------------------------
# Option rendering
# ---------------------------------------------------------------------------

def render_option(opt: Any) -> str:
    """
    Render a single source option object to a deterministic non-empty string.

    Source schema: {"id": "o0", "name": "...", "criteria": null | "..."}

    Rendering:
      <name>
    or if criteria is present:
      <name>
      Criteria: <criteria>

    The source option ID (o0, o1, ...) is positional only and not included.
    """
    if isinstance(opt, str):
        return opt
    if not isinstance(opt, dict):
        return json.dumps(opt, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    name = opt.get("name")
    if not isinstance(name, str) or not name.strip():
        # Fall back to stable JSON
        return json.dumps(opt, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    parts = [name.strip()]
    criteria = opt.get("criteria")
    if criteria is not None and isinstance(criteria, str) and criteria.strip():
        parts.append(f"Criteria: {criteria.strip()}")
    elif criteria is not None and not isinstance(criteria, str):
        # Non-string criteria: include as JSON
        parts.append(f"Criteria: {json.dumps(criteria, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}")

    return "\n".join(parts)


def render_options(raw_options: Any) -> list[str]:
    if not isinstance(raw_options, list):
        raise ValueError(f"options is not a list: {type(raw_options)}")
    result: list[str] = []
    for i, opt in enumerate(raw_options):
        s = render_option(opt)
        if not s:
            raise ValueError(f"option {i} rendered to empty string")
        result.append(s)
    return result

# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

_ONE_HOT_TOL = 1e-6

def validate_target(target: list[float], n_options: int,
                    answer_kind: str) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(len(target) == n_options,
          f"target length {len(target)} != options length {n_options}")
    check(all(isinstance(v, (int, float)) for v in target),
          "target contains non-numeric value")
    check(all(v >= 0 for v in target),
          "target contains negative probability")
    check(all(abs(v) < 1e9 for v in target),  # finite check
          "target contains non-finite value")

    total = sum(target)
    check(abs(total - 1.0) < 1e-4,
          f"target probabilities sum to {total:.6f}, expected 1.0")

    if answer_kind == "hard":
        ones  = sum(1 for v in target if abs(v - 1.0) < _ONE_HOT_TOL)
        zeros = sum(1 for v in target if abs(v - 0.0) < _ONE_HOT_TOL)
        check(ones == 1, f"hard target has {ones} ones, expected 1")
        check(zeros == len(target) - 1,
              f"hard target has {zeros} zeros, expected {len(target) - 1}")

    return errors


def normalize_target(target: list[Any]) -> list[float]:
    return [float(v) for v in target]

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_record_id(split: str, index: int) -> str:
    """Deterministic per-record ID using original split and source order."""
    return f"openjev140k/{split}/{index:08d}"


def make_group_id(split: str, index: int) -> str:
    """One unique group per source record (no original group structure)."""
    return f"openjev140k/{split}/{index:08d}"

# ---------------------------------------------------------------------------
# Deduplication fingerprint
# ---------------------------------------------------------------------------

def content_fingerprint(state: str, question: str,
                        options: list[str]) -> str:
    payload = json.dumps(
        {"state": state, "question": question, "options": options},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def convert_record(
    raw: dict,
    split: str,
    index: int,
    include_soft: bool,
    lang_filter: str,   # "en" or "all"
    verbose: bool = False,
) -> dict:
    """
    Convert one OpenJevData-140k source record to pjevfilldb-compatible dict.
    Raises ValueError with descriptive message on invalid/excluded input.
    """
    # --- Language ---
    lang = classify_record_language(raw)
    if lang_filter == "en" and lang != "en":
        raise ValueError(f"language_excluded:{lang}")

    # --- Task kind ---
    task = raw.get("task")
    if not isinstance(task, str) or task not in VALID_TASKS:
        raise ValueError(f"unsupported task: {task!r}")
    kind = task  # direct mapping: choice/noul/score

    # --- Answer ---
    answer = raw.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("missing or non-object field: answer")
    answer_kind = answer.get("kind")
    if not isinstance(answer_kind, str) or answer_kind not in VALID_KINDS:
        raise ValueError(f"unsupported answer.kind: {answer_kind!r}")
    if answer_kind == "soft" and not include_soft:
        raise ValueError("answer_kind_excluded:soft")

    raw_probs = answer.get("probabilities")
    if not isinstance(raw_probs, list) or len(raw_probs) == 0:
        raise ValueError("missing or empty answer.probabilities")

    # --- Options ---
    raw_options = raw.get("options")
    if not isinstance(raw_options, list) or len(raw_options) == 0:
        raise ValueError("missing or empty options")
    options = render_options(raw_options)
    if len(options) < 2:
        raise ValueError(f"too few options: {len(options)}")

    # --- Target ---
    target = normalize_target(raw_probs)
    errs = validate_target(target, len(options), answer_kind)
    if errs:
        raise ValueError("invalid target: " + "; ".join(errs))

    # --- State ---
    state = raw.get("state")
    if state is None:
        state = ""
    if not isinstance(state, (str, dict, list)):
        state = str(state)

    # --- Question ---
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing or empty question")
    question = question.strip()

    # --- Category / domain ---
    category = raw.get("category")
    domain = category if isinstance(category, str) else ""

    # --- Score values ---
    score_values: list[int] | None = None
    if kind == "score":
        sv = raw.get("score_values")
        if isinstance(sv, list) and sv:
            score_values = [int(v) for v in sv]
            if len(score_values) != len(options):
                raise ValueError(
                    f"score_values length {len(score_values)} != "
                    f"options length {len(options)}")

    # --- IDs ---
    record_id = make_record_id(split, index)
    group_id  = make_group_id(split, index)

    # --- Metadata ---
    metadata: dict[str, Any] = {
        "language":     lang,
        "domain":       domain,
        "answer_kind":  answer_kind,
        "source_split": split,
    }
    if score_values is not None:
        metadata["score_values"] = score_values

    out: dict[str, Any] = {
        "group_id": group_id,
        "id":       record_id,
        "kind":     kind,
        "metadata": metadata,
        "options":  options,
        "question": question,
        "source":   SOURCE_STRING,
        "split":    split,
        "state":    state,
        "target":   target,
    }
    return out

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(rec: dict) -> list[str]:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(bool(rec.get("id")),       "id is empty")
    check(bool(rec.get("group_id")), "group_id is empty")
    check(rec.get("kind") in VALID_TASKS,
          f"kind not in valid set: {rec.get('kind')!r}")
    check(isinstance(rec.get("state"), (str, dict, list)),
          "state is not a JSON-serializable type")
    check(bool(rec.get("question")), "question is empty")

    options = rec.get("options", [])
    target  = rec.get("target",  [])
    check(isinstance(options, list),                      "options is not a list")
    check(isinstance(target,  list),                      "target is not a list")
    check(len(options) >= 2,                              f"too few options: {len(options)}")
    check(all(isinstance(o, str) for o in options),       "options contains non-string")
    check(all(o for o in options),                        "options contains empty string")
    check(len(target) == len(options),
          f"target length {len(target)} != options length {len(options)}")

    meta = rec.get("metadata", {})
    check(meta.get("language") == "en",  "metadata.language != 'en'")

    return errors

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class ConversionStats:
    def __init__(self) -> None:
        self.n_read         = 0
        self.n_converted    = 0
        self.n_skipped      = 0
        self.skip_reasons:   dict[str, int] = defaultdict(int)
        self.by_split:       dict[str, int] = defaultdict(int)
        self.by_kind:        dict[str, int] = defaultdict(int)
        self.by_answer_kind: dict[str, int] = defaultdict(int)
        self.by_domain:      dict[str, int] = defaultdict(int)
        self.by_lang:        dict[str, int] = defaultdict(int)
        self.by_n_options:   dict[int, int]  = defaultdict(int)
        self.by_n_levels:    dict[int, int]  = defaultdict(int)
        self.fingerprints:   dict[str, str]  = {}  # fp → first id
        self.duplicate_content = 0
        self.ids:            set[str]        = set()
        self.duplicate_ids   = 0

    def record_converted(self, rec: dict) -> None:
        self.n_converted += 1
        rid  = rec["id"]
        if rid in self.ids:
            self.duplicate_ids += 1
        self.ids.add(rid)

        meta = rec.get("metadata", {})
        self.by_split[rec["split"]] += 1
        self.by_kind[rec["kind"]] += 1
        self.by_answer_kind[meta.get("answer_kind", "?")] += 1
        self.by_domain[meta.get("domain", "")] += 1
        self.by_lang[meta.get("language", "?")] += 1
        n = len(rec["options"])
        self.by_n_options[n] += 1
        if rec["kind"] == "score":
            self.by_n_levels[n] += 1

        # Deduplication
        fp = content_fingerprint(
            state=json.dumps(rec["state"], ensure_ascii=False, sort_keys=True),
            question=rec["question"],
            options=rec["options"],
        )
        if fp in self.fingerprints:
            self.duplicate_content += 1
        else:
            self.fingerprints[fp] = rid

    def record_skipped(self, reason: str) -> None:
        self.n_skipped += 1
        key = reason.split(":")[0]   # group by prefix
        self.skip_reasons[key] += 1

    def to_manifest(
        self,
        dataset_repo: str,
        revision: str,
        splits: list[str],
        lang_filter: str,
        answer_kinds: list[str],
        output_jsonl: str | None,
        conversion_version: int,
    ) -> dict:
        output_size = 0
        output_sha256 = ""
        if output_jsonl and os.path.exists(output_jsonl):
            output_size = os.path.getsize(output_jsonl)
            with open(output_jsonl, "rb") as f:
                output_sha256 = hashlib.sha256(f.read()).hexdigest()

        return {
            "dataset_repository":    dataset_repo,
            "dataset_revision":      revision,
            "selected_splits":       splits,
            "language_filter":       lang_filter,
            "answer_kind_filter":    answer_kinds,
            "conversion_format_version": conversion_version,
            "records_read":          self.n_read,
            "records_converted":     self.n_converted,
            "records_skipped":       self.n_skipped,
            "skip_reasons":          dict(self.skip_reasons),
            "by_split":              dict(self.by_split),
            "by_kind":               dict(self.by_kind),
            "by_answer_kind":        dict(self.by_answer_kind),
            "by_domain_top20":       dict(
                sorted(self.by_domain.items(), key=lambda x: -x[1])[:20]),
            "by_detected_language":  dict(self.by_lang),
            "by_n_options":          {str(k): v
                                      for k, v in sorted(self.by_n_options.items())},
            "by_n_score_levels":     {str(k): v
                                      for k, v in sorted(self.by_n_levels.items())},
            "duplicate_ids":         self.duplicate_ids,
            "duplicate_content":     self.duplicate_content,
            "output_jsonl":          output_jsonl,
            "output_jsonl_size_bytes": output_size,
            "output_jsonl_sha256":   output_sha256,
        }

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_splits(
    splits: list[str],
    revision: str | None,
    verbose: bool,
) -> tuple[dict[str, Any], str]:
    """
    Load selected splits from HuggingFace.
    Returns (split_name → iterable, resolved_revision).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: uv add datasets",
              file=sys.stderr)
        sys.exit(1)

    datasets_by_split: dict[str, Any] = {}
    resolved_rev = revision or "unknown"

    for split in splits:
        kwargs: dict[str, Any] = {
            "path":  DATASET_REPO,
            "split": split,
        }
        if revision:
            kwargs["revision"] = revision

        if verbose:
            print(f"  Loading split={split} ...", file=sys.stderr)
        ds = load_dataset(**kwargs)

        # Try to extract commit SHA from dataset info
        if resolved_rev in ("unknown", "") or not revision:
            try:
                if hasattr(ds, "_info") and hasattr(ds._info, "version"):
                    v = str(ds._info.version)
                    if v and v not in ("0.0.0", "unknown"):
                        resolved_rev = v
                if hasattr(ds, "info"):
                    info = ds.info
                    for attr in ("download_checksums", "dataset_name"):
                        val = getattr(info, attr, None)
                        if val:
                            break
            except Exception:
                pass

        datasets_by_split[split] = ds
        if verbose:
            print(f"  Split {split}: {len(ds)} records", file=sys.stderr)

    return datasets_by_split, resolved_rev


def inspect_schema(record: dict, verbose: bool) -> None:
    """Print schema of the first record for debugging."""
    print("\n--- Source schema (first record) ---", file=sys.stderr)
    for k, v in record.items():
        t = type(v).__name__
        if isinstance(v, (str,)):
            preview = v[:80].replace("\n", "\\n")
            print(f"  {k!r:30s} ({t}): {preview!r}", file=sys.stderr)
        elif isinstance(v, list):
            print(f"  {k!r:30s} (list[{len(v)}]): "
                  f"{json.dumps(v[:2], ensure_ascii=False)[:120]}",
                  file=sys.stderr)
        elif isinstance(v, dict):
            print(f"  {k!r:30s} (dict{set(v.keys())})", file=sys.stderr)
        else:
            print(f"  {k!r:30s} ({t}): {v!r}", file=sys.stderr)
    print("---", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_all(
    datasets_by_split: dict[str, Any],
    output_jsonl: str,
    include_soft: bool,
    lang_filter: str,
    limit: int,
    validate_only: bool,
    verbose: bool,
    revision: str,
) -> ConversionStats:
    stats = ConversionStats()
    schema_inspected = False

    out_f = None
    if not validate_only:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        out_f = open(output_jsonl, "w", encoding="utf-8", newline="\n")

    try:
        for split, ds in datasets_by_split.items():
            split_idx = 0
            for raw in ds:
                if limit > 0 and stats.n_read >= limit:
                    break

                raw = dict(raw)
                stats.n_read += 1

                if not schema_inspected and verbose:
                    inspect_schema(raw, verbose)
                    schema_inspected = True

                try:
                    rec = convert_record(
                        raw=raw,
                        split=split,
                        index=split_idx,
                        include_soft=include_soft,
                        lang_filter=lang_filter,
                        verbose=verbose,
                    )
                except ValueError as e:
                    reason = str(e)
                    stats.record_skipped(reason)
                    if verbose:
                        print(f"  [skip] {split}#{split_idx}: {reason}",
                              file=sys.stderr)
                    split_idx += 1
                    continue

                errs = validate_record(rec)
                if errs:
                    reason = "validation: " + "; ".join(errs)
                    stats.record_skipped(reason)
                    if verbose:
                        print(f"  [invalid] {split}#{split_idx}: {reason}",
                              file=sys.stderr)
                    split_idx += 1
                    continue

                stats.record_converted(rec)

                if not validate_only and out_f is not None:
                    line = json.dumps(rec, ensure_ascii=False, sort_keys=False,
                                      separators=(",", ":"))
                    out_f.write(line + "\n")

                if verbose and stats.n_converted % 5000 == 0:
                    print(f"  converted {stats.n_converted} records ...",
                          file=sys.stderr)

                split_idx += 1

    finally:
        if out_f is not None:
            out_f.close()

    # Line-count sanity check
    if not validate_only and os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines != stats.n_converted:
            print(f"WARNING: line count {n_lines} != converted {stats.n_converted}",
                  file=sys.stderr)

    return stats


def find_pjevfilldb() -> str | None:
    """Search for the pjevfilldb executable in common build locations."""
    candidates = [
        "build/tools/pjevfilldb/pjevfilldb",
        "build/tools/pjevfilldb/pjevfilldb.exe",
        "build/Release/pjevfilldb.exe",
        "build/Debug/pjevfilldb.exe",
        "bin/pjevfilldb",
        "bin/pjevfilldb.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def run_pjevfilldb(
    pjevfilldb_path: str,
    input_jsonl: str,
    output_db: str,
    model: str,
    rebuild: bool,
    verbose: bool,
    limit: int,
) -> int:
    """Invoke the native pjevfilldb builder and return its exit code."""
    if not os.path.isfile(pjevfilldb_path):
        print(f"ERROR: pjevfilldb not found at {pjevfilldb_path!r}",
              file=sys.stderr)
        return 1

    cmd = [
        pjevfilldb_path,
        "--input",  input_jsonl,
        "--output", output_db,
        "--model",  model,
    ]
    if rebuild:
        cmd.append("--rebuild")
    if verbose:
        cmd.append("--verbose")
    if limit > 0:
        cmd += ["--limit", str(limit)]

    print(f"\nRunning pjevfilldb:", file=sys.stderr)
    print(f"  {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd)
    return result.returncode

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert shenjunhao/OpenJevData-140k to Open Jev exemplar JSONL\n"
            "and populate the native pjevfilldb SQLite database.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run tools/convert_openjev140k.py \\\n"
            "    --splits SFT \\\n"
            "    --language en \\\n"
            "    --answer-kinds hard \\\n"
            "    --output-jsonl datasets/openjev140k-exemplars.jsonl \\\n"
            "    --output-db models/examplar.db \\\n"
            "    --model models/nomic-embed-text-v1.5.gguf\n"
        ),
    )
    p.add_argument("--splits", default="SFT",
                   metavar="SFT|RL|SFT,RL",
                   help="Dataset splits to include (default: SFT)")
    p.add_argument("--language", default="en",
                   choices=["en", "all"],
                   help="Language filter: 'en' (default) or 'all'")
    p.add_argument("--answer-kinds", default="hard",
                   metavar="hard|hard,soft",
                   help="Answer kind filter (default: hard)")
    p.add_argument("--output-jsonl", default="datasets/openjev140k-exemplars.jsonl",
                   metavar="PATH", help="Intermediate JSONL path")
    p.add_argument("--output-db", default="models/examplar.db",
                   metavar="PATH", help="Output SQLite database path")
    p.add_argument("--model", default="models/nomic-embed-text-v1.5.gguf",
                   metavar="PATH", help="Nomic GGUF embedding model path")
    p.add_argument("--pjevfilldb", default=None,
                   metavar="PATH", help="Path to pjevfilldb executable")
    p.add_argument("--revision", default=None,
                   metavar="REVISION",
                   help="HuggingFace dataset git revision/commit hash")
    p.add_argument("--limit", type=int, default=0,
                   metavar="N",
                   help="Process at most N records (0 = all)")
    p.add_argument("--validate-only", action="store_true",
                   help="Convert and validate without writing output")
    p.add_argument("--rebuild", action="store_true",
                   help="Pass --rebuild to pjevfilldb (recreate DB from scratch)")
    p.add_argument("--skip-db-build", action="store_true",
                   help="Write JSONL but skip the pjevfilldb step")
    p.add_argument("--verbose", action="store_true",
                   help="Detailed progress logging")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Parse splits
    splits_raw = [s.strip() for s in args.splits.split(",")]
    invalid_splits = [s for s in splits_raw if s not in VALID_SPLITS]
    if invalid_splits:
        print(f"ERROR: unknown splits: {invalid_splits}. "
              f"Valid: {sorted(VALID_SPLITS)}", file=sys.stderr)
        return 1
    splits = splits_raw

    # Parse answer kinds
    answer_kinds_raw = [k.strip() for k in args.answer_kinds.split(",")]
    invalid_kinds = [k for k in answer_kinds_raw if k not in VALID_KINDS]
    if invalid_kinds:
        print(f"ERROR: unknown answer-kinds: {invalid_kinds}. "
              f"Valid: {sorted(VALID_KINDS)}", file=sys.stderr)
        return 1
    include_soft = "soft" in answer_kinds_raw

    output_jsonl = args.output_jsonl
    output_db    = args.output_db
    manifest_path = (
        os.path.splitext(output_jsonl)[0] + ".manifest.json"
    )

    # Safety: do not overwrite existing DB unless --rebuild
    if not args.validate_only and not args.skip_db_build and not args.rebuild:
        if os.path.exists(output_db):
            print(f"ERROR: database already exists at {output_db!r}.\n"
                  f"       Pass --rebuild to overwrite, or choose a different "
                  f"--output-db path.", file=sys.stderr)
            return 1

    print(f"Dataset:      {DATASET_REPO}", file=sys.stderr)
    print(f"Splits:       {splits}", file=sys.stderr)
    print(f"Language:     {args.language}", file=sys.stderr)
    print(f"Answer kinds: {answer_kinds_raw}", file=sys.stderr)
    print(f"Output JSONL: {output_jsonl}", file=sys.stderr)
    print(f"Output DB:    {output_db}", file=sys.stderr)
    if args.validate_only:
        print("Mode:         validate-only (no output written)", file=sys.stderr)

    t0 = time.monotonic()

    # --- Load datasets ---
    datasets_by_split, resolved_rev = load_splits(
        splits=splits, revision=args.revision, verbose=args.verbose)
    if args.revision:
        resolved_rev = args.revision

    print(f"Dataset revision: {resolved_rev}", file=sys.stderr)

    # --- Convert ---
    stats = convert_all(
        datasets_by_split=datasets_by_split,
        output_jsonl=output_jsonl,
        include_soft=include_soft,
        lang_filter=args.language,
        limit=args.limit,
        validate_only=args.validate_only,
        verbose=args.verbose,
        revision=resolved_rev,
    )

    elapsed_conv = time.monotonic() - t0

    # --- Print summary ---
    print(f"\nRecords read:      {stats.n_read}", file=sys.stderr)
    print(f"Converted:         {stats.n_converted}", file=sys.stderr)
    print(f"Skipped:           {stats.n_skipped}", file=sys.stderr)
    if stats.duplicate_ids:
        print(f"WARNING: {stats.duplicate_ids} duplicate IDs", file=sys.stderr)
    if stats.duplicate_content:
        print(f"NOTE: {stats.duplicate_content} duplicate content fingerprints",
              file=sys.stderr)
    if stats.skip_reasons:
        print("Skip reasons:", file=sys.stderr)
        for reason, count in sorted(stats.skip_reasons.items(),
                                    key=lambda x: -x[1]):
            print(f"  {count:7d}  {reason}", file=sys.stderr)
    print(f"By split:      {dict(stats.by_split)}", file=sys.stderr)
    print(f"By kind:       {dict(stats.by_kind)}", file=sys.stderr)
    print(f"By ans. kind:  {dict(stats.by_answer_kind)}", file=sys.stderr)
    print(f"By language:   {dict(stats.by_lang)}", file=sys.stderr)
    print(f"Conversion time: {elapsed_conv:.1f}s", file=sys.stderr)

    # --- Write manifest ---
    manifest = stats.to_manifest(
        dataset_repo=DATASET_REPO,
        revision=resolved_rev,
        splits=splits,
        lang_filter=args.language,
        answer_kinds=answer_kinds_raw,
        output_jsonl=output_jsonl if not args.validate_only else None,
        conversion_version=CONVERSION_VERSION,
    )
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Manifest: {manifest_path}", file=sys.stderr)

    if args.validate_only:
        print("\nValidation complete (no output written).", file=sys.stderr)
        return 0

    if stats.n_converted == 0:
        print("ERROR: no records converted — database build skipped.",
              file=sys.stderr)
        return 1

    if args.skip_db_build:
        print("\nSkipping pjevfilldb step (--skip-db-build).", file=sys.stderr)
        return 0

    # --- Find pjevfilldb ---
    filldb_path = args.pjevfilldb or find_pjevfilldb()
    if not filldb_path:
        print(
            "\nWARNING: pjevfilldb executable not found.\n"
            "Build it first, then run:\n"
            f"  pjevfilldb --input {output_jsonl} "
            f"--output {output_db} --model {args.model}",
            file=sys.stderr,
        )
        return 0

    # --- Run pjevfilldb ---
    rc = run_pjevfilldb(
        pjevfilldb_path=filldb_path,
        input_jsonl=output_jsonl,
        output_db=output_db,
        model=args.model,
        rebuild=args.rebuild,
        verbose=args.verbose,
        limit=args.limit,
    )
    if rc not in (0, 2):
        print(f"ERROR: pjevfilldb exited with code {rc}", file=sys.stderr)
        return rc

    t_total = time.monotonic() - t0
    db_size = os.path.getsize(output_db) if os.path.exists(output_db) else 0
    print(f"\nDatabase: {output_db} ({db_size / 1024 / 1024:.2f} MB)",
          file=sys.stderr)
    print(f"Total time: {t_total:.1f}s", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
