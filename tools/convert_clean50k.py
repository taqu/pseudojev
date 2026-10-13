#!/usr/bin/env python3
"""
Convert samatv256/jev-decisions-v1 (general-clean-50k) to Open Jev R0 JSONL.

R0 External Corpus Retrieval Sanity Check converter.
Output is compatible with tools/pjev-r0 JSONL input contract.

NOTE: General CLEAN-50K is a training subset of Jev Decisions v1.
This output should NOT be described as the official Open Jev validation set.

Serialization policy:
  - UTF-8, ensure_ascii=False
  - One JSON object per line
  - Keys emitted in fixed declaration order via sort_keys=False
  - Nested dicts serialized with sort_keys=True for determinism
  - No trailing whitespace or BOM
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

DATASET_REPO   = "samatv256/jev-decisions-v1"
DATASET_CONFIG = "general-clean-50k"
DATASET_SPLIT  = "train"
SOURCE_STRING  = f"{DATASET_REPO}/{DATASET_CONFIG}"
CONVERSION_VERSION = 1

FALLBACK_QUESTION = (
    "Which of the available actions should be selected"
    " given the current state?"
)

# ---------------------------------------------------------------------------
# Schema discovery helpers
# ---------------------------------------------------------------------------

def _first_str(d: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in d and isinstance(d[k], str):
            return k
    return None

def _first_any(d: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in d:
            return k
    return None

def discover_schema(sample: dict, verbose: bool = False) -> dict[str, str]:
    """Return a mapping of logical role -> actual field name."""
    keys = set(sample.keys())

    ID_CANDS      = ["id", "decision_id", "example_id", "uid", "row_id"]
    GROUP_CANDS   = ["group_id", "trajectory_id", "episode_id", "task_id", "group"]
    QUESTION_CANDS= ["question", "query", "prompt", "instruction"]
    OPTIONS_CANDS = ["options", "answer_options", "choices", "candidates",
                     "actions", "alternatives"]
    TARGET_CANDS  = ["target_index", "answer_index", "correct_index", "label",
                     "answer", "correct_option", "correct", "target"]
    STATE_CANDS   = ["state", "context", "observation", "scenario", "situation"]

    mapping: dict[str, str] = {}

    for role, cands in [
        ("id", ID_CANDS),
        ("group_id", GROUP_CANDS),
        ("question", QUESTION_CANDS),
        ("options",  OPTIONS_CANDS),
        ("target",   TARGET_CANDS),
        ("state",    STATE_CANDS),
    ]:
        found = _first_any(sample, cands)
        if found:
            mapping[role] = found
        elif verbose:
            print(f"  [schema] role '{role}' not found (tried {cands})",
                  file=sys.stderr)

    if verbose:
        print(f"  [schema] available keys: {sorted(keys)}", file=sys.stderr)
        print(f"  [schema] mapping:        {mapping}",      file=sys.stderr)

    return mapping

# ---------------------------------------------------------------------------
# Option rendering
# ---------------------------------------------------------------------------

def _stable_json(obj: Any) -> str:
    """JSON-serialize with sorted keys for determinism."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))

def render_option(opt: Any) -> str:
    """
    Render a single candidate as a deterministic nonempty string.

    Handles both the generic schema {type, name, description, parameters}
    and the CLEAN-50K schema {type, label, description, schema_json}.
    Only model-visible fields are emitted.
    """
    if isinstance(opt, str):
        return opt
    if not isinstance(opt, dict):
        return _stable_json(opt)

    parts: list[str] = []

    def emit(display: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            # Attempt to parse JSON-encoded strings for clean output
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    parts.append(f"{display}: {_stable_json(parsed)}")
                    return
                except json.JSONDecodeError:
                    pass
            parts.append(f"{display}: {value}")
        elif isinstance(value, (dict, list)):
            parts.append(f"{display}: {_stable_json(value)}")
        else:
            parts.append(f"{display}: {value}")

    # Emit in stable order; prefer `name` over `label` for the Name field
    emit("Type",        opt.get("type"))
    emit("Name",        opt.get("name") or opt.get("label"))
    emit("Description", opt.get("description"))
    # Prefer `parameters` (dict); fall back to `schema_json` (JSON string)
    params = opt.get("parameters") or opt.get("schema_json")
    emit("Parameters",  params)

    # Any remaining keys not covered above
    covered = {"type", "name", "label", "description", "parameters", "schema_json"}
    for k, v in opt.items():
        if k in covered:
            continue
        emit(k, v)

    rendered = "\n".join(parts)
    return rendered if rendered else _stable_json(opt)

def render_options(raw_options: Any) -> list[str]:
    if not isinstance(raw_options, list):
        raise ValueError(f"options is not a list: {type(raw_options)}")
    result = []
    for i, opt in enumerate(raw_options):
        s = render_option(opt)
        if not s:
            raise ValueError(f"option {i} rendered to empty string")
        result.append(s)
    return result

# ---------------------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------------------

def _parse_history_entry(entry: Any) -> Any:
    """
    Parse a history entry, decoding JSON-encoded `content` strings.
    Handles CLEAN-50K format where content is a JSON-encoded message object.
    """
    if not isinstance(entry, dict):
        return entry
    result = dict(entry)
    content = result.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                result["content"] = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return result


def normalize_state(raw: Any) -> Any:
    """
    Return a JSON-serializable state value.

    The R0 parser accepts any JSON value for state (object or string).
    We prefer to keep structured objects intact.
    If the source provides a JSON-serialized strings, parse them.
    """
    if isinstance(raw, dict):
        # Strip target-derived supervision fields that must never appear here.
        forbidden = {"answer_index", "correct_index", "target_index",
                     "correct_option", "correct"}
        cleaned: dict[str, Any] = {}
        for k, v in raw.items():
            if k in forbidden:
                continue
            # Parse JSON-encoded history entries
            if k == "history" and isinstance(v, list):
                cleaned[k] = [_parse_history_entry(e) for e in v]
            elif isinstance(v, str):
                stripped = v.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        cleaned[k] = json.loads(stripped)
                        continue
                    except json.JSONDecodeError:
                        pass
                cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return normalize_state(parsed)
            except json.JSONDecodeError:
                pass
        return raw  # plain string state; R0 accepts it

    if raw is None:
        return {}

    # Scalars, lists etc. — return as-is; R0 accepts any JSON value
    return raw

# ---------------------------------------------------------------------------
# Target conversion
# ---------------------------------------------------------------------------

def make_one_hot(index: Any, n_options: int) -> list[float]:
    """Convert a correct-option index to a one-hot float list."""
    if not isinstance(index, int):
        try:
            index = int(index)
        except (TypeError, ValueError):
            raise ValueError(f"target index is not an integer: {index!r}")
    if index < 0 or index >= n_options:
        raise ValueError(
            f"target index {index} out of range [0, {n_options})")
    target = [0.0] * n_options
    target[index] = 1.0
    return target

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def content_hash(record: dict) -> str:
    """Deterministic hash of model-visible record content."""
    payload = _stable_json({
        "question": record.get("question", ""),
        "options":  record.get("options",  []),
        "state":    record.get("state",    {}),
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def convert_record(
    raw: dict,
    mapping: dict[str, str],
    dataset_revision: str,
    verbose: bool = False,
) -> dict:
    """
    Convert one source record to an Open Jev R0-compatible dict.
    Raises ValueError with a descriptive message on invalid input.
    """
    # --- ID ---
    original_id: str | None = None
    if "id" in mapping:
        val = raw.get(mapping["id"])
        if isinstance(val, str) and val:
            original_id = val
        elif val is not None:
            original_id = str(val)

    if not original_id:
        # Deterministic content hash fallback
        original_id = "hash_" + content_hash(raw)

    record_id  = f"clean50k/{original_id}"

    # --- Group ID ---
    group_id = record_id  # default: no trajectory info
    if "group_id" in mapping:
        val = raw.get(mapping["group_id"])
        if isinstance(val, str) and val:
            group_id = f"clean50k/{val}"

    # --- Question ---
    question = FALLBACK_QUESTION
    if "question" in mapping:
        val = raw.get(mapping["question"])
        if isinstance(val, str) and val.strip():
            question = val.strip()

    # --- Options ---
    if "options" not in mapping:
        raise ValueError("no options field found in record")
    raw_options = raw[mapping["options"]]
    options = render_options(raw_options)
    if len(options) < 2:
        raise ValueError(f"too few options: {len(options)}")

    # Check for accidental collisions after rendering
    if len(set(options)) != len(options):
        raise ValueError(
            "rendered options contain duplicates after string conversion")

    # --- State ---
    raw_state = raw.get(mapping.get("state", "state"))
    state = normalize_state(raw_state)

    # --- Target ---
    if "target" not in mapping:
        raise ValueError("no target field found in record")
    raw_target = raw[mapping["target"]]
    target = make_one_hot(raw_target, len(options))

    # --- Anti-leakage check ---
    _check_no_leakage(state, question, options, target)

    # --- Metadata ---
    metadata: dict[str, Any] = {
        "language":          "en",
        "domain":            "agent_decision",
        "family":            "tool_choice",
        "dataset":           DATASET_REPO,
        "dataset_config":    DATASET_CONFIG,
        "source_split":      DATASET_SPLIT,
        "evaluation_role":   "external_retrieval_queries",
        "original_id":       original_id,
        "dataset_revision":  dataset_revision,
        "conversion_version": CONVERSION_VERSION,
    }
    # Preserve original dataset source identifier (e.g. nvidia/Nemotron-RL...)
    if "source" in raw and isinstance(raw["source"], str):
        metadata["original_source"] = raw["source"]
    # Preserve extra source-level provenance fields when present
    for k in ("trajectory_id", "episode_id", "task_id",
              "scenario_family", "template_id", "domain", "family"):
        if k in raw and isinstance(raw[k], str):
            metadata[k] = raw[k]

    out: dict[str, Any] = {
        "group_id": group_id,
        "id":       record_id,
        "kind":     "choice",
        "metadata": metadata,
        "options":  options,
        "question": question,
        "source":   SOURCE_STRING,
        "split":    "validation",   # compatibility alias for R0 --split validation
        "state":    state,
        "target":   target,
    }
    return out

# ---------------------------------------------------------------------------
# Anti-leakage check
# ---------------------------------------------------------------------------

def _check_no_leakage(
    state: Any,
    question: str,
    options: list[str],
    target: list[float],
) -> None:
    """
    Verify that the correct answer index does not appear in model-visible fields.
    This is a structural check, not a semantic one.
    """
    correct_idx = target.index(1.0)

    # State must not embed the correct index as a top-level integer field
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, int) and v == correct_idx:
                # Only flag fields that look like supervision leaks
                suspect_keys = {
                    "answer_index", "correct_index", "target_index",
                    "correct_option", "label", "correct", "answer",
                }
                if k in suspect_keys:
                    raise ValueError(
                        f"potential answer leakage: state.{k}={v} "
                        f"equals correct_idx={correct_idx}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(rec: dict, strict: bool = False) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(isinstance(rec, dict), "not a JSON object")
    check(bool(rec.get("id")),       "id is empty")
    check(bool(rec.get("group_id")), "group_id is empty")
    check(rec.get("kind") == "choice", f"kind != 'choice': {rec.get('kind')!r}")
    check(isinstance(rec.get("state"), (dict, str, list, int, float)),
          "state is not a JSON value")
    check(bool(rec.get("question")), "question is empty")

    options = rec.get("options", [])
    target  = rec.get("target",  [])
    check(isinstance(options, list),              "options is not a list")
    check(isinstance(target,  list),              "target is not a list")
    check(len(options) >= 2,                      f"too few options: {len(options)}")
    check(all(isinstance(o, str) for o in options), "options contains non-string")
    check(all(o for o in options),                "options contains empty string")
    check(len(target) == len(options),
          f"target length {len(target)} != options length {len(options)}")

    if isinstance(target, list) and len(target) > 0:
        ones = sum(1 for v in target if v == 1.0)
        zeros = sum(1 for v in target if v == 0.0)
        check(ones == 1,  f"target has {ones} ones, expected exactly 1")
        check(zeros == len(target) - 1,
              f"target has {zeros} zeros, expected {len(target) - 1}")
        check(all(isinstance(v, float) for v in target),
              "target contains non-float")

    return errors

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class ConversionStats:
    def __init__(self) -> None:
        self.n_read       = 0
        self.n_converted  = 0
        self.n_invalid    = 0
        self.n_skipped    = 0
        self.skip_reasons: dict[str, int] = defaultdict(int)
        self.option_counts: dict[int, int] = defaultdict(int)
        self.state_lengths: list[int] = []
        self.ids: set[str] = set()
        self.duplicate_ids = 0

    def record_converted(self, rec: dict) -> None:
        self.n_converted += 1
        rid = rec["id"]
        if rid in self.ids:
            self.duplicate_ids += 1
        self.ids.add(rid)
        n = len(rec["options"])
        self.option_counts[n] += 1
        # Approximate state text length
        state_str = json.dumps(rec["state"], ensure_ascii=False)
        self.state_lengths.append(len(state_str))

    def record_invalid(self, reason: str) -> None:
        self.n_invalid += 1
        self.skip_reasons[reason] += 1

    def record_skipped(self, reason: str) -> None:
        self.n_skipped += 1
        self.skip_reasons[reason] += 1

    def to_dict(
        self,
        source_dataset: str,
        revision: str,
        output_path: str | None,
        conversion_version: int,
    ) -> dict:
        counts = list(self.option_counts.items())
        counts.sort()
        n_opts = sum(k * v for k, v in counts)
        total  = sum(v for _, v in counts)
        mean_opts = n_opts / total if total else 0

        lens = self.state_lengths
        mean_state = sum(lens) / len(lens) if lens else 0
        max_state  = max(lens, default=0)

        output_size = 0
        if output_path and os.path.exists(output_path):
            output_size = os.path.getsize(output_path)

        return {
            "source_dataset":          source_dataset,
            "dataset_revision":        revision,
            "conversion_version":      conversion_version,
            "records_read":            self.n_read,
            "records_converted":       self.n_converted,
            "records_invalid":         self.n_invalid,
            "records_skipped":         self.n_skipped,
            "unique_ids":              len(self.ids),
            "duplicate_ids":           self.duplicate_ids,
            "skip_reasons":            dict(self.skip_reasons),
            "option_count_distribution": {str(k): v for k, v in counts},
            "option_count_mean":       round(mean_opts, 2),
            "option_count_max":        max(k for k, _ in counts) if counts else 0,
            "state_length_mean_chars": round(mean_state, 1),
            "state_length_max_chars":  max_state,
            "output_jsonl_size_bytes": output_size,
        }

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_from_huggingface(revision: str | None) -> tuple[Any, str]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. "
              "Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    kwargs: dict[str, Any] = {
        "path":   DATASET_REPO,
        "name":   DATASET_CONFIG,
        "split":  DATASET_SPLIT,
    }
    if revision:
        kwargs["revision"] = revision

    print(f"Loading {DATASET_REPO} ({DATASET_CONFIG}) split={DATASET_SPLIT} ...",
          file=sys.stderr)
    ds = load_dataset(**kwargs)

    # Get revision from dataset info if not provided
    resolved_rev = revision or ""
    try:
        if hasattr(ds, "info") and hasattr(ds.info, "download_checksums"):
            # best-effort; may not always be available
            pass
        if hasattr(ds, "_info"):
            info = ds._info
            if hasattr(info, "version"):
                resolved_rev = str(info.version) or resolved_rev
    except Exception:
        pass

    return ds, resolved_rev

def load_from_parquet(path: str) -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        print("ERROR: 'pyarrow' package not installed. "
              "Run: pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    print(f"Loading from local Parquet: {path}", file=sys.stderr)
    table = pq.read_table(path)
    return table.to_pylist()

# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert(
    records: Any,
    output_path: str,
    stats_path: str,
    limit: int,
    validate_only: bool,
    strict: bool,
    verbose: bool,
    dataset_revision: str,
) -> int:
    stats = ConversionStats()

    # Discover schema from first record
    mapping: dict[str, str] = {}
    first_record: dict | None = None

    iter_records = iter(records)
    try:
        first_record = dict(next(iter_records))
    except StopIteration:
        print("ERROR: dataset is empty", file=sys.stderr)
        return 1

    mapping = discover_schema(first_record, verbose=True)
    print(f"Discovered field mapping: {mapping}", file=sys.stderr)

    # Reattach first record
    import itertools
    all_records = itertools.chain([first_record], iter_records)

    out_f = None
    if not validate_only:
        out_f = open(output_path, "w", encoding="utf-8", newline="\n")

    try:
        for raw in all_records:
            if limit > 0 and stats.n_read >= limit:
                break

            stats.n_read += 1
            raw = dict(raw)

            try:
                rec = convert_record(
                    raw, mapping, dataset_revision, verbose=verbose)
            except ValueError as e:
                reason = str(e)
                if verbose:
                    print(f"  [skip] convert error on record {stats.n_read}: {reason}",
                          file=sys.stderr)
                stats.record_invalid(reason[:80])
                if strict:
                    print(f"STRICT MODE: aborting on first invalid record: {reason}",
                          file=sys.stderr)
                    return 1
                continue

            errors = validate_record(rec, strict=strict)
            if errors:
                reason = "; ".join(errors)
                if verbose:
                    print(f"  [invalid] record {stats.n_read}: {reason}",
                          file=sys.stderr)
                stats.record_invalid(reason[:80])
                if strict:
                    print(f"STRICT MODE: validation failed: {reason}",
                          file=sys.stderr)
                    return 1
                continue

            stats.record_converted(rec)

            if not validate_only and out_f is not None:
                line = json.dumps(rec, ensure_ascii=False, sort_keys=False,
                                  separators=(",", ":"))
                out_f.write(line + "\n")

            if verbose and stats.n_converted % 1000 == 0:
                print(f"  converted {stats.n_converted} records ...",
                      file=sys.stderr)

    finally:
        if out_f is not None:
            out_f.close()

    # Verify output line count
    if not validate_only and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines != stats.n_converted:
            print(f"WARNING: output line count {n_lines} != converted count "
                  f"{stats.n_converted}", file=sys.stderr)

    # Print summary
    print(f"\nRecords read:      {stats.n_read}", file=sys.stderr)
    print(f"Converted:         {stats.n_converted}", file=sys.stderr)
    print(f"Invalid/skipped:   {stats.n_invalid}", file=sys.stderr)
    if stats.duplicate_ids:
        print(f"WARNING: {stats.duplicate_ids} duplicate IDs", file=sys.stderr)
    if stats.skip_reasons:
        print("Skip reasons:", file=sys.stderr)
        for reason, count in sorted(stats.skip_reasons.items(),
                                    key=lambda x: -x[1]):
            print(f"  {count:6d}  {reason}", file=sys.stderr)

    # Write stats
    stats_dict = stats.to_dict(
        source_dataset=f"{DATASET_REPO}/{DATASET_CONFIG}",
        revision=dataset_revision,
        output_path=output_path if not validate_only else None,
        conversion_version=CONVERSION_VERSION,
    )
    stats_dir = os.path.dirname(stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Stats written to: {stats_path}", file=sys.stderr)

    return 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert samatv256/jev-decisions-v1 (general-clean-50k) "
            "to Open Jev R0 JSONL.\n\n"
            "R0 External Corpus Retrieval Sanity Check."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", metavar="PATH",
                   default="datasets/clean50k-r0.jsonl",
                   help="Output JSONL path (default: datasets/clean50k-r0.jsonl)")
    p.add_argument("--limit", type=int, default=0,
                   metavar="N",
                   help="Process at most N records (0 = all)")
    p.add_argument("--revision", metavar="REVISION", default=None,
                   help="Hugging Face dataset revision/commit hash")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate without writing output")
    p.add_argument("--strict", action="store_true",
                   help="Stop on first invalid record")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-record progress")
    p.add_argument("--input-parquet", metavar="PATH", default=None,
                   help="Load from local Parquet file instead of HuggingFace")
    return p

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output
    stats_path  = os.path.splitext(output_path)[0] + ".stats.json"
    # Override with canonical stats path
    stats_dir   = os.path.dirname(output_path) or "datasets"
    stats_path  = os.path.join(stats_dir, "clean50k-r0.stats.json")

    dataset_revision = args.revision or "unknown"

    # Load data
    if args.input_parquet:
        records = load_from_parquet(args.input_parquet)
        if not dataset_revision or dataset_revision == "unknown":
            dataset_revision = f"local:{os.path.basename(args.input_parquet)}"
    else:
        records, resolved_rev = load_from_huggingface(args.revision)
        if args.revision:
            # User-supplied revision always takes precedence
            dataset_revision = args.revision
        elif resolved_rev and resolved_rev not in ("unknown", "0.0.0", ""):
            dataset_revision = resolved_rev

    t0 = time.monotonic()
    ret = convert(
        records      = records,
        output_path  = output_path,
        stats_path   = stats_path,
        limit        = args.limit,
        validate_only= args.validate_only,
        strict       = args.strict,
        verbose      = args.verbose,
        dataset_revision = dataset_revision,
    )
    elapsed = time.monotonic() - t0
    print(f"Elapsed: {elapsed:.1f}s", file=sys.stderr)

    return ret

if __name__ == "__main__":
    sys.exit(main())
