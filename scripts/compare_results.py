# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Compare stored `pjev run` results and generate a Markdown (and optional JSON) report.

Aggregate metrics are read from each result's `metrics.<primitive>` block, so their
definitions stay in pjev (see doc/metrics.md). Item-level comparisons (margin gain,
correctness flips, largest margin changes, target-vs-target equivalence) are computed from
`noul_items` / `choice_items`. `--verify` recomputes the aggregates from the items
independently and reports any disagreement with the stored values.

Examples (paths may contain `{split}`):

  uv run scripts/compare_results.py --primitive noul \\
      --baseline results/measure/base_{split}.json \\
      --target E1=results/measure/e1_{split}.json \\
      --output results/measure/report_noul.md --verify

  uv run scripts/compare_results.py --primitive choice \\
      --baseline results/measure/e2/base_{split}.json \\
      --target E2=results/measure/e2/e2_{split}.json \\
      --target E2+KV=results/measure/e2/e2kv_{split}.json \\
      --equivalence E2,E2+KV --output results/measure/e2/report_choice.md --verify
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

GAIN_TOL = 1e-9     # |gain| below this is "unchanged" (same as pjev MARGIN_GAIN_TOL)
VERIFY_TOL = 1e-6   # tolerance when checking recomputed aggregates against stored values

# (label, path in metrics.<primitive>, format, better: "lower" | "higher" | None)
Row = tuple[str, str, str, Optional[str]]

NOUL_ROWS: list[Row] = [
    ("labeled", "labeled", "int", None),
    ("accuracy", "accuracy", "f4", "higher"),
    ("NLL", "nll", "f4", "lower"),
    ("Brier", "brier", "f4", "lower"),
    ("ECE", "ece", "f4", "lower"),
    ("NLL (raw, T=1)", "raw.nll", "f4", "lower"),
    ("Brier (raw, T=1)", "raw.brier", "f4", "lower"),
    ("ECE (raw, T=1)", "raw.ece", "f4", "lower"),
    ("mean conf. correct (raw)", "raw.mean_confidence_correct", "f4", "higher"),
    ("mean signed margin", "mean_signed_margin", "f4", "higher"),
    ("median signed margin", "median_signed_margin", "f4", "higher"),
    ("p10 signed margin", "p10_signed_margin", "f4", "higher"),
    ("min signed margin", "min_signed_margin", "f4", "higher"),
    ("order disagreement rate", "order_disagreement_rate", "f4", "lower"),
    ("eval_ms", "eval_ms", "int", "lower"),
]

CHOICE_ROWS: list[Row] = [
    ("labeled", "labeled", "int", None),
    ("accuracy", "accuracy", "f4", "higher"),
    ("NLL", "nll", "f4", "lower"),
    ("Brier", "brier", "f4", "lower"),
    ("ECE", "ece", "f4", "lower"),
    ("NLL (raw, T=1)", "raw.nll", "f4", "lower"),
    ("Brier (raw, T=1)", "raw.brier", "f4", "lower"),
    ("ECE (raw, T=1)", "raw.ece", "f4", "lower"),
    ("mean winner margin", "mean_signed_winner_margin", "f4", "higher"),
    ("median winner margin", "median_signed_winner_margin", "f4", "higher"),
    ("p10 winner margin", "p10_signed_winner_margin", "f4", "higher"),
    ("min winner margin", "min_signed_winner_margin", "f4", "higher"),
    ("rotation disagreement rate", "rotation_disagreement_rate", "f4", "lower"),
    ("mean distinct winners", "mean_number_of_distinct_winners", "f4", "lower"),
    ("mean semantic variance", "mean_semantic_logit_variance", "f4", "lower"),
    ("max semantic variance", "max_semantic_logit_variance", "f4", "lower"),
    ("evaluations per item", "rotations_per_item", "f2", "lower"),
    ("prompt tokens", "prompt_tokens", "int", "lower"),
    ("eval_ms", "eval_ms", "int", "lower"),
]

ROWS = {"noul": NOUL_ROWS, "choice": CHOICE_ROWS}

# Metrics that summarize quality (used for the improved/worse tally; cost rows excluded).
QUALITY_KEYS = {
    "noul": ["accuracy", "nll", "brier", "ece", "raw.nll", "raw.brier", "raw.ece",
             "mean_signed_margin", "median_signed_margin", "p10_signed_margin", "min_signed_margin"],
    "choice": ["accuracy", "nll", "brier", "ece", "raw.nll", "raw.brier", "raw.ece",
               "mean_signed_winner_margin", "median_signed_winner_margin",
               "p10_signed_winner_margin", "min_signed_winner_margin"],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    label: str
    path: Path
    data: dict[str, Any]
    primitive: str

    @property
    def metrics(self) -> dict[str, Any]:
        return self.data.get("metrics", {}).get(self.primitive, {})

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.data.get(f"{self.primitive}_items", [])

    def items_by_id(self) -> dict[str, dict[str, Any]]:
        return {it["source_id"]: it for it in self.items if it.get("source_id")}


def load_run(label: str, path: Path, primitive: str) -> Run:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if f"{primitive}_items" not in data:
        raise SystemExit(f"{path}: no '{primitive}_items' (re-run 'pjev run' with a current build)")
    return Run(label, path, data, primitive)


def get_path(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---------------------------------------------------------------------------
# Statistics (mirrors src/experiment/noul_metrics.cpp)
# ---------------------------------------------------------------------------

def percentile_linear(values: list[float], q: float) -> Optional[float]:
    """Hyndman & Fan type 7 (numpy default): linear interpolation between closest ranks."""
    if not values:
        return None
    v = sorted(values)
    h = (len(v) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def margin_summary(values: list[float]) -> dict[str, Optional[float]]:
    vals = [x for x in values if is_num(x)]
    return {
        "mean": mean(vals),
        "median": percentile_linear(vals, 0.5),
        "p10": percentile_linear(vals, 0.1),
        "min": min(vals) if vals else None,
    }


# ---------------------------------------------------------------------------
# Item-level comparisons
# ---------------------------------------------------------------------------

@dataclass
class Gain:
    n_matched: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    improved: int = 0
    degraded: int = 0
    unchanged: int = 0
    correct_to_wrong: list[str] = field(default_factory=list)
    wrong_to_correct: list[str] = field(default_factory=list)
    changes: list[tuple[str, float, float, float]] = field(default_factory=list)  # id, base, target, gain

    def to_json(self) -> dict[str, Any]:
        return {
            "n_matched": self.n_matched,
            "mean_margin_gain": self.mean,
            "median_margin_gain": self.median,
            "improved_margin_count": self.improved,
            "degraded_margin_count": self.degraded,
            "unchanged_margin_count": self.unchanged,
            "correct_to_wrong_count": len(self.correct_to_wrong),
            "wrong_to_correct_count": len(self.wrong_to_correct),
            "correct_to_wrong": self.correct_to_wrong,
            "wrong_to_correct": self.wrong_to_correct,
        }


def margin_gain(base: Run, target: Run) -> Gain:
    """Matched by source_id over labeled items present in both runs."""
    g = Gain()
    b_items = base.items_by_id()
    gains: list[float] = []
    for t in target.items:
        sid = t.get("source_id")
        b = b_items.get(sid)
        if b is None or t.get("correct") is None or b.get("correct") is None:
            continue
        bm, tm = b.get("signed_margin"), t.get("signed_margin")
        if not (is_num(bm) and is_num(tm)):
            continue
        d = tm - bm
        gains.append(d)
        g.changes.append((sid, bm, tm, d))
        if abs(d) < GAIN_TOL:
            g.unchanged += 1
        elif d > 0:
            g.improved += 1
        else:
            g.degraded += 1
        if b["correct"] and not t["correct"]:
            g.correct_to_wrong.append(sid)
        elif not b["correct"] and t["correct"]:
            g.wrong_to_correct.append(sid)
    g.n_matched = len(gains)
    g.mean = mean(gains)
    g.median = percentile_linear(gains, 0.5)
    return g


def item_logits(primitive: str, it: dict[str, Any]) -> list[float]:
    if primitive == "choice":
        return list(it.get("logits") or [])
    m = it.get("semantic_margin")
    return [m] if is_num(m) else []


def equivalence(primitive: str, a: Run, b: Run) -> dict[str, Any]:
    """Item-by-item agreement of two runs that should be numerically equivalent."""
    a_items, b_items = a.items_by_id(), b.items_by_id()
    common = sorted(set(a_items) & set(b_items))
    pred_mismatch: list[str] = []
    max_logit = 0.0
    max_prob = 0.0
    for sid in common:
        x, y = a_items[sid], b_items[sid]
        if x.get("prediction") != y.get("prediction"):
            pred_mismatch.append(sid)
        lx, ly = item_logits(primitive, x), item_logits(primitive, y)
        for p, q in zip(lx, ly):
            max_logit = max(max_logit, abs(p - q))
        px = x.get("probs") if primitive == "choice" else [x.get("p_true")]
        py = y.get("probs") if primitive == "choice" else [y.get("p_true")]
        for p, q in zip(px or [], py or []):
            if is_num(p) and is_num(q):
                max_prob = max(max_prob, abs(p - q))
    return {
        "a": a.label,
        "b": b.label,
        "n_common": len(common),
        "prediction_mismatches": pred_mismatch,
        "max_abs_logit_diff": max_logit,
        "max_abs_prob_diff": max_prob,
    }


# ---------------------------------------------------------------------------
# Independent recomputation (--verify)
# ---------------------------------------------------------------------------

def softmax(logits: list[float], t: float = 1.0) -> list[float]:
    m = max(x / t for x in logits)
    e = [math.exp(x / t - m) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def ece(conf: list[float], correct: list[bool], n_bins: int) -> Optional[float]:
    if not conf:
        return None
    cnt = [0] * n_bins
    csum = [0.0] * n_bins
    acc = [0] * n_bins
    for c, ok in zip(conf, correct):
        b = min(max(int(c * n_bins), 0), n_bins - 1)
        cnt[b] += 1
        csum[b] += c
        acc[b] += int(ok)
    n = len(conf)
    return sum(cnt[b] / n * abs(csum[b] / cnt[b] - acc[b] / cnt[b]) for b in range(n_bins) if cnt[b])


def recompute_noul(items: list[dict[str, Any]], n_bins: int) -> dict[str, Any]:
    eps = 1e-12
    lab = [it for it in items if it.get("ground_truth") is not None]
    out: dict[str, Any] = {"labeled": len(lab)}
    if lab:
        out["accuracy"] = sum(bool(it["correct"]) for it in lab) / len(lab)
    for key, block in (("p_true", ""), ("p_true_raw", "raw.")):
        nll, brier, conf, corr = [], [], [], []
        for it in lab:
            p, y = it.get(key), bool(it["ground_truth"])
            if not is_num(p):
                continue
            pg = min(max(p if y else 1 - p, eps), 1 - eps)
            nll.append(-math.log(pg))
            brier.append((p - (1.0 if y else 0.0)) ** 2)
            conf.append(p if it["prediction"] else 1 - p)
            corr.append(it["prediction"] == y)
        out[block + "nll"] = mean(nll)
        out[block + "brier"] = mean(brier)
        out[block + "ece"] = ece(conf, corr, n_bins)
    s = margin_summary([it.get("signed_margin") for it in lab])
    for k, v in s.items():
        out[f"{k}_signed_margin"] = v
    orders = [it for it in items if "order_disagreement" in it]
    if orders:
        out["order_disagreement_rate"] = sum(bool(it["order_disagreement"]) for it in orders) / len(orders)
    return out


def recompute_choice(items: list[dict[str, Any]], n_bins: int) -> dict[str, Any]:
    eps = 1e-15  # compute_primitive_metrics convention
    lab = [it for it in items if it.get("ground_truth_index") is not None and it.get("logits")]
    out: dict[str, Any] = {"labeled": len(lab)}
    if lab:
        out["accuracy"] = sum(bool(it["correct"]) for it in lab) / len(lab)
    temps = {it.get("temperature", 1.0) for it in lab}
    t_run = next(iter(temps)) if len(temps) == 1 else 1.0
    for t, block in ((t_run, ""), (1.0, "raw.")):
        nll, brier, conf, corr = [], [], [], []
        for it in lab:
            gt = it["ground_truth_index"]
            p = softmax(it["logits"], t)
            nll.append(-math.log(max(p[gt], eps)))
            brier.append(sum((pi - (1.0 if i == gt else 0.0)) ** 2 for i, pi in enumerate(p)))
            pred = max(range(len(p)), key=lambda i: p[i])
            conf.append(p[pred])
            corr.append(pred == gt)
        out[block + "nll"] = mean(nll)
        out[block + "brier"] = mean(brier)
        out[block + "ece"] = ece(conf, corr, n_bins)
    s = margin_summary([it.get("signed_margin") for it in lab])
    for k, v in s.items():
        out[f"{k}_signed_winner_margin"] = v
    rot = [it for it in items if it.get("rotations")]
    if rot:
        out["rotation_disagreement_rate"] = sum(bool(it["rotation_disagreement"]) for it in rot) / len(rot)
        out["mean_number_of_distinct_winners"] = mean([it["distinct_winners"] for it in rot])
        out["rotations_per_item"] = mean([it["rotation_count"] for it in rot])
        var = []
        for it in rot:
            k = len(it["keys"])
            per_opt = []
            for i in range(k):
                vals = [r["semantic_logits"][i] for r in it["rotations"]]
                mu = sum(vals) / len(vals)
                per_opt.append(sum((x - mu) ** 2 for x in vals) / len(vals))
            var.append(sum(per_opt) / len(per_opt))
        out["mean_semantic_logit_variance"] = mean(var)
    return out


def verify(run: Run) -> list[str]:
    n_bins = run.metrics.get("ece_bins", 15) or 15
    rec = recompute_noul(run.items, n_bins) if run.primitive == "noul" else recompute_choice(run.items, n_bins)
    problems = []
    for key, val in rec.items():
        stored = get_path(run.metrics, key)
        if val is None and stored is None:
            continue
        if not (is_num(val) and is_num(stored)) or abs(val - stored) > VERIFY_TOL * max(1.0, abs(stored)):
            problems.append(f"{run.label} {key}: stored {stored!r} vs recomputed {val!r}")
    return problems


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v: Any, kind: str) -> str:
    if not is_num(v):
        return "n/a"
    if kind == "int":
        return str(int(v))
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.4f}"


def fmt_delta(base: Any, val: Any, kind: str, better: Optional[str]) -> str:
    if not (is_num(base) and is_num(val)):
        return ""
    d = val - base
    if kind == "int":
        s = f"{int(d):+d}"
    else:
        s = f"{d:+.4f}"
    if better and abs(d) > GAIN_TOL:
        good = (d < 0) if better == "lower" else (d > 0)
        s += " ▲" if good else " ▼"
    return s


def tally(primitive: str, base: Run, target: Run) -> tuple[list[str], list[str]]:
    better_of = {path: better for _, path, _, better in ROWS[primitive]}
    improved, worse = [], []
    for key in QUALITY_KEYS[primitive]:
        b, t = get_path(base.metrics, key), get_path(target.metrics, key)
        if not (is_num(b) and is_num(t)) or abs(t - b) <= GAIN_TOL:
            continue
        good = (t < b) if better_of.get(key) == "lower" else (t > b)
        (improved if good else worse).append(key)
    return improved, worse


def md_table(header: list[str], rows: list[list[str]], align: Optional[list[str]] = None) -> str:
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def split_report(primitive: str, split: str, base: Run, targets: list[Run],
                 top_n: int, equiv_pairs: list[tuple[str, str]],
                 verify_problems: Optional[list[str]]) -> tuple[str, dict[str, Any]]:
    out: list[str] = [f"## {primitive} / {split}", ""]
    src = [f"- baseline: `{base.path.as_posix()}`"] + [f"- {t.label}: `{t.path.as_posix()}`" for t in targets]
    out += src + [""]

    header = ["metric", base.label]
    for t in targets:
        header += [t.label, f"Δ {t.label}"]
    rows = []
    for label, path, kind, better in ROWS[primitive]:
        bv = get_path(base.metrics, path)
        tvs = [get_path(t.metrics, path) for t in targets]
        if not is_num(bv) and not any(is_num(v) for v in tvs):
            continue
        row = [label, fmt(bv, kind)]
        for tv in tvs:
            row += [fmt(tv, kind), fmt_delta(bv, tv, kind, better)]
        rows.append(row)
    out += [md_table(header, rows), "", "▲ better than baseline, ▼ worse.", ""]

    data: dict[str, Any] = {"baseline": {"label": base.label, "path": base.path.as_posix(), "metrics": base.metrics},
                            "targets": []}
    b_ms = get_path(base.metrics, "eval_ms")
    for t in targets:
        g = margin_gain(base, t)
        improved, worse = tally(primitive, base, t)
        t_ms = get_path(t.metrics, "eval_ms")
        ratio = (t_ms / b_ms) if is_num(t_ms) and is_num(b_ms) and b_ms > 0 else None
        out += [f"### {t.label} vs {base.label}", ""]
        out += [
            f"- matched items: {g.n_matched}; margin gain mean {fmt(g.mean, 'f4')}, median {fmt(g.median, 'f4')}",
            f"- margin improved / degraded / unchanged: {g.improved} / {g.degraded} / {g.unchanged}",
            f"- {base.label} correct → {t.label} wrong: {len(g.correct_to_wrong)}"
            + (f" ({', '.join(g.correct_to_wrong)})" if g.correct_to_wrong else ""),
            f"- {base.label} wrong → {t.label} correct: {len(g.wrong_to_correct)}"
            + (f" ({', '.join(g.wrong_to_correct)})" if g.wrong_to_correct else ""),
            f"- quality metrics better: {len(improved)}, worse: {len(worse)}"
            + (f" (worse: {', '.join(worse)})" if worse else ""),
            f"- latency: {fmt(t_ms, 'int')} ms vs {fmt(b_ms, 'int')} ms"
            + (f" (×{ratio:.2f})" if ratio is not None else ""),
            "",
        ]
        if top_n > 0 and g.changes:
            ranked = sorted(g.changes, key=lambda c: c[3])
            worst = [c for c in ranked if c[3] < -GAIN_TOL][:top_n]
            best = [c for c in reversed(ranked) if c[3] > GAIN_TOL][:top_n]
            for title, sel in ((f"Largest margin losses (top {top_n})", worst),
                               (f"Largest margin gains (top {top_n})", best)):
                if not sel:
                    continue
                out += [f"**{title}**", ""]
                out += [md_table(["source_id", base.label, t.label, "gain"],
                                 [[f"`{sid}`", f"{bm:.4f}", f"{tm:.4f}", f"{d:+.4f}"] for sid, bm, tm, d in sel]), ""]
        data["targets"].append({"label": t.label, "path": t.path.as_posix(), "metrics": t.metrics,
                                "margin_gain": g.to_json(), "quality_better": improved, "quality_worse": worse,
                                "latency_ratio": ratio})

    by_label = {r.label: r for r in [base, *targets]}
    data["equivalence"] = []
    for a_label, b_label in equiv_pairs:
        if a_label not in by_label or b_label not in by_label:
            continue
        e = equivalence(primitive, by_label[a_label], by_label[b_label])
        data["equivalence"].append(e)
        out += [f"### Equivalence: {a_label} vs {b_label}", "",
                f"- common items: {e['n_common']}",
                f"- prediction mismatches: {len(e['prediction_mismatches'])}"
                + (f" ({', '.join(e['prediction_mismatches'])})" if e["prediction_mismatches"] else ""),
                f"- max |Δ logit|: {e['max_abs_logit_diff']:.3g}",
                f"- max |Δ prob|: {e['max_abs_prob_diff']:.3g}", ""]

    if verify_problems is not None:
        data["verify_problems"] = verify_problems
        out += ["### Verification", ""]
        if verify_problems:
            out += ["Recomputed aggregates disagree with stored metrics:", ""] + [f"- {p}" for p in verify_problems]
        else:
            out += ["All stored aggregates match an independent recomputation from the items."]
        out += [""]
    return "\n".join(out), data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_target(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--target expects LABEL=PATH, got {s!r}")
    label, path = s.split("=", 1)
    return label, path


def parse_pair(s: str) -> tuple[str, str]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--equivalence expects A,B, got {s!r}")
    return parts[0], parts[1]


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primitive", choices=["noul", "choice"], required=True)
    ap.add_argument("--baseline", required=True, help="baseline result JSON (may contain {split})")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--target", action="append", type=parse_target, required=True,
                    help="LABEL=PATH (repeatable; PATH may contain {split})")
    ap.add_argument("--splits", nargs="+", default=["easy", "original", "hard"],
                    help="values substituted for {split} (default: easy original hard)")
    ap.add_argument("--equivalence", action="append", type=parse_pair, default=[],
                    help="A,B: compare two run labels item by item (e.g. E2,E2+KV)")
    ap.add_argument("--top", type=int, default=5, help="largest margin changes to list per target (0 = none)")
    ap.add_argument("--verify", action="store_true", help="recompute aggregates from items and check stored values")
    ap.add_argument("--title", default=None)
    ap.add_argument("--output", type=Path, help="Markdown report path (default: stdout)")
    ap.add_argument("--json", type=Path, help="also write the report data as JSON")
    args = ap.parse_args(argv)

    uses_split = "{split}" in args.baseline or any("{split}" in p for _, p in args.target)
    splits = args.splits if uses_split else ["-"]

    title = args.title or f"{args.primitive}: {args.baseline_label} vs " + ", ".join(l for l, _ in args.target)
    sections = [f"# {title}", ""]
    report: dict[str, Any] = {"primitive": args.primitive, "splits": {}}
    any_problem = False

    for split in splits:
        sub: Callable[[str], Path] = lambda p: Path(p.replace("{split}", split))
        base = load_run(args.baseline_label, sub(args.baseline), args.primitive)
        targets = [load_run(label, sub(p), args.primitive) for label, p in args.target]
        problems = None
        if args.verify:
            problems = [p for r in [base, *targets] for p in verify(r)]
            any_problem = any_problem or bool(problems)
        text, data = split_report(args.primitive, split, base, targets, args.top, args.equivalence, problems)
        sections.append(text)
        report["splits"][split] = data

    md = "\n".join(sections).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    if any_problem:
        print("verification: stored metrics disagree with recomputation (see report)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
