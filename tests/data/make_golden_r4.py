#!/usr/bin/env python3
"""Build the R4 golden regression set (tests/data/golden_r4.jsonl).

The expected values are the measurements of the frozen R4 policy B_task_only@A_latency on the R4
held-out evaluation partition, recorded before the release-candidate implementation existed:

  results/r4/evaluation/queries.jsonl   per-query neighbours, 1-shot and 2-shot logits (pjev-r4 run)
  results/r4/routing/predictions.jsonl  per-policy predictions (pjev-r4 evaluate)
  datasets/validation.jsonl             task content (question, state, options, domain, family)

These inputs live on the experimental branch (features/r3, commit 96a19a5); the generated file is
checked in so the release branch does not need them. Selection is deterministic (sorted by query
hash). Usage, from a checkout of the experimental branch:

  python -I tests/data/make_golden_r4.py --root . --out tests/data/golden_r4.jsonl
"""
import argparse
import hashlib
import json
import math
import os

POLICY = "B_task_only@A_latency"
TAU = 0.005
K = 5
PER_KIND = 8


def state_text(state):
    # r1::state_text: strings verbatim, structured states as compact JSON (key order preserved).
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def vote(neighbours, k, tau):
    # r4::vote, weighted: w = exp((s - s_max) / tau); ties -> label of the best-ranked neighbour.
    nb = [n for n in neighbours[:k] if n["label"] >= 0]
    if not nb:
        return -1, {}
    s_max = max(n["estimated_cosine"] for n in nb)
    weight, first = {}, {}
    for i, n in enumerate(nb):
        w = math.exp((n["estimated_cosine"] - s_max) / tau)
        weight[n["label"]] = weight.get(n["label"], 0.0) + w
        first.setdefault(n["label"], i)
    best = None
    for label in sorted(weight):
        if best is None:
            best = label
            continue
        tol = 1e-12 * max(weight[label], weight[best])
        if weight[label] > weight[best] + tol or (abs(weight[label] - weight[best]) <= tol and first[label] < first[best]):
            best = label
    return best, weight


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="tests/data/golden_r4.jsonl")
    a = ap.parse_args()
    p = lambda rel: os.path.join(a.root, rel)
    queries_path = p("results/r4/evaluation/queries.jsonl")
    routing_path = p("results/r4/routing/predictions.jsonl")
    dataset_path = p("datasets/validation.jsonl")

    routed = {}
    with open(routing_path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["strategy"] == POLICY:
                routed[j["query_id"]] = j
    measured = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            measured[j["query_id"]] = j
    tasks = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["id"] in measured:
                tasks[j["id"]] = j

    def tags(q):
        t = []
        rt = q["retrieval"]
        if not rt["topk_rank1_matches_r1"]:
            t.append("topk_rank1_differs_from_r1")
        if q["kind"] == "noul" and q["r1"]["prediction_index"] != rt["neighbours"][0]["label"]:
            t.append("llm_departs_from_nearest_label")
        if q["kind"] == "score":
            labels = [n["label"] for n in rt["neighbours"][:K]]
            uniform = max(set(labels), key=lambda l: (labels.count(l), -labels.index(l)))
            if uniform != routed[q["query_id"]]["prediction_index"]:
                t.append("weighted_vote_differs_from_uniform")
            if labels[0] != routed[q["query_id"]]["prediction_index"]:
                t.append("vote_differs_from_nearest_label")
            t.append("levels_%d" % q["number_of_levels"])
        if q["kind"] == "noul" and routed[q["query_id"]]["prediction_index"] == 1:
            t.append("predicts_true")
        if isinstance(tasks[q["query_id"]]["state"], str):
            t.append("string_state")
        return t

    by_kind = {"choice": [], "noul": [], "score": []}
    for qid in sorted(measured, key=lambda i: measured[i]["query_hash"]):
        by_kind[measured[qid]["kind"]].append(qid)

    # Per kind: first the special cases (one or two of each tag), then fill in hash order with
    # distinct scenario families so the set covers several task families.
    wanted = {
        "choice": [("topk_rank1_differs_from_r1", 2), ("string_state", 1)],
        "noul": [("llm_departs_from_nearest_label", 2), ("predicts_true", 3), ("topk_rank1_differs_from_r1", 1),
                 ("string_state", 1)],
        "score": [("levels_3", 2), ("weighted_vote_differs_from_uniform", 1),
                  ("vote_differs_from_nearest_label", 1), ("topk_rank1_differs_from_r1", 1)],
    }
    selected = []
    for kind, ids in by_kind.items():
        picked = []
        for tag, n in wanted[kind]:
            for qid in ids:
                if len([x for x in picked if tag in tags(measured[x])]) >= n:
                    break
                if qid not in picked and tag in tags(measured[qid]):
                    picked.append(qid)
        families = {tasks[x]["metadata"].get("scenario_family") for x in picked}
        for distinct in (True, False):
            for qid in ids:
                if len(picked) >= PER_KIND:
                    break
                fam = tasks[qid]["metadata"].get("scenario_family")
                if qid not in picked and (not distinct or fam not in families):
                    picked.append(qid)
                    families.add(fam)
        selected += picked

    out = []
    numbering = {}
    for qid in selected:
        q, t, r = measured[qid], tasks[qid], routed[qid]
        kind = q["kind"]
        if kind == "choice":
            options = [[o, o] for o in t["options"]]
        elif kind == "score":
            options = [[str(i), o] for i, o in enumerate(t["options"])]
        else:
            assert t["options"] == ["no", "yes"], qid
            options = [["false", ""], ["true", ""]]
        nb = q["retrieval"]["neighbours"]
        exp = {"strategy": r["terminal_action"], "prediction_index": r["prediction_index"],
               "prediction_key": options[r["prediction_index"]][0]}
        if kind == "choice":
            assert r["terminal_action"] == "nearest_2shot_a"
            d = q["two_shot_a"]
            assert d["prediction_index"] == r["prediction_index"]
            exp.update(strategy="nearest_2shot", demonstration_ids=d["example_ids"], logits=d["logits"],
                       prompt_tokens=d["prompt_tokens"])
        elif kind == "noul":
            assert r["terminal_action"] == "nearest_1shot"
            d = q["r1"]
            assert d["prediction_index"] == r["prediction_index"]
            exp.update(demonstration_ids=d["example_ids"], logits=d["logits"], prompt_tokens=d["prompt_tokens"])
        else:
            assert r["terminal_action"] == "knn_5"
            label, weight = vote(nb, K, TAU)
            assert label == r["prediction_index"], qid
            total = sum(weight.values())
            exp.update(neighbour_ids=[x["example_id"] for x in nb[:K]],
                       neighbour_labels=[x["label"] for x in nb[:K]],
                       neighbour_hamming=[x["hamming_distance"] for x in nb[:K]],
                       vote_shares=[weight.get(i, 0.0) / total for i in range(len(options))])
        numbering[kind] = numbering.get(kind, 0) + 1
        out.append({
            "case": "%s-%02d" % (kind, numbering[kind]),
            "query_id": qid,
            "tags": tags(q),
            "ground_truth_index": q["ground_truth_index"],
            "input": {"type": kind, "state": state_text(t["state"]), "question": t["question"],
                      "domain": t["metadata"].get("domain", ""), "family": t["metadata"].get("family", ""),
                      "options": options},
            "expected": exp,
        })

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("wrote %d cases to %s" % (len(out), a.out))
    for rel, path in (("results/r4/evaluation/queries.jsonl", queries_path),
                      ("results/r4/routing/predictions.jsonl", routing_path),
                      ("datasets/validation.jsonl", dataset_path)):
        print("  %s sha256 %s" % (rel, sha256(path)))


if __name__ == "__main__":
    main()
