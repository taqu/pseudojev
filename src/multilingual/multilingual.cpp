#include "multilingual.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace pjev
{

std::string error_transition_str(ErrorTransition t)
{
    switch (t) {
    case ErrorTransition::BOTH_CORRECT:       return "both_correct";
    case ErrorTransition::REF_ONLY_CORRECT:   return "ref_only_correct";
    case ErrorTransition::TARGET_ONLY_CORRECT: return "target_only_correct";
    default:                                   return "both_wrong";
    }
}

static nlohmann::json cal_delta_json(const PrimitiveCalibrationMetrics& ref,
                                      const PrimitiveCalibrationMetrics& tgt)
{
    auto d = [](double a, double b) { return (a >= 0 && b >= 0) ? b - a : -1.0; };
    return {
        {"nll",      d(ref.nll,      tgt.nll)},
        {"brier",    d(ref.brier,    tgt.brier)},
        {"ece",      d(ref.ece,      tgt.ece)},
        {"accuracy", d(ref.accuracy, tgt.accuracy)},
        {"mae",      d(ref.mae,      tgt.mae)}
    };
}

nlohmann::json PairedPrimitiveStats::to_json() const
{
    return {
        {"n",                       n},
        {"both_correct",            both_correct},
        {"ref_only_correct",        ref_only_correct},
        {"target_only_correct",     target_only_correct},
        {"both_wrong",              both_wrong},
        {"n_consistent",            n_consistent},
        {"ref_accuracy",            ref_accuracy},
        {"target_accuracy",         target_accuracy},
        {"semantic_consistency_rate", semantic_consistency_rate},
        {"mean_conf_shift",         mean_conf_shift},
        {"mean_target_prob_shift",  mean_target_prob_shift},
        {"ref_calibration",         ref_cal.to_json()},
        {"target_calibration",      target_cal.to_json()},
        {"delta",                   cal_delta_json(ref_cal, target_cal)},
        {"tokens", {
            {"ref_mean",    ref_mean_tokens},
            {"target_mean", target_mean_tokens},
            {"delta",       target_mean_tokens - ref_mean_tokens}
        }},
        {"eval_ms", {
            {"ref_mean",    ref_mean_eval_ms},
            {"target_mean", target_mean_eval_ms},
            {"delta",       target_mean_eval_ms - ref_mean_eval_ms}
        }}
    };
}

nlohmann::json MultilingualResult::to_json() const
{
    nlohmann::json recs = nlohmann::json::array();
    for (const auto& r : records) {
        recs.push_back({
            {"pair_id",              r.pair_id},
            {"type",                 r.type},
            {"difficulty",           r.difficulty},
            {"ref_correct",          r.ref_correct},
            {"target_correct",       r.target_correct},
            {"semantically_consistent", r.semantically_consistent},
            {"transition",           error_transition_str(r.transition)},
            {"ref_confidence",       r.ref_confidence},
            {"target_confidence",    r.target_confidence},
            {"ref_target_prob",      r.ref_target_prob},
            {"target_target_prob",   r.target_target_prob},
            {"ref_prompt_tokens",    r.ref_prompt_tokens},
            {"target_prompt_tokens", r.target_prompt_tokens},
            {"ref_eval_ms",          r.ref_eval_ms},
            {"target_eval_ms",       r.target_eval_ms}
        });
    }
    return {
        {"reference_language", reference_language},
        {"target_language",    target_language},
        {"model",              model_path},
        {"noul",               noul.to_json()},
        {"choice",             choice.to_json()},
        {"score",              score.to_json()},
        {"records",            recs}
    };
}

// Build map from key → ItemResult* for fast pair lookup
static std::map<std::string, const ItemResult*>
build_lookup(const std::vector<ItemResult>& items)
{
    std::map<std::string, const ItemResult*> m;
    for (const auto& ir : items) {
        const std::string key = ir.pair_id.empty() ? ir.id : ir.pair_id;
        m[key] = &ir;
    }
    return m;
}

// Compute PairedPrimitiveStats from collected records of a given type
static PairedPrimitiveStats compute_stats(
    const std::vector<const PairedRecord*>& recs,
    const std::vector<CalibrationSample>& ref_samples,
    const std::vector<CalibrationSample>& tgt_samples)
{
    PairedPrimitiveStats s;
    s.n = (int)recs.size();
    if (s.n == 0) return s;

    int n_ref_correct = 0, n_tgt_correct = 0;
    double sum_conf_shift = 0.0, sum_prob_shift = 0.0;
    double sum_ref_tokens = 0.0, sum_tgt_tokens = 0.0;
    double sum_ref_ms = 0.0, sum_tgt_ms = 0.0;

    for (const auto* r : recs) {
        if (r->ref_correct)    n_ref_correct++;
        if (r->target_correct) n_tgt_correct++;
        if (r->semantically_consistent) s.n_consistent++;

        switch (r->transition) {
        case ErrorTransition::BOTH_CORRECT:        s.both_correct++;       break;
        case ErrorTransition::REF_ONLY_CORRECT:    s.ref_only_correct++;   break;
        case ErrorTransition::TARGET_ONLY_CORRECT: s.target_only_correct++; break;
        default:                                   s.both_wrong++;         break;
        }

        sum_conf_shift  += r->target_confidence - r->ref_confidence;
        sum_prob_shift  += r->target_target_prob - r->ref_target_prob;
        sum_ref_tokens  += r->ref_prompt_tokens;
        sum_tgt_tokens  += r->target_prompt_tokens;
        sum_ref_ms      += (double)r->ref_eval_ms;
        sum_tgt_ms      += (double)r->target_eval_ms;
    }

    s.ref_accuracy    = (double)n_ref_correct / s.n;
    s.target_accuracy = (double)n_tgt_correct / s.n;
    s.semantic_consistency_rate = (double)s.n_consistent / s.n;
    s.mean_conf_shift        = sum_conf_shift  / s.n;
    s.mean_target_prob_shift = sum_prob_shift  / s.n;
    s.ref_mean_tokens    = sum_ref_tokens / s.n;
    s.target_mean_tokens = sum_tgt_tokens / s.n;
    s.ref_mean_eval_ms   = sum_ref_ms / s.n;
    s.target_mean_eval_ms = sum_tgt_ms / s.n;

    // Calibration metrics if logits were collected
    if (!ref_samples.empty())
        s.ref_cal = compute_primitive_metrics(ref_samples, 1.0);
    if (!tgt_samples.empty())
        s.target_cal = compute_primitive_metrics(tgt_samples, 1.0);

    return s;
}

MultilingualResult compare_runs(const RunResult& reference,
                                 const RunResult& target,
                                 const std::string& ref_lang,
                                 const std::string& target_lang,
                                 const std::string& model_path)
{
    MultilingualResult result;
    result.reference_language = ref_lang;
    result.target_language    = target_lang;
    result.model_path         = model_path;

    auto ref_map = build_lookup(reference.items);

    // Calibration sample collections per primitive
    std::vector<CalibrationSample> ref_noul_s, ref_choice_s, ref_score_s;
    std::vector<CalibrationSample> tgt_noul_s, tgt_choice_s, tgt_score_s;

    // Per-primitive record buckets
    std::vector<const PairedRecord*> noul_recs, choice_recs, score_recs;
    // Hold records in stable storage
    std::vector<PairedRecord> all_records;
    all_records.reserve(target.items.size());

    for (const auto& tgt_ir : target.items) {
        if (!tgt_ir.ok) continue;

        const std::string lookup_key = tgt_ir.pair_id.empty() ? tgt_ir.id : tgt_ir.pair_id;
        auto it = ref_map.find(lookup_key);
        if (it == ref_map.end()) continue;
        const ItemResult& ref_ir = *it->second;
        if (!ref_ir.ok) continue;

        PairedRecord rec;
        rec.pair_id   = lookup_key;
        rec.type      = tgt_ir.type;
        rec.difficulty = tgt_ir.difficulty.empty() ? ref_ir.difficulty : tgt_ir.difficulty;
        rec.ref_correct    = ref_ir.correct;
        rec.target_correct = tgt_ir.correct;

        // Semantic consistency: same semantic key selected
        const std::string ref_key = (ref_ir.selected >= 0 && ref_ir.selected < (int)ref_ir.keys.size())
                                  ? ref_ir.keys[ref_ir.selected] : "";
        const std::string tgt_key = (tgt_ir.selected >= 0 && tgt_ir.selected < (int)tgt_ir.keys.size())
                                  ? tgt_ir.keys[tgt_ir.selected] : "";
        rec.semantically_consistent = (!ref_key.empty() && ref_key == tgt_key);

        // Error transition
        if (rec.ref_correct && rec.target_correct)
            rec.transition = ErrorTransition::BOTH_CORRECT;
        else if (rec.ref_correct && !rec.target_correct)
            rec.transition = ErrorTransition::REF_ONLY_CORRECT;
        else if (!rec.ref_correct && rec.target_correct)
            rec.transition = ErrorTransition::TARGET_ONLY_CORRECT;
        else
            rec.transition = ErrorTransition::BOTH_WRONG;

        // Confidence
        rec.ref_confidence    = (ref_ir.selected >= 0 && ref_ir.selected < (int)ref_ir.probs.size())
                              ? ref_ir.probs[ref_ir.selected] : 0.0;
        rec.target_confidence = (tgt_ir.selected >= 0 && tgt_ir.selected < (int)tgt_ir.probs.size())
                              ? tgt_ir.probs[tgt_ir.selected] : 0.0;
        rec.ref_target_prob = (ref_ir.correct_index >= 0 && ref_ir.correct_index < (int)ref_ir.probs.size())
                            ? ref_ir.probs[ref_ir.correct_index] : 0.0;
        rec.target_target_prob = (tgt_ir.correct_index >= 0 && tgt_ir.correct_index < (int)tgt_ir.probs.size())
                                ? tgt_ir.probs[tgt_ir.correct_index] : 0.0;

        rec.ref_prompt_tokens    = ref_ir.prompt_token_count;
        rec.target_prompt_tokens = tgt_ir.prompt_token_count;
        rec.ref_eval_ms    = ref_ir.eval_ms;
        rec.target_eval_ms = tgt_ir.eval_ms;

        all_records.push_back(rec);

        // Calibration samples
        auto add_sample = [](const ItemResult& ir,
                             std::vector<CalibrationSample>& bucket)
        {
            if (ir.corrected_logits.empty() || ir.correct_index < 0) return;
            CalibrationSample s;
            s.logits        = ir.corrected_logits;
            s.correct_index = ir.correct_index;
            s.type          = ir.type;
            s.expected_score = ir.expected_score;
            bucket.push_back(s);
        };

        if (tgt_ir.type == "noul") {
            add_sample(ref_ir, ref_noul_s); add_sample(tgt_ir, tgt_noul_s);
        } else if (tgt_ir.type == "choice") {
            add_sample(ref_ir, ref_choice_s); add_sample(tgt_ir, tgt_choice_s);
        } else if (tgt_ir.type == "score") {
            add_sample(ref_ir, ref_score_s); add_sample(tgt_ir, tgt_score_s);
        }
    }

    // Move records into result and build per-primitive pointer vectors
    result.records = std::move(all_records);
    for (const auto& rec : result.records) {
        if      (rec.type == "noul")   noul_recs.push_back(&rec);
        else if (rec.type == "choice") choice_recs.push_back(&rec);
        else if (rec.type == "score")  score_recs.push_back(&rec);
    }

    result.noul   = compute_stats(noul_recs,   ref_noul_s,   tgt_noul_s);
    result.choice = compute_stats(choice_recs, ref_choice_s, tgt_choice_s);
    result.score  = compute_stats(score_recs,  ref_score_s,  tgt_score_s);

    return result;
}

std::vector<std::string> validate_pairs(
    const std::vector<DatasetRow>& ref_rows,
    const std::vector<DatasetRow>& target_rows)
{
    std::vector<std::string> violations;

    // Build ref map: pair_id or id → row
    std::map<std::string, const DatasetRow*> ref_map;
    for (const auto& r : ref_rows) {
        const std::string k = r.pair_id.empty() ? r.id : r.pair_id;
        ref_map[k] = &r;
    }

    for (const auto& tgt : target_rows) {
        const std::string k = tgt.pair_id.empty() ? tgt.id : tgt.pair_id;
        auto it = ref_map.find(k);
        if (it == ref_map.end()) continue; // unpaired — not a violation

        const DatasetRow& ref = *it->second;

        if (ref.input.type != tgt.input.type)
            violations.push_back("pair " + k + ": type mismatch (" +
                                  ref.input.type + " vs " + tgt.input.type + ")");

        if (ref.input.options.size() != tgt.input.options.size())
            violations.push_back("pair " + k + ": option count mismatch (" +
                                  std::to_string(ref.input.options.size()) + " vs " +
                                  std::to_string(tgt.input.options.size()) + ")");

        // Verify expected labels match (same semantics)
        if (!ref.expected.is_null() && !tgt.expected.is_null() &&
            ref.expected != tgt.expected)
            violations.push_back("pair " + k + ": expected label mismatch");
    }

    return violations;
}

} // namespace pjev
