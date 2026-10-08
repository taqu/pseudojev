#include "choice_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>

namespace pjev
{
using json = nlohmann::json;
namespace
{
    json num(double v)
    {
        return std::isfinite(v) ? json(v) : json(nullptr);
    }

    // PrimitiveCalibrationMetrics marks undefined values with -1.
    double cal(double v)
    {
        return v < 0.0 ? METRIC_UNDEFINED : v;
    }

    template<class T>
    std::vector<T> vec_or_empty(const json& j, const char* k)
    {
        std::vector<T> v;
        if(j.contains(k) && j[k].is_array())
            for(const auto& e: j[k])
                if(e.is_number())
                    v.push_back(e.get<T>());
        return v;
    }

    int32_t int_or(const json& j, const char* k, int32_t def)
    {
        return (j.contains(k) && j[k].is_number_integer()) ? j[k].get<int32_t>() : def;
    }

    double mean_of(const std::vector<double>& v)
    {
        if(v.empty())
            return METRIC_UNDEFINED;
        double s = 0.0;
        for(double x: v) s += x;
        return s / (double)v.size();
    }
} // namespace

double signed_winner_margin(const std::vector<double>& scores, int32_t gt)
{
    if(gt < 0 || gt >= (int32_t)scores.size() || scores.size() < 2)
        return METRIC_UNDEFINED;
    double best_wrong = -std::numeric_limits<double>::infinity();
    for(int32_t i = 0; i < (int32_t)scores.size(); i++)
        if(i != gt)
            best_wrong = std::max(best_wrong, scores[i]);
    return scores[gt] - best_wrong;
}

// ---------------------------------------------------------------------------
// ChoiceSample
// ---------------------------------------------------------------------------

int32_t ChoiceSample::distinct_winners() const
{
    std::set<int32_t> w;
    for(const auto& r: rotations) w.insert(r.prediction);
    return (int32_t)w.size();
}

std::vector<int32_t> ChoiceSample::winner_counts() const
{
    std::vector<int32_t> c(keys.size(), 0);
    for(const auto& r: rotations)
        if(r.prediction >= 0 && r.prediction < (int32_t)c.size())
            c[r.prediction]++;
    return c;
}

double ChoiceSample::winner_agreement() const
{
    if(rotations.empty())
        return METRIC_UNDEFINED;
    int32_t agree = 0;
    for(const auto& r: rotations)
        if(r.prediction == prediction)
            agree++;
    return (double)agree / rotations.size();
}

std::vector<double> ChoiceSample::semantic_logit_variance() const
{
    std::vector<double> var;
    if(rotations.empty())
        return var;
    const size_t k = keys.size();
    for(size_t i = 0; i < k; i++) {
        double sum = 0.0, sq = 0.0;
        size_t cnt = 0;
        for(const auto& r: rotations) {
            if(i >= r.semantic_logits.size())
                continue;
            sum += r.semantic_logits[i];
            sq += r.semantic_logits[i] * r.semantic_logits[i];
            cnt++;
        }
        if(cnt == 0)
            continue;
        double mean = sum / cnt;
        var.push_back(std::max(0.0, sq / cnt - mean * mean));
    }
    return var;
}

double ChoiceSample::mean_semantic_logit_variance() const
{
    return mean_of(semantic_logit_variance());
}

double ChoiceSample::max_semantic_logit_variance() const
{
    auto v = semantic_logit_variance();
    return v.empty() ? METRIC_UNDEFINED : *std::max_element(v.begin(), v.end());
}

json ChoiceSample::to_json() const
{
    json j = {
        {"source_id", source_id},
        {"keys", keys},
        {"ground_truth", labeled() ? json(keys[ground_truth]) : json(nullptr)},
        {"ground_truth_index", labeled() ? json(ground_truth) : json(nullptr)},
        {"prediction", prediction},
        {"correct", labeled() ? json(correct()) : json(nullptr)},
        {"temperature", temperature},
        {"logits", logits},
        {"raw_logits", raw_logits},
        {"probs", probs},
        {"signed_margin", num(signed_margin())},
        {"prompt_tokens", prompt_tokens}};
    if(!rotations.empty()) {
        json rots = json::array();
        for(const auto& r: rotations) {
            rots.push_back({
                {"rotation", r.rotation},
                {"cand_to_sem", r.cand_to_sem},
                {"raw_logits", r.raw_logits},
                {"corrected_logits", r.corrected_logits},
                {"semantic_logits", r.semantic_logits},
                {"prediction", r.prediction},
                {"prefix_reused", r.prefix_reused}});
        }
        j["rotation_count"] = (int32_t)rotations.size();
        j["rotations"] = rots;
        j["winner_counts"] = winner_counts();
        j["distinct_winners"] = distinct_winners();
        j["rotation_disagreement"] = rotation_disagreement();
        j["mean_semantic_logit_variance"] = num(mean_semantic_logit_variance());
        j["max_semantic_logit_variance"] = num(max_semantic_logit_variance());
    }
    return j;
}

ChoiceSample ChoiceSample::from_json(const json& j)
{
    ChoiceSample s;
    if(j.contains("source_id") && j["source_id"].is_string())
        s.source_id = j["source_id"].get<std::string>();
    if(j.contains("keys") && j["keys"].is_array())
        for(const auto& k: j["keys"])
            if(k.is_string())
                s.keys.push_back(k.get<std::string>());
    s.ground_truth = int_or(j, "ground_truth_index", -1);
    s.prediction = int_or(j, "prediction", -1);
    if(j.contains("temperature") && j["temperature"].is_number())
        s.temperature = j["temperature"].get<double>();
    s.logits = vec_or_empty<double>(j, "logits");
    s.raw_logits = vec_or_empty<double>(j, "raw_logits");
    s.probs = vec_or_empty<double>(j, "probs");
    s.prompt_tokens = int_or(j, "prompt_tokens", 0);
    if(j.contains("rotations") && j["rotations"].is_array()) {
        for(const auto& rj: j["rotations"]) {
            ChoiceRotation r;
            r.rotation = int_or(rj, "rotation", 0);
            r.cand_to_sem = vec_or_empty<int32_t>(rj, "cand_to_sem");
            r.raw_logits = vec_or_empty<double>(rj, "raw_logits");
            r.corrected_logits = vec_or_empty<double>(rj, "corrected_logits");
            r.semantic_logits = vec_or_empty<double>(rj, "semantic_logits");
            r.prediction = int_or(rj, "prediction", -1);
            r.prefix_reused = int_or(rj, "prefix_reused", 0);
            s.rotations.push_back(std::move(r));
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

ChoiceMetrics compute_choice_metrics(const std::vector<ChoiceSample>& samples, int32_t n_bins)
{
    ChoiceMetrics m;
    if(!samples.empty())
        m.temperature = samples[0].temperature;

    std::vector<CalibrationSample> cal_s;
    std::vector<double> sm, sm_raw, rot_counts, distinct, agreement, var_mean, var_max;
    int32_t n_correct = 0;

    for(const auto& s: samples) {
        m.n++;
        m.prompt_tokens += s.prompt_tokens;
        if(!s.rotations.empty()) {
            m.has_rotations = true;
            m.rotation_n++;
            rot_counts.push_back((double)s.rotations.size());
            distinct.push_back((double)s.distinct_winners());
            agreement.push_back(s.winner_agreement());
            if(s.rotation_disagreement())
                m.samples_with_any_rotation_disagreement++;
            double vm = s.mean_semantic_logit_variance(), vx = s.max_semantic_logit_variance();
            if(std::isfinite(vm))
                var_mean.push_back(vm);
            if(std::isfinite(vx))
                var_max.push_back(vx);
        }
        if(!s.labeled())
            continue;
        m.labeled++;
        if(s.correct())
            n_correct++;
        sm.push_back(s.signed_margin());
        sm_raw.push_back(s.raw_signed_margin());

        CalibrationSample c;
        c.type = "choice";
        c.correct_index = s.ground_truth;
        for(double l: s.logits) c.logits.push_back((float)l);
        cal_s.push_back(std::move(c));
    }

    if(m.labeled > 0)
        m.accuracy = (double)n_correct / m.labeled;
    m.calibrated = compute_primitive_metrics(cal_s, m.temperature, n_bins);
    m.raw = compute_primitive_metrics(cal_s, 1.0, n_bins);
    m.signed_margin = summarize_margins(sm);
    m.raw_signed_margin = summarize_margins(sm_raw);
    if(m.has_rotations) {
        m.mean_rotations = mean_of(rot_counts);
        m.mean_distinct_winners = mean_of(distinct);
        m.mean_winner_agreement = mean_of(agreement);
        m.mean_semantic_logit_variance = mean_of(var_mean);
        m.mean_max_semantic_logit_variance = mean_of(var_max);
        if(!var_max.empty())
            m.max_semantic_logit_variance = *std::max_element(var_max.begin(), var_max.end());
    }
    return m;
}

void ChoiceMetrics::merge_into(json& j) const
{
    // Top-level probability metrics use the run's temperature (calibrated when an artifact is
    // loaded). Brier is the summed multi-class form, as in compute_primitive_metrics.
    j["nll"] = num(cal(calibrated.nll));
    j["brier"] = num(cal(calibrated.brier));
    j["ece"] = num(cal(calibrated.ece));
    j["ece_bins"] = calibrated.n_bins;
    j["brier_convention"] = "sum over options";
    j["temperature"] = temperature;

    j["mean_signed_winner_margin"] = num(signed_margin.mean);
    j["median_signed_winner_margin"] = num(signed_margin.median);
    j["p10_signed_winner_margin"] = num(signed_margin.p10);
    j["min_signed_winner_margin"] = num(signed_margin.min);

    auto prob_j = [](const PrimitiveCalibrationMetrics& p) {
        return json{
            {"n", p.n},
            {"nll", num(cal(p.nll))},
            {"brier", num(cal(p.brier))},
            {"ece", num(cal(p.ece))},
            {"ece_bins", p.n_bins}};
    };
    j["raw"] = prob_j(raw);
    j["calibrated"] = prob_j(calibrated);
    j["margins"] = json{
        {"source", "pre-temperature logits"},
        {"corrected", signed_margin.to_json()},
        {"raw", raw_signed_margin.to_json()}};
    j["prompt_tokens"] = prompt_tokens;

    if(has_rotations) {
        j["rotation_n"] = rotation_n;
        j["rotations_per_item"] = num(mean_rotations);
        j["samples_with_any_rotation_disagreement"] = samples_with_any_rotation_disagreement;
        j["rotation_disagreement_rate"] = num(rotation_disagreement_rate());
        j["mean_number_of_distinct_winners"] = num(mean_distinct_winners);
        j["mean_winner_agreement"] = num(mean_winner_agreement);
        j["mean_semantic_logit_variance"] = num(mean_semantic_logit_variance);
        j["mean_max_semantic_logit_variance"] = num(mean_max_semantic_logit_variance);
        j["max_semantic_logit_variance"] = num(max_semantic_logit_variance);
    }
}

MarginGain compute_margin_gain(const std::vector<ChoiceSample>& baseline,
                               const std::vector<ChoiceSample>& target,
                               double tol)
{
    auto outcomes = [](const std::vector<ChoiceSample>& v) {
        std::vector<MatchedOutcome> o;
        for(const auto& s: v)
            if(s.labeled())
                o.push_back({s.source_id, s.signed_margin(), s.correct()});
        return o;
    };
    return compute_margin_gain(outcomes(baseline), outcomes(target), tol);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

std::string format_choice_comparison(const std::string& title,
                                     const std::string& baseline_label, const ChoiceMetrics& b, int64_t b_ms,
                                     const std::string& target_label, const ChoiceMetrics& t, int64_t t_ms,
                                     const MarginGain& gain)
{
    std::string out;
    char buf[256];
    auto f = [](double v) -> std::string {
        if(!std::isfinite(v))
            return "n/a";
        char s[32];
        snprintf(s, sizeof(s), "%.4f", v);
        return s;
    };
    auto row = [&](const char* name, const std::string& a, const std::string& c) {
        snprintf(buf, sizeof(buf), "%-30s %14s %14s\n", name, a.c_str(), c.c_str());
        out += buf;
    };
    auto rot = [&](const ChoiceMetrics& m, double v) { return m.has_rotations ? f(v) : std::string("n/a"); };
    auto ms = [](int64_t v) { return v < 0 ? std::string("-") : std::to_string(v); };

    out += title + "\n\n";
    row("metric", baseline_label, target_label);
    out += std::string(60, '-') + "\n";
    row("labeled", std::to_string(b.labeled), std::to_string(t.labeled));
    row("accuracy", f(b.accuracy), f(t.accuracy));
    row("NLL", f(cal(b.calibrated.nll)), f(cal(t.calibrated.nll)));
    row("Brier", f(cal(b.calibrated.brier)), f(cal(t.calibrated.brier)));
    row("ECE", f(cal(b.calibrated.ece)), f(cal(t.calibrated.ece)));
    row("NLL (raw, T=1)", f(cal(b.raw.nll)), f(cal(t.raw.nll)));
    row("Brier (raw, T=1)", f(cal(b.raw.brier)), f(cal(t.raw.brier)));
    row("ECE (raw, T=1)", f(cal(b.raw.ece)), f(cal(t.raw.ece)));
    out += "\n";
    row("mean winner margin", f(b.signed_margin.mean), f(t.signed_margin.mean));
    row("median winner margin", f(b.signed_margin.median), f(t.signed_margin.median));
    row("p10 winner margin", f(b.signed_margin.p10), f(t.signed_margin.p10));
    row("min winner margin", f(b.signed_margin.min), f(t.signed_margin.min));
    out += "\n";
    row("rotation disagreement rate", rot(b, b.rotation_disagreement_rate()), rot(t, t.rotation_disagreement_rate()));
    row("mean distinct winners", rot(b, b.mean_distinct_winners), rot(t, t.mean_distinct_winners));
    row("mean semantic variance", rot(b, b.mean_semantic_logit_variance), rot(t, t.mean_semantic_logit_variance));
    row("max semantic variance", rot(b, b.max_semantic_logit_variance), rot(t, t.max_semantic_logit_variance));
    row("evaluations per item", b.has_rotations ? f(b.mean_rotations) : "1", t.has_rotations ? f(t.mean_rotations) : "1");
    out += "\n";
    row("prompt tokens", std::to_string(b.prompt_tokens), std::to_string(t.prompt_tokens));
    row("eval_ms", ms(b_ms), ms(t_ms));

    out += format_margin_gain(baseline_label, target_label, gain);
    return out;
}
} // namespace pjev
