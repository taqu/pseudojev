#include "score_metrics.h"

#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

double expected_score(const std::vector<double>& probs, const std::vector<int32_t>& levels)
{
    if(probs.empty() || probs.size() != levels.size())
        return METRIC_UNDEFINED;
    double s = 0.0;
    for(size_t i = 0; i < probs.size(); i++) s += (double)levels[i] * probs[i];
    return s;
}

std::vector<int32_t> error_distance_histogram(const std::vector<int32_t>& predicted,
                                              const std::vector<int32_t>& ground_truth)
{
    std::vector<int32_t> h;
    for(size_t i = 0; i < predicted.size() && i < ground_truth.size(); i++) {
        int32_t d = std::abs(predicted[i] - ground_truth[i]);
        if((int32_t)h.size() <= d)
            h.resize(d + 1, 0);
        h[d]++;
    }
    return h;
}

// ---------------------------------------------------------------------------
// ScoreSample
// ---------------------------------------------------------------------------

int32_t ScoreSample::gt_position() const
{
    for(int32_t i = 0; i < (int32_t)levels.size(); i++)
        if(levels[i] == ground_truth)
            return i;
    return -1;
}

double ScoreSample::ground_truth_probability() const
{
    int32_t p = gt_position();
    return (p >= 0 && p < (int32_t)probs.size()) ? probs[p] : METRIC_UNDEFINED;
}

int32_t ScoreSample::abs_error() const
{
    return labeled() ? std::abs(prediction - ground_truth) : -1;
}

double ScoreSample::expected_abs_error() const
{
    return labeled() ? std::abs(expected_score() - ground_truth) : METRIC_UNDEFINED;
}

double ScoreSample::raw_expected_abs_error() const
{
    return labeled() ? std::abs(raw_expected_score() - ground_truth) : METRIC_UNDEFINED;
}

json ScoreSample::to_json() const
{
    return json{
        {"source_id", source_id},
        {"levels", levels},
        {"labels", labels},
        {"token_ids", token_ids},
        {"ground_truth", labeled() ? json(ground_truth) : json(nullptr)},
        {"prediction", prediction},
        {"correct", labeled() ? json(correct()) : json(nullptr)},
        {"temperature", temperature},
        {"logits", logits},
        {"raw_logits", raw_logits},
        {"probs", probs},
        {"raw_probs", raw_probs},
        {"expected_score", num(expected_score())},
        {"raw_expected_score", num(raw_expected_score())},
        {"ground_truth_probability", num(labeled() ? ground_truth_probability() : METRIC_UNDEFINED)},
        {"signed_margin", num(signed_margin())},
        {"abs_error", labeled() ? json(abs_error()) : json(nullptr)},
        {"expected_abs_error", num(expected_abs_error())},
        {"raw_expected_abs_error", num(raw_expected_abs_error())},
        {"prompt_tokens", prompt_tokens},
        {"evaluations", evaluations}};
}

ScoreSample ScoreSample::from_json(const json& j)
{
    ScoreSample s;
    if(j.contains("source_id") && j["source_id"].is_string())
        s.source_id = j["source_id"].get<std::string>();
    s.levels = vec_or_empty<int32_t>(j, "levels");
    if(j.contains("labels") && j["labels"].is_array())
        for(const auto& l: j["labels"])
            if(l.is_string())
                s.labels.push_back(l.get<std::string>());
    s.token_ids = vec_or_empty<int32_t>(j, "token_ids");
    s.ground_truth = int_or(j, "ground_truth", -1);
    s.prediction = int_or(j, "prediction", -1);
    if(j.contains("temperature") && j["temperature"].is_number())
        s.temperature = j["temperature"].get<double>();
    s.logits = vec_or_empty<double>(j, "logits");
    s.raw_logits = vec_or_empty<double>(j, "raw_logits");
    s.probs = vec_or_empty<double>(j, "probs");
    s.raw_probs = vec_or_empty<double>(j, "raw_probs");
    s.prompt_tokens = int_or(j, "prompt_tokens", 0);
    s.evaluations = int_or(j, "evaluations", 1);
    return s;
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

ScoreMetrics compute_score_metrics(const std::vector<ScoreSample>& samples, int32_t n_bins)
{
    ScoreMetrics m;
    if(!samples.empty())
        m.temperature = samples[0].temperature;

    TypeMetrics tm; // reuse the existing QWK
    std::vector<CalibrationSample> cal_s;
    std::vector<double> abs_err, exp_err, raw_exp_err, sm, sm_raw, dist, raw_dist, evals;
    std::vector<int32_t> pred, gt;
    int32_t n_correct = 0, n_large = 0;

    for(const auto& s: samples) {
        m.n++;
        m.n_levels = std::max(m.n_levels, (int32_t)s.levels.size());
        m.prompt_tokens += s.prompt_tokens;
        evals.push_back((double)s.evaluations);
        double d = s.expected_argmax_distance(), rd = s.raw_expected_argmax_distance();
        if(std::isfinite(d))
            dist.push_back(d);
        if(std::isfinite(rd))
            raw_dist.push_back(rd);
        if(!s.labeled())
            continue;

        m.labeled++;
        if(s.correct())
            n_correct++;
        int32_t e = s.abs_error();
        abs_err.push_back((double)e);
        if(e >= m.large_error_threshold)
            n_large++;
        exp_err.push_back(s.expected_abs_error());
        raw_exp_err.push_back(s.raw_expected_abs_error());
        sm.push_back(s.signed_margin());
        sm_raw.push_back(s.raw_signed_margin());
        pred.push_back(s.prediction);
        gt.push_back(s.ground_truth);
        tm.labeled++;
        tm.predicted_levels.push_back(s.prediction);
        tm.actual_levels.push_back(s.ground_truth);

        CalibrationSample c;
        c.type = "score";
        c.correct_index = s.gt_position();
        c.expected_score = (double)s.ground_truth;
        for(double l: s.logits) c.logits.push_back((float)l);
        cal_s.push_back(std::move(c));
    }

    if(m.labeled > 0) {
        m.accuracy = (double)n_correct / m.labeled;
        m.large_error_rate = (double)n_large / m.labeled;
    }
    m.mae = mean_of(abs_err);
    m.expected_mae = mean_of(exp_err);
    m.raw_expected_mae = mean_of(raw_exp_err);
    if(m.labeled >= 2 && m.n_levels >= 2)
        m.qwk = tm.qwk(m.n_levels);
    m.calibrated = compute_primitive_metrics(cal_s, m.temperature, n_bins);
    m.raw = compute_primitive_metrics(cal_s, 1.0, n_bins);
    m.signed_margin = summarize_margins(sm);
    m.raw_signed_margin = summarize_margins(sm_raw);
    m.mean_expected_argmax_distance = mean_of(dist);
    m.raw_mean_expected_argmax_distance = mean_of(raw_dist);
    m.error_histogram = error_distance_histogram(pred, gt);
    m.evaluations_per_item = mean_of(evals);
    return m;
}

void ScoreMetrics::merge_into(json& j) const
{
    // Existing "mae" (discrete) and "qwk" keys are left as computed by TypeMetrics.
    j["expected_score_mae"] = num(expected_mae);
    j["raw_expected_score_mae"] = num(raw_expected_mae);
    j["nll"] = num(cal(calibrated.nll));
    j["brier"] = num(cal(calibrated.brier));
    j["ece"] = num(cal(calibrated.ece));
    j["ece_bins"] = calibrated.n_bins;
    j["brier_convention"] = "sum over levels";
    j["temperature"] = temperature;

    j["mean_signed_margin"] = num(signed_margin.mean);
    j["median_signed_margin"] = num(signed_margin.median);
    j["p10_signed_margin"] = num(signed_margin.p10);
    j["min_signed_margin"] = num(signed_margin.min);
    j["mean_expected_argmax_distance"] = num(mean_expected_argmax_distance);
    j["raw_mean_expected_argmax_distance"] = num(raw_mean_expected_argmax_distance);

    j["error_distance_histogram"] = error_histogram;
    j["large_error_threshold"] = large_error_threshold;
    j["large_error_rate"] = num(large_error_rate);
    j["prompt_tokens"] = prompt_tokens;
    j["evaluations_per_item"] = num(evaluations_per_item);

    auto prob_j = [](const PrimitiveCalibrationMetrics& p) {
        return json{
            {"n", p.n},
            {"nll", num(cal(p.nll))},
            {"brier", num(cal(p.brier))},
            {"ece", num(cal(p.ece))},
            {"ece_bins", p.n_bins}};
    };
    json raw_j = prob_j(raw);
    raw_j["expected_score_mae"] = num(raw_expected_mae);
    raw_j["mean_expected_argmax_distance"] = num(raw_mean_expected_argmax_distance);
    json cal_j = prob_j(calibrated);
    cal_j["expected_score_mae"] = num(expected_mae);
    cal_j["mean_expected_argmax_distance"] = num(mean_expected_argmax_distance);
    j["raw"] = raw_j;
    j["calibrated"] = cal_j;
    j["margins"] = json{
        {"source", "pre-temperature logits"},
        {"corrected", signed_margin.to_json()},
        {"raw", raw_signed_margin.to_json()}};
}

MarginGain compute_margin_gain(const std::vector<ScoreSample>& baseline,
                               const std::vector<ScoreSample>& target,
                               double tol)
{
    auto outcomes = [](const std::vector<ScoreSample>& v) {
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

std::string format_score_comparison(const std::string& title,
                                    const std::string& baseline_label, const ScoreMetrics& b, int64_t b_ms,
                                    const std::string& target_label, const ScoreMetrics& t, int64_t t_ms,
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
    auto row = [&](const std::string& name, const std::string& a, const std::string& c) {
        snprintf(buf, sizeof(buf), "%-30s %14s %14s\n", name.c_str(), a.c_str(), c.c_str());
        out += buf;
    };
    auto hist = [](const ScoreMetrics& m, size_t d) {
        return std::to_string(d < m.error_histogram.size() ? m.error_histogram[d] : 0);
    };
    auto ms = [](int64_t v) { return v < 0 ? std::string("-") : std::to_string(v); };

    out += title + "\n\n";
    row("metric", baseline_label, target_label);
    out += std::string(60, '-') + "\n";
    row("labeled", std::to_string(b.labeled), std::to_string(t.labeled));
    row("accuracy", f(b.accuracy), f(t.accuracy));
    row("MAE", f(b.mae), f(t.mae));
    row("expected-score MAE", f(b.expected_mae), f(t.expected_mae));
    row("expected-score MAE (raw)", f(b.raw_expected_mae), f(t.raw_expected_mae));
    row("QWK", f(b.qwk), f(t.qwk));
    out += "\n";
    row("NLL", f(cal(b.calibrated.nll)), f(cal(t.calibrated.nll)));
    row("Brier", f(cal(b.calibrated.brier)), f(cal(t.calibrated.brier)));
    row("ECE", f(cal(b.calibrated.ece)), f(cal(t.calibrated.ece)));
    row("NLL (raw, T=1)", f(cal(b.raw.nll)), f(cal(t.raw.nll)));
    row("Brier (raw, T=1)", f(cal(b.raw.brier)), f(cal(t.raw.brier)));
    row("ECE (raw, T=1)", f(cal(b.raw.ece)), f(cal(t.raw.ece)));
    out += "\n";
    row("mean signed margin", f(b.signed_margin.mean), f(t.signed_margin.mean));
    row("median signed margin", f(b.signed_margin.median), f(t.signed_margin.median));
    row("p10 signed margin", f(b.signed_margin.p10), f(t.signed_margin.p10));
    row("min signed margin", f(b.signed_margin.min), f(t.signed_margin.min));
    row("mean expected-argmax dist.", f(b.mean_expected_argmax_distance), f(t.mean_expected_argmax_distance));
    row("mean exp.-argmax dist. (raw)", f(b.raw_mean_expected_argmax_distance), f(t.raw_mean_expected_argmax_distance));
    out += "\n";
    size_t max_d = std::max(b.error_histogram.size(), t.error_histogram.size());
    for(size_t d = 0; d < std::max<size_t>(max_d, 3); d++)
        row("error distance " + std::to_string(d), hist(b, d), hist(t, d));
    row("large error rate (>= " + std::to_string(b.large_error_threshold) + ")", f(b.large_error_rate), f(t.large_error_rate));
    out += "\n";
    row("eval_ms", ms(b_ms), ms(t_ms));
    row("prompt tokens", std::to_string(b.prompt_tokens), std::to_string(t.prompt_tokens));
    row("evaluations per item", f(b.evaluations_per_item), f(t.evaluations_per_item));

    out += format_margin_gain(baseline_label, target_label, gain);
    return out;
}
} // namespace pjev
