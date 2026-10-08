#include "noul_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace pjev
{
using json = nlohmann::json;
namespace
{
    json num(double v)
    {
        return std::isfinite(v) ? json(v) : json(nullptr);
    }

    double num_or_nan(const json& j, const char* k)
    {
        return (j.contains(k) && j[k].is_number()) ? j[k].get<double>() : METRIC_UNDEFINED;
    }

    bool bool_or(const json& j, const char* k, bool def)
    {
        return (j.contains(k) && j[k].is_boolean()) ? j[k].get<bool>() : def;
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

double semantic_margin(const std::vector<float>& logits, const std::vector<std::string>& keys)
{
    if(logits.size() != keys.size())
        return METRIC_UNDEFINED;
    int ti = -1, fi = -1;
    for(int i = 0; i < (int)keys.size(); i++) {
        if(keys[i] == "true")
            ti = i;
        else if(keys[i] == "false")
            fi = i;
    }
    if(ti < 0 || fi < 0)
        return METRIC_UNDEFINED;
    return (double)logits[ti] - (double)logits[fi];
}

double signed_margin(double semantic_margin, bool ground_truth)
{
    return ground_truth ? semantic_margin : -semantic_margin;
}

double binary_nll(double p_true, bool ground_truth)
{
    double p = ground_truth ? p_true : 1.0 - p_true;
    p = std::clamp(p, NOUL_PROB_EPS, 1.0 - NOUL_PROB_EPS);
    return -std::log(p);
}

double binary_brier(double p_true, bool ground_truth)
{
    double d = p_true - (ground_truth ? 1.0 : 0.0);
    return d * d;
}

double percentile_linear(std::vector<double> values, double q)
{
    if(values.empty())
        return METRIC_UNDEFINED;
    std::sort(values.begin(), values.end());
    q = std::clamp(q, 0.0, 1.0);
    double h = (double)(values.size() - 1) * q;
    size_t lo = (size_t)std::floor(h);
    size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (h - (double)lo) * (values[hi] - values[lo]);
}

// ---------------------------------------------------------------------------
// NoulSample
// ---------------------------------------------------------------------------

double NoulSample::confidence() const
{
    return prediction ? p_true : 1.0 - p_true;
}

double NoulSample::signed_margin() const
{
    if(!labeled)
        return METRIC_UNDEFINED;
    return pjev::signed_margin(corrected_semantic_margin, ground_truth);
}

json NoulSample::to_json() const
{
    json j = {
        {"source_id", source_id},
        {"ground_truth", labeled ? json(ground_truth) : json(nullptr)},
        {"prediction", prediction},
        {"correct", labeled ? json(correct()) : json(nullptr)},
        {"p_true", num(p_true)},
        {"p_true_raw", num(p_true_raw)},
        {"confidence", num(confidence())},
        {"raw_semantic_margin", num(raw_semantic_margin)},
        {"semantic_margin", num(corrected_semantic_margin)},
        {"signed_margin", num(signed_margin())}};
    if(has_orders) {
        j["order1_semantic_margin"] = num(order1_semantic_margin);
        j["order2_semantic_margin"] = num(order2_semantic_margin);
        j["ensemble_semantic_margin"] = num(corrected_semantic_margin);
        j["order1_prediction"] = order1_prediction();
        j["order2_prediction"] = order2_prediction();
        j["order_disagreement"] = order_disagreement();
    }
    return j;
}

NoulSample NoulSample::from_json(const json& j)
{
    NoulSample s;
    if(j.contains("source_id") && j["source_id"].is_string())
        s.source_id = j["source_id"].get<std::string>();
    s.labeled = j.contains("ground_truth") && j["ground_truth"].is_boolean();
    s.ground_truth = bool_or(j, "ground_truth", false);
    s.prediction = bool_or(j, "prediction", false);
    s.p_true = num_or_nan(j, "p_true");
    s.p_true_raw = num_or_nan(j, "p_true_raw");
    s.raw_semantic_margin = num_or_nan(j, "raw_semantic_margin");
    s.corrected_semantic_margin = num_or_nan(j, "semantic_margin");
    s.has_orders = j.contains("order1_semantic_margin");
    s.order1_semantic_margin = num_or_nan(j, "order1_semantic_margin");
    s.order2_semantic_margin = num_or_nan(j, "order2_semantic_margin");
    return s;
}

// ---------------------------------------------------------------------------
// Probability metrics
// ---------------------------------------------------------------------------

BinaryProbMetrics compute_binary_prob_metrics(const std::vector<double>& p_true,
                                              const std::vector<bool>& ground_truth,
                                              const std::vector<bool>& prediction,
                                              int32_t n_bins)
{
    BinaryProbMetrics m;
    m.ece_bins = n_bins;
    if(n_bins < 1 || p_true.size() != ground_truth.size() || p_true.size() != prediction.size())
        return m;

    double sum_nll = 0.0, sum_brier = 0.0, sum_conf_correct = 0.0;
    int32_t n_correct = 0, n_valid = 0;
    std::vector<int32_t> bin_count(n_bins, 0), bin_correct(n_bins, 0);
    std::vector<double> bin_conf_sum(n_bins, 0.0);

    for(size_t i = 0; i < p_true.size(); i++) {
        double p = p_true[i];
        if(!std::isfinite(p))
            continue;
        n_valid++;
        sum_nll += binary_nll(p, ground_truth[i]);
        sum_brier += binary_brier(p, ground_truth[i]);

        bool correct = prediction[i] == ground_truth[i];
        double conf = prediction[i] ? p : 1.0 - p;
        if(correct) {
            n_correct++;
            sum_conf_correct += conf;
        }
        int32_t b = std::clamp((int32_t)(conf * n_bins), 0, n_bins - 1);
        bin_count[b]++;
        bin_conf_sum[b] += conf;
        if(correct)
            bin_correct[b]++;
    }

    m.n = n_valid;
    if(n_valid == 0)
        return m;
    m.nll = sum_nll / n_valid;
    m.brier = sum_brier / n_valid;
    if(n_correct > 0)
        m.mean_confidence_correct = sum_conf_correct / n_correct;

    double ece = 0.0;
    for(int32_t b = 0; b < n_bins; b++) {
        if(bin_count[b] == 0)
            continue;
        double frac = (double)bin_count[b] / n_valid;
        double acc = (double)bin_correct[b] / bin_count[b];
        double conf = bin_conf_sum[b] / bin_count[b];
        ece += frac * std::fabs(conf - acc);
    }
    m.ece = ece;
    return m;
}

json BinaryProbMetrics::to_json() const
{
    return json{
        {"n", n},
        {"nll", num(nll)},
        {"brier", num(brier)},
        {"ece", num(ece)},
        {"ece_bins", ece_bins},
        {"mean_confidence_correct", num(mean_confidence_correct)}};
}

// ---------------------------------------------------------------------------
// Margins
// ---------------------------------------------------------------------------

MarginSummary summarize_margins(const std::vector<double>& values)
{
    MarginSummary s;
    std::vector<double> v;
    v.reserve(values.size());
    for(double x: values)
        if(std::isfinite(x))
            v.push_back(x);
    s.n = (int32_t)v.size();
    if(v.empty())
        return s;
    s.mean = mean_of(v);
    s.median = percentile_linear(v, 0.5);
    s.p10 = percentile_linear(v, 0.1);
    s.min = *std::min_element(v.begin(), v.end());
    return s;
}

json MarginSummary::to_json() const
{
    return json{
        {"n", n},
        {"mean", num(mean)},
        {"median", num(median)},
        {"p10", num(p10)},
        {"min", num(min)}};
}

// ---------------------------------------------------------------------------
// NoulMetrics
// ---------------------------------------------------------------------------

NoulMetrics compute_noul_metrics(const std::vector<NoulSample>& samples, int32_t n_bins)
{
    NoulMetrics m;
    std::vector<double> p_cal, p_raw, sm, sm_raw;
    std::vector<bool> gt, pred;
    int32_t n_correct = 0;

    for(const auto& s: samples) {
        m.n++;
        if(s.has_orders) {
            m.has_orders = true;
            if(std::isfinite(s.order1_semantic_margin) && std::isfinite(s.order2_semantic_margin)) {
                m.order_n++;
                if(s.order_disagreement())
                    m.order_disagreement_count++;
            }
        }
        if(!s.labeled)
            continue;
        m.labeled++;
        if(s.correct())
            n_correct++;
        p_cal.push_back(s.p_true);
        p_raw.push_back(s.p_true_raw);
        gt.push_back(s.ground_truth);
        pred.push_back(s.prediction);
        sm.push_back(s.signed_margin());
        sm_raw.push_back(signed_margin(s.raw_semantic_margin, s.ground_truth));
    }

    if(m.labeled > 0)
        m.accuracy = (double)n_correct / m.labeled;
    m.calibrated = compute_binary_prob_metrics(p_cal, gt, pred, n_bins);
    m.raw = compute_binary_prob_metrics(p_raw, gt, pred, n_bins);
    m.signed_margin = summarize_margins(sm);
    m.raw_signed_margin = summarize_margins(sm_raw);
    return m;
}

void NoulMetrics::merge_into(json& j) const
{
    // Top-level probability metrics are from the probabilities the run reported
    // (calibrated when a calibration artifact is loaded); raw (T = 1) are kept separately.
    j["nll"] = num(calibrated.nll);
    j["brier"] = num(calibrated.brier);
    j["ece"] = num(calibrated.ece);
    j["ece_bins"] = calibrated.ece_bins;
    j["mean_confidence_correct"] = num(calibrated.mean_confidence_correct);

    j["mean_signed_margin"] = num(signed_margin.mean);
    j["median_signed_margin"] = num(signed_margin.median);
    j["p10_signed_margin"] = num(signed_margin.p10);
    j["min_signed_margin"] = num(signed_margin.min);

    j["raw"] = raw.to_json();
    j["calibrated"] = calibrated.to_json();
    j["margins"] = json{
        {"source", "pre-temperature logits"},
        {"corrected", signed_margin.to_json()},
        {"raw", raw_signed_margin.to_json()}};

    if(has_orders) {
        j["order_n"] = order_n;
        j["order_disagreement_count"] = order_disagreement_count;
        j["order_disagreement_rate"] = num(order_disagreement_rate());
    }
}

// ---------------------------------------------------------------------------
// Margin gain
// ---------------------------------------------------------------------------

MarginGain compute_margin_gain(const std::vector<MatchedOutcome>& baseline,
                               const std::vector<MatchedOutcome>& target,
                               double tol)
{
    MarginGain g;
    std::map<std::string, const MatchedOutcome*> base_by_id;
    for(const auto& s: baseline)
        if(!s.source_id.empty())
            base_by_id[s.source_id] = &s;

    std::vector<double> gains;
    for(const auto& t: target) {
        auto it = base_by_id.find(t.source_id);
        if(it == base_by_id.end())
            continue;
        double gain = t.signed_margin - it->second->signed_margin;
        if(!std::isfinite(gain))
            continue;
        gains.push_back(gain);
        if(std::fabs(gain) < tol)
            g.unchanged++;
        else if(gain > 0.0)
            g.improved++;
        else
            g.degraded++;
        if(it->second->correct && !t.correct)
            g.correct_to_wrong++;
        else if(!it->second->correct && t.correct)
            g.wrong_to_correct++;
    }
    g.n_matched = (int32_t)gains.size();
    g.mean_gain = mean_of(gains);
    g.median_gain = percentile_linear(gains, 0.5);
    return g;
}

MarginGain compute_margin_gain(const std::vector<NoulSample>& baseline,
                               const std::vector<NoulSample>& target,
                               double tol)
{
    auto outcomes = [](const std::vector<NoulSample>& v) {
        std::vector<MatchedOutcome> o;
        for(const auto& s: v)
            if(s.labeled)
                o.push_back({s.source_id, s.signed_margin(), s.correct()});
        return o;
    };
    return compute_margin_gain(outcomes(baseline), outcomes(target), tol);
}

json MarginGain::to_json() const
{
    return json{
        {"n_matched", n_matched},
        {"mean_margin_gain", num(mean_gain)},
        {"median_margin_gain", num(median_gain)},
        {"improved_margin_count", improved},
        {"degraded_margin_count", degraded},
        {"unchanged_margin_count", unchanged},
        {"correct_to_wrong_count", correct_to_wrong},
        {"wrong_to_correct_count", wrong_to_correct}};
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

std::string format_noul_comparison(const std::string& title,
                                   const std::string& baseline_label, const NoulMetrics& b, int64_t b_ms,
                                   const std::string& target_label, const NoulMetrics& t, int64_t t_ms,
                                   const MarginGain& gain)
{
    std::string out;
    char buf[256];
    auto f = [](double v) -> std::string {
        if(!std::isfinite(v))
            return "-";
        char s[32];
        snprintf(s, sizeof(s), "%.4f", v);
        return s;
    };
    auto row = [&](const char* name, const std::string& a, const std::string& c) {
        snprintf(buf, sizeof(buf), "%-26s %14s %14s\n", name, a.c_str(), c.c_str());
        out += buf;
    };
    auto row_i = [&](const char* name, int64_t a, int64_t c) {
        row(name, a < 0 ? "-" : std::to_string(a), c < 0 ? "-" : std::to_string(c));
    };

    out += title + "\n\n";
    row("metric", baseline_label, target_label);
    out += std::string(56, '-') + "\n";
    row_i("labeled", b.labeled, t.labeled);
    row("accuracy", f(b.accuracy), f(t.accuracy));
    row("NLL", f(b.calibrated.nll), f(t.calibrated.nll));
    row("Brier", f(b.calibrated.brier), f(t.calibrated.brier));
    row("ECE", f(b.calibrated.ece), f(t.calibrated.ece));
    row("NLL (raw, T=1)", f(b.raw.nll), f(t.raw.nll));
    row("Brier (raw, T=1)", f(b.raw.brier), f(t.raw.brier));
    row("ECE (raw, T=1)", f(b.raw.ece), f(t.raw.ece));
    row("mean conf. correct", f(b.calibrated.mean_confidence_correct), f(t.calibrated.mean_confidence_correct));
    row("mean conf. correct (raw)", f(b.raw.mean_confidence_correct), f(t.raw.mean_confidence_correct));
    row("mean signed margin", f(b.signed_margin.mean), f(t.signed_margin.mean));
    row("median signed margin", f(b.signed_margin.median), f(t.signed_margin.median));
    row("p10 signed margin", f(b.signed_margin.p10), f(t.signed_margin.p10));
    row("min signed margin", f(b.signed_margin.min), f(t.signed_margin.min));
    if(b.has_orders || t.has_orders) {
        row("order disagreement",
            b.has_orders ? std::to_string(b.order_disagreement_count) + "/" + std::to_string(b.order_n) : "-",
            t.has_orders ? std::to_string(t.order_disagreement_count) + "/" + std::to_string(t.order_n) : "-");
    }
    row_i("eval_ms", b_ms, t_ms);

    out += format_margin_gain(baseline_label, target_label, gain);
    return out;
}

std::string format_margin_gain(const std::string& baseline_label, const std::string& target_label,
                               const MarginGain& gain)
{
    auto f = [](double v) -> std::string {
        if(!std::isfinite(v))
            return "-";
        char s[32];
        snprintf(s, sizeof(s), "%.4f", v);
        return s;
    };
    std::string out = "\nmargin gain (" + target_label + " - " + baseline_label + "), matched by source_id\n";
    char buf[256];
    snprintf(buf, sizeof(buf),
             "  matched %d  mean %s  median %s  improved %d  degraded %d  unchanged %d\n",
             gain.n_matched, f(gain.mean_gain).c_str(), f(gain.median_gain).c_str(),
             gain.improved, gain.degraded, gain.unchanged);
    out += buf;
    snprintf(buf, sizeof(buf), "  %s correct -> %s wrong: %d   %s wrong -> %s correct: %d\n",
             baseline_label.c_str(), target_label.c_str(), gain.correct_to_wrong,
             baseline_label.c_str(), target_label.c_str(), gain.wrong_to_correct);
    out += buf;
    return out;
}
} // namespace pjev
