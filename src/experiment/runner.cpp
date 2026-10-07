#include "runner.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include "../util/timer.h"
#include "spdlog/spdlog.h"

namespace pjev
{
using json = nlohmann::json;
namespace
{
    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    bool check_correct(const ItemResult& ir, const json& expected, Type type)
    {
        if(!ir.ok)
            return false;
        if(expected.is_null())
            return false;
        if(type == Type::Noul || type == Type::Choice) {
            if(!expected.is_string())
                return false;
            if(ir.selected < 0 || ir.selected >= (int)ir.keys.size())
                return false;
            return ir.keys[ir.selected] == expected.get<std::string>();
        }
        if(type == Type::Score) {
            if(!expected.is_number_integer())
                return false;
            return ir.selected == expected.get<int>();
        }
        return false;
    }

    void accumulate(RunMetrics& m, const ItemResult& ir, const json& expected, Type type, int n_levels)
    {
        m.n_total++;
        if(!ir.ok) {
            m.n_errors++;
            return;
        }

        TypeMetrics* tm = nullptr;
        if(type == Type::Choice)
            tm = &m.choice;
        else if(type == Type::Noul)
            tm = &m.noul;
        else if(type == Type::Score) {
            tm = &m.score;
            if(n_levels > m.score_n_levels)
                m.score_n_levels = n_levels;
        }
        if(!tm)
            return;

        tm->n++;
        if(expected.is_null())
            return;

        if(type == Type::Noul || type == Type::Choice) {
            if(!expected.is_string())
                return;
            tm->labeled++;
            if(ir.correct)
                tm->correct++;
        } else if(type == Type::Score) {
            if(!expected.is_number_integer())
                return;
            int actual = expected.get<int>();
            tm->labeled++;
            if(ir.correct)
                tm->correct++;
            double aerr = std::fabs((double)ir.selected - (double)actual);
            tm->abs_err += aerr;
            tm->predicted_levels.push_back(ir.selected);
            tm->actual_levels.push_back(actual);
        }
    }
} // namespace

// ---------------------------------------------------------------------------
// run_experiment
// ---------------------------------------------------------------------------

RunResult run_experiment(ILlamaBackend& backend,
                         const std::vector<DatasetRow>& rows,
                         const ExperimentConfig& cfg)
{
    RunResult result;
    result.config = cfg;

    DecisionEngine engine(backend, PromptConfig{cfg.layout, cfg.scheme}, cfg.calibration, cfg.ensemble);

    // Pre-compute blank logits cache per (type, n_options) if prior correction enabled
    std::map<std::pair<int32_t, int32_t>, std::vector<float>> prior_cache;
    if(cfg.prior_correction) {
        for(const auto& row: rows) {
            auto key = std::make_pair(static_cast<int32_t>(row.input.type), static_cast<int32_t>(row.input.options.size()));
            if(prior_cache.find(key) == prior_cache.end()) {
                prior_cache[key] = engine.compute_blank_logits(row.input.type, (int)row.input.options.size());
            }
        }
    }

    for(size_t i = 0; i < rows.size(); ++i) {
        const DatasetRow& row = rows[i];
        spdlog::info("{} {} {}/{}", row.id, static_cast<int32_t>(row.input.type), i + 1, rows.size());
        ItemResult ir;
        ir.id = row.id;
        ir.type = row.input.type;

        DecisionInput inp = row.input;
        inp.options = permute_options(row.input.options, cfg.option_order, cfg.random_seed);
        inp.prior_correction = cfg.prior_correction;
        inp.collect_corrected_logits = cfg.collect_corrected_logits;
        if(cfg.prior_correction) {
            auto key = std::make_pair(static_cast<int32_t>(row.input.type), static_cast<int32_t>(row.input.options.size()));
            auto it = prior_cache.find(key);
            if(it != prior_cache.end() && !it->second.empty()) {
                inp.prior_logits = it->second;
            }
        }

        Timer t;
        t.start();
        DecisionOutput out = engine.decide(inp);
        t.stop();

        ir.ok = out.ok;
        ir.error = out.error;
        ir.selected = out.selected;
        ir.probs = out.probs;
        ir.raw_probs = out.raw_probs;
        ir.corrected_logits = out.corrected_logits;
        ir.keys = out.keys;
        ir.prompt_token_count = out.prompt_token_count;
        ir.eval_ms = t.elapsed().count();
        ir.ensemble_diag = out.ensemble_diag;
        // Multilingual metadata from dataset row
        ir.language = row.language;
        ir.pair_id = row.pair_id;
        ir.difficulty = row.difficulty;

        if(out.ok && out.selected >= 0 && out.selected < (int)out.keys.size()) {
            ir.stable_key = out.keys[out.selected];
        }

        ir.correct = check_correct(ir, row.expected, row.input.type);

        // Populate correct_index and expected_score for calibration use
        if(!row.expected.is_null()) {
            if(row.input.type == Type::Noul || row.input.type == Type::Choice) {
                if(row.expected.is_string()) {
                    std::string exp_key = row.expected.get<std::string>();
                    for(int ki = 0; ki < (int)ir.keys.size(); ki++) {
                        if(ir.keys[ki] == exp_key) {
                            ir.correct_index = ki;
                            break;
                        }
                    }
                }
            } else if(row.input.type == Type::Score) {
                if(row.expected.is_number_integer()) {
                    ir.correct_index = row.expected.get<int>();
                    ir.expected_score = (double)ir.correct_index;
                }
            }
        }

        accumulate(result.metrics, ir, row.expected, row.input.type, (int)inp.options.size());
        result.items.push_back(std::move(ir));
    }

    // Build NoulSamples and compute confidence/calibration/margin metrics.
    static constexpr double kMarginEps = 1e-12;
    for(const auto& ir: result.items) {
        if(ir.type != Type::Noul || !ir.ok)
            continue;
        // Locate true/false candidate indices by key
        int true_idx = -1, false_idx = -1;
        for(int i = 0; i < (int)ir.keys.size(); i++) {
            if(ir.keys[i] == "true")       true_idx  = i;
            else if(ir.keys[i] == "false") false_idx = i;
        }
        if(true_idx < 0 || false_idx < 0 ||
           true_idx  >= (int)ir.raw_probs.size() ||
           false_idx >= (int)ir.raw_probs.size() ||
           true_idx  >= (int)ir.probs.size() ||
           false_idx >= (int)ir.probs.size())
            continue;

        NoulSample ns;
        ns.id             = ir.id;
        ns.has_ground_truth = (ir.correct_index >= 0);
        if(ns.has_ground_truth) {
            ns.ground_truth = (ir.keys[ir.correct_index] == "true");
            ns.correct      = ir.correct;
        }
        ns.p_true_calib   = ir.probs[true_idx];

        // Semantic margin = log(p_true_raw / p_false_raw) = logit_true - logit_false (pre-temperature)
        double pt = std::max(kMarginEps, ir.raw_probs[true_idx]);
        double pf = std::max(kMarginEps, ir.raw_probs[false_idx]);
        ns.semantic_margin = std::log(pt) - std::log(pf);

        // E1 ensemble diagnostics
        if(ir.ensemble_diag.has_value()) {
            ns.has_ensemble = true;
            ns.m1           = ir.ensemble_diag->m1;
            ns.m2           = ir.ensemble_diag->m2;
            ns.ord1_true    = (ir.ensemble_diag->m1 > 0.0);
            ns.ord2_true    = (ir.ensemble_diag->m2 > 0.0);
        }

        result.noul_samples.push_back(ns);
    }
    result.noul_metrics = compute_noul_metrics(result.noul_samples);

    return result;
}

// ---------------------------------------------------------------------------
// RunResult::to_json
// ---------------------------------------------------------------------------
json RunResult::to_json() const
{
    const auto& m = metrics;
    const auto& c = config;

    int64_t choice_eval_ms = 0;
    int64_t noul_eval_ms = 0;
    int64_t score_eval_ms = 0;
    for(auto&& item: items) {
        switch(item.type) {
        case Type::Choice:
            choice_eval_ms += item.eval_ms;
            break;
        case Type::Noul:
            noul_eval_ms += item.eval_ms;
            break;
        case Type::Score:
            score_eval_ms += item.eval_ms;
            break;
        default:
            break;
        }
    }

    json choice_j = {
        {"n", m.choice.n},
        {"labeled", m.choice.labeled},
        {"accuracy", m.choice.accuracy()},
        {"eval_ms", choice_eval_ms}};
    json noul_j = {
        {"n", m.noul.n},
        {"labeled", m.noul.labeled},
        {"accuracy", m.noul.accuracy()},
        {"eval_ms", noul_eval_ms}};
    // Merge confidence/calibration/margin metrics
    for(const auto& [k, v]: noul_metrics.to_json().items())
        noul_j[k] = v;
    json score_j = {
        {"n", m.score.n},
        {"labeled", m.score.labeled},
        {"accuracy", m.score.accuracy()},
        {"mae", m.score.mae()},
        {"qwk", m.score.qwk(m.score_n_levels)},
        {"eval_ms", score_eval_ms}};

    json stab_j = {
        {"n_questions", stability.n_questions},
        {"n_stable", stability.n_stable},
        {"stability_rate", stability.stability_rate()}};

    return json{
        {"experiment", c.name},
        {"layout", c.layout_str()},
        {"scheme", c.scheme_str()},
        {"option_order", c.order_str()},
        {"prior_correction", c.prior_correction},
        {"metrics", json{
                        {"n_total", m.n_total},
                        {"n_errors", m.n_errors},
                        {"choice", choice_j},
                        {"noul", noul_j},
                        {"score", score_j}}},
        {"stability", stab_j}};
}

// ---------------------------------------------------------------------------
// CompareResult::to_json
// ---------------------------------------------------------------------------

json CompareResult::to_json() const
{
    json runs_j = json::array();
    for(const auto& r: runs) runs_j.push_back(r.to_json());
    json result = json{
        {"experiment", experiment_name},
        {"model", model_path},
        {"runs", runs_j}};

    // When exactly 2 runs are present and have noul samples, compute margin gain.
    // Convention: runs[0] = baseline, runs[1] = E1.
    if(runs.size() == 2 &&
       !runs[0].noul_samples.empty() && !runs[1].noul_samples.empty()) {
        auto mg = compute_margin_gain(runs[1].noul_samples, runs[0].noul_samples);
        if(mg.n_matched > 0)
            result["noul_margin_gain"] = mg.to_json();
    }

    return result;
}

// ---------------------------------------------------------------------------
// Named experiments
// ---------------------------------------------------------------------------

CompareResult exp_candidate_binding(ILlamaBackend& backend,
                                    const std::vector<DatasetRow>& rows,
                                    const std::string& model_path)
{
    CompareResult cr;
    cr.experiment_name = "candidate-binding";
    cr.model_path = model_path;

    ExperimentConfig cfg_natural;
    cfg_natural.name = "natural";
    cfg_natural.scheme = Scheme::NATURAL;
    cr.runs.push_back(run_experiment(backend, rows, cfg_natural));

    ExperimentConfig cfg_letters;
    cfg_letters.name = "letters";
    cfg_letters.scheme = Scheme::LETTERS;
    cr.runs.push_back(run_experiment(backend, rows, cfg_letters));

    return cr;
}

CompareResult exp_option_order(ILlamaBackend& backend,
                               const std::vector<DatasetRow>& rows,
                               const std::string& model_path)
{
    CompareResult cr;
    cr.experiment_name = "option-order";
    cr.model_path = model_path;

    ExperimentConfig cfg_orig;
    cfg_orig.name = "original";
    cfg_orig.option_order = OptionOrder::ORIGINAL;
    RunResult run_orig = run_experiment(backend, rows, cfg_orig);

    ExperimentConfig cfg_rev;
    cfg_rev.name = "reversed";
    cfg_rev.option_order = OptionOrder::REVERSED;
    RunResult run_rev = run_experiment(backend, rows, cfg_rev);

    // Compute stability across the two orderings (choice and noul only)
    StabilityMetrics stab;
    stab.n_permutations_per_q = 2;
    size_t n = std::min(run_orig.items.size(), run_rev.items.size());
    for(size_t i = 0; i < n; i++) {
        const auto& a = run_orig.items[i];
        const auto& b = run_rev.items[i];
        // Only count choice and noul items (skip score — reordering changes key positions)
        if(a.type == Type::Score)
            continue;
        if(!a.ok || !b.ok)
            continue;
        stab.n_questions++;
        if(a.stable_key == b.stable_key)
            stab.n_stable++;
    }

    run_orig.stability = stab;
    run_rev.stability = stab;

    cr.runs.push_back(std::move(run_orig));
    cr.runs.push_back(std::move(run_rev));
    return cr;
}

CompareResult exp_prompt_layout(ILlamaBackend& backend,
                                const std::vector<DatasetRow>& rows,
                                const std::string& model_path)
{
    CompareResult cr;
    cr.experiment_name = "prompt-layout";
    cr.model_path = model_path;

    struct LayoutSpec
    {
        std::string name;
        Layout layout;
    };
    LayoutSpec specs[] = {
        {"auto", Layout::AUTO},
        {"state-first", Layout::STATE_FIRST},
        {"state-last", Layout::STATE_LAST},
        {"question-first", Layout::QUESTION_FIRST}};
    for(const auto& s: specs) {
        ExperimentConfig cfg;
        cfg.name = s.name;
        cfg.layout = s.layout;
        cr.runs.push_back(run_experiment(backend, rows, cfg));
    }
    return cr;
}

CompareResult exp_prior_correction(ILlamaBackend& backend,
                                   const std::vector<DatasetRow>& rows,
                                   const std::string& model_path)
{
    CompareResult cr;
    cr.experiment_name = "prior-correction";
    cr.model_path = model_path;

    ExperimentConfig cfg_no;
    cfg_no.name = "no-prior-correction";
    cfg_no.prior_correction = false;
    cr.runs.push_back(run_experiment(backend, rows, cfg_no));

    ExperimentConfig cfg_yes;
    cfg_yes.name = "with-prior-correction";
    cfg_yes.prior_correction = true;
    cr.runs.push_back(run_experiment(backend, rows, cfg_yes));

    return cr;
}

CompareResult exp_score_formulation(ILlamaBackend& backend,
                                    const std::vector<DatasetRow>& rows,
                                    const std::string& model_path)
{
    CompareResult cr;
    cr.experiment_name = "score-formulation";
    cr.model_path = model_path;

    // Filter to score rows only
    std::vector<DatasetRow> score_rows;
    for(const auto& r: rows) {
        if(r.input.type == Type::Score)
            score_rows.push_back(r);
    }

    ExperimentConfig cfg_natural;
    cfg_natural.name = "natural";
    cfg_natural.scheme = Scheme::NATURAL;
    cr.runs.push_back(run_experiment(backend, score_rows, cfg_natural));

    ExperimentConfig cfg_letters;
    cfg_letters.name = "letters";
    cfg_letters.scheme = Scheme::LETTERS;
    cr.runs.push_back(run_experiment(backend, score_rows, cfg_letters));

    return cr;
}
} // namespace pjev
