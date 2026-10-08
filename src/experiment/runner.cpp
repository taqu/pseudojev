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

    int key_index(const std::vector<std::string>& keys, const char* k)
    {
        for(int i = 0; i < (int)keys.size(); i++)
            if(keys[i] == k)
                return i;
        return -1;
    }

    // Maps a noul DecisionOutput into semantic true/false space. Candidate positions are
    // resolved by key, so the result is independent of option order.
    NoulSample make_noul_sample(const DatasetRow& row, const DecisionOutput& out)
    {
        NoulSample s;
        s.source_id = row.id;
        if(row.expected.is_string()) {
            const std::string e = row.expected.get<std::string>();
            if(e == "true" || e == "false") {
                s.labeled = true;
                s.ground_truth = (e == "true");
            }
        }
        s.prediction = out.selected >= 0 && out.selected < (int)out.keys.size() && out.keys[out.selected] == "true";
        s.p_true = out.p_true;
        int ti = key_index(out.keys, "true");
        if(ti >= 0 && ti < (int)out.raw_probs.size())
            s.p_true_raw = out.raw_probs[ti];

        if(out.ensemble_diag) {
            const auto& d = *out.ensemble_diag;
            s.has_orders = true;
            s.order1_semantic_margin = d.m1;
            s.order2_semantic_margin = d.m2;
            s.corrected_semantic_margin = d.m_ensemble;
            s.raw_semantic_margin = (semantic_margin(d.ord1_raw_logits, d.ord1_keys) +
                                     semantic_margin(d.ord2_raw_logits, d.ord2_keys)) / 2.0;
        } else {
            s.corrected_semantic_margin = semantic_margin(out.corrected_logits, out.keys);
            s.raw_semantic_margin = semantic_margin(out.raw_logits, out.keys);
        }
        return s;
    }

    // Maps a choice DecisionOutput into semantic option space (out.keys order). For E2 the
    // per-rotation scores are already remapped by the engine via the explicit rotation mapping.
    ChoiceSample make_choice_sample(const DatasetRow& row, const DecisionOutput& out, double temperature)
    {
        ChoiceSample s;
        s.source_id = row.id;
        s.keys = out.keys;
        if(row.expected.is_string()) {
            const std::string e = row.expected.get<std::string>();
            for(int i = 0; i < (int)out.keys.size(); i++)
                if(out.keys[i] == e)
                    s.ground_truth = i;
        }
        s.prediction = out.selected;
        s.temperature = temperature;
        s.logits.assign(out.corrected_logits.begin(), out.corrected_logits.end());
        s.raw_logits.assign(out.raw_logits.begin(), out.raw_logits.end());
        s.probs = out.probs;
        s.prompt_tokens = out.prompt_token_count;
        if(out.choice_ensemble_diag) {
            for(const auto& rd: out.choice_ensemble_diag->rotations) {
                ChoiceRotation r;
                r.rotation = rd.rotation;
                r.cand_to_sem = rd.cand_to_sem;
                r.raw_logits.assign(rd.raw_logits.begin(), rd.raw_logits.end());
                r.corrected_logits.assign(rd.corrected_logits.begin(), rd.corrected_logits.end());
                r.semantic_logits.assign(rd.semantic_logits.begin(), rd.semantic_logits.end());
                r.prediction = rd.semantic_prediction;
                r.prefix_reused = rd.prefix_reused;
                s.rotations.push_back(std::move(r));
            }
        }
        return s;
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
        // noul / choice always collect logits so semantic margins can be measured
        inp.collect_corrected_logits = cfg.collect_corrected_logits || row.input.type == Type::Noul ||
                                       row.input.type == Type::Choice;
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
        if(cfg.collect_corrected_logits)
            ir.corrected_logits = out.corrected_logits;
        ir.keys = out.keys;
        ir.prompt_token_count = out.prompt_token_count;
        ir.eval_ms = t.elapsed().count();
        // Multilingual metadata from dataset row
        ir.language = row.language;
        ir.pair_id = row.pair_id;
        ir.difficulty = row.difficulty;

        if(out.ok && out.selected >= 0 && out.selected < (int)out.keys.size()) {
            ir.stable_key = out.keys[out.selected];
        }

        ir.correct = check_correct(ir, row.expected, row.input.type);
        if(out.ok && row.input.type == Type::Noul)
            ir.noul = make_noul_sample(row, out);
        if(out.ok && row.input.type == Type::Choice)
            ir.choice = make_choice_sample(row, out, cfg.calibration.temperature_for(Type::Choice));

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

    return result;
}

// ---------------------------------------------------------------------------
// RunResult::to_json
// ---------------------------------------------------------------------------
std::vector<NoulSample> RunResult::noul_samples() const
{
    std::vector<NoulSample> samples;
    for(const auto& item: items)
        if(item.noul)
            samples.push_back(*item.noul);
    return samples;
}

std::vector<ChoiceSample> RunResult::choice_samples() const
{
    std::vector<ChoiceSample> samples;
    for(const auto& item: items)
        if(item.choice)
            samples.push_back(*item.choice);
    return samples;
}

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
    const std::vector<ChoiceSample> choice_samples_v = choice_samples();
    compute_choice_metrics(choice_samples_v).merge_into(choice_j);
    json choice_items_j = json::array();
    for(const auto& s: choice_samples_v) choice_items_j.push_back(s.to_json());
    json noul_j = {
        {"n", m.noul.n},
        {"labeled", m.noul.labeled},
        {"accuracy", m.noul.accuracy()},
        {"eval_ms", noul_eval_ms}};
    const std::vector<NoulSample> noul_samples_v = noul_samples();
    compute_noul_metrics(noul_samples_v).merge_into(noul_j);
    json noul_items_j = json::array();
    for(const auto& s: noul_samples_v) noul_items_j.push_back(s.to_json());
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
        {"noul_ensemble", c.ensemble.noul_mode == NoulEnsembleMode::BINARY_ORDER ? "binary-order" : "none"},
        {"choice_ensemble", c.ensemble.choice_mode == ChoiceEnsembleMode::CYCLIC_ROTATION
                                ? (c.ensemble.choice_prefix_reuse ? "cyclic-rotation+prefix-reuse" : "cyclic-rotation")
                                : "none"},
        {"metrics", json{
                        {"n_total", m.n_total},
                        {"n_errors", m.n_errors},
                        {"choice", choice_j},
                        {"noul", noul_j},
                        {"score", score_j}}},
        {"stability", stab_j},
        {"noul_items", noul_items_j},
        {"choice_items", choice_items_j}};
}

// ---------------------------------------------------------------------------
// CompareResult::to_json
// ---------------------------------------------------------------------------

json CompareResult::to_json() const
{
    json runs_j = json::array();
    for(const auto& r: runs) runs_j.push_back(r.to_json());
    return json{
        {"experiment", experiment_name},
        {"model", model_path},
        {"runs", runs_j}};
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
