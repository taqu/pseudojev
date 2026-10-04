#include "multilingual/multilingual.h"
#include "experiment/dataset.h"
#include "experiment/runner.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else        { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

// ---------------------------------------------------------------------------
// Helper: build a minimal ItemResult for testing
// ---------------------------------------------------------------------------
static pjev::ItemResult make_item(const std::string& id,
                                   const std::string& type,
                                   bool correct,
                                   int selected,
                                   const std::vector<std::string>& keys,
                                   const std::vector<double>& probs,
                                   int correct_index = -1,
                                   const std::string& pair_id = "",
                                   const std::string& language = "",
                                   const std::string& difficulty = "")
{
    pjev::ItemResult ir;
    ir.id            = id;
    ir.type          = type;
    ir.ok            = true;
    ir.correct       = correct;
    ir.selected      = selected;
    ir.keys          = keys;
    ir.probs         = probs;
    ir.correct_index = correct_index;
    ir.pair_id       = pair_id;
    ir.language      = language;
    ir.difficulty    = difficulty;
    ir.prompt_token_count = 10 + (int)language.size();
    ir.eval_ms       = 50;
    return ir;
}

int main()
{
    using namespace pjev;

    // -----------------------------------------------------------------------
    // 1. error_transition_str
    // -----------------------------------------------------------------------
    {
        check(error_transition_str(ErrorTransition::BOTH_CORRECT)       == "both_correct",       "transition str both_correct");
        check(error_transition_str(ErrorTransition::REF_ONLY_CORRECT)   == "ref_only_correct",   "transition str ref_only_correct");
        check(error_transition_str(ErrorTransition::TARGET_ONLY_CORRECT) == "target_only_correct", "transition str target_only_correct");
        check(error_transition_str(ErrorTransition::BOTH_WRONG)         == "both_wrong",         "transition str both_wrong");
    }

    // -----------------------------------------------------------------------
    // 2. Dataset language metadata parsing
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json j = {
            {"id", "item-001"},
            {"state", "The sky is blue."},
            {"type", "noul"},
            {"expected", "true"},
            {"question", "Is the sky blue?"},
            {"language", "en"},
            {"pair_id", "item-001"},
            {"source_language", "en"},
            {"translation_provenance", "original"},
            {"difficulty", "easy"}
        };
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "metadata parse: no error");
        check(row.language == "en", "metadata parse: language=en");
        check(row.pair_id == "item-001", "metadata parse: pair_id");
        check(row.source_language == "en", "metadata parse: source_language");
        check(row.translation_provenance == "original", "metadata parse: translation_provenance");
        check(row.difficulty == "easy", "metadata parse: difficulty");
    }

    // -----------------------------------------------------------------------
    // 3. Dataset missing metadata defaults to empty strings
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json j = {
            {"id", "item-002"},
            {"state", "Hello"},
            {"type", "noul"},
            {"expected", "true"},
            {"question", "Is this a greeting?"}
        };
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "metadata defaults: no error");
        check(row.language.empty(), "metadata defaults: language empty");
        check(row.pair_id.empty(), "metadata defaults: pair_id empty");
        check(row.difficulty.empty(), "metadata defaults: difficulty empty");
    }

    // -----------------------------------------------------------------------
    // 4. validate_pairs: matching types pass
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json jbase = {
            {"id", "item-003"},
            {"state", "s"},
            {"type", "noul"},
            {"expected", "true"},
            {"question", "q"},
            {"pair_id", "item-003"}
        };
        DatasetRow ref_row, tgt_row;
        parse_row(jbase, ref_row);
        // Target with same pair_id and type
        nlohmann::ordered_json jtgt = jbase;
        jtgt["id"] = "item-003-ja";
        jtgt["language"] = "ja";
        DatasetRow tgt;
        parse_row(jtgt, tgt);

        auto violations = validate_pairs({ref_row}, {tgt});
        check(violations.empty(), "validate_pairs: matching types pass");
    }

    // -----------------------------------------------------------------------
    // 5. validate_pairs: type mismatch detected
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json jref = {
            {"id", "pair-x"}, {"state", "s"}, {"type", "noul"},
            {"expected", "true"}, {"question", "q"}, {"pair_id", "pair-x"}
        };
        nlohmann::ordered_json jtgt = {
            {"id", "pair-x-ja"}, {"state", "s2"}, {"type", "choice"},
            {"expected", "A"}, {"question", "q"},
            {"choices", {{"A", "opt A"}, {"B", "opt B"}}},
            {"pair_id", "pair-x"}
        };
        DatasetRow r1, r2;
        parse_row(jref, r1);
        parse_row(jtgt, r2);
        auto v = validate_pairs({r1}, {r2});
        check(!v.empty(), "validate_pairs: type mismatch flagged");
    }

    // -----------------------------------------------------------------------
    // 6. validate_pairs: option count mismatch detected
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json jref = {
            {"id", "pair-y"}, {"state", "s"}, {"type", "choice"},
            {"expected", "A"}, {"question", "q"},
            {"choices", {{"A", "A"}, {"B", "B"}}},
            {"pair_id", "pair-y"}
        };
        nlohmann::ordered_json jtgt = {
            {"id", "pair-y-ja"}, {"state", "s2"}, {"type", "choice"},
            {"expected", "A"}, {"question", "q"},
            {"choices", {{"A", "A"}, {"B", "B"}, {"C", "C"}}},
            {"pair_id", "pair-y"}
        };
        DatasetRow r1, r2;
        parse_row(jref, r1);
        parse_row(jtgt, r2);
        auto v = validate_pairs({r1}, {r2});
        check(!v.empty(), "validate_pairs: option count mismatch flagged");
    }

    // -----------------------------------------------------------------------
    // 7. validate_pairs: unpaired items not flagged
    // -----------------------------------------------------------------------
    {
        nlohmann::ordered_json j = {
            {"id", "solo"}, {"state", "s"}, {"type", "noul"},
            {"expected", "true"}, {"question", "q"}
        };
        DatasetRow r;
        parse_row(j, r);
        auto v = validate_pairs({r}, {});
        check(v.empty(), "validate_pairs: no target = no violation");
    }

    // -----------------------------------------------------------------------
    // 8. compare_runs: basic pairing by pair_id
    // -----------------------------------------------------------------------
    {
        // Reference: EN, 3 choice items — all correct
        RunResult ref_run;
        ref_run.config.name = "ref";
        for (int i = 0; i < 3; i++) {
            auto ir = make_item("en-" + std::to_string(i), "choice",
                                true, 0,
                                {"A", "B", "C"},
                                {0.7, 0.2, 0.1},
                                0, "pair-" + std::to_string(i), "en", "easy");
            ref_run.items.push_back(ir);
        }

        // Target: JA, 3 choice items — 2 correct, 1 wrong (pair-1 wrong)
        RunResult tgt_run;
        tgt_run.config.name = "tgt";
        for (int i = 0; i < 3; i++) {
            bool correct = (i != 1);
            int  sel     = correct ? 0 : 1;
            auto ir = make_item("ja-" + std::to_string(i), "choice",
                                correct, sel,
                                {"A", "B", "C"},
                                correct ? std::vector<double>{0.6, 0.3, 0.1}
                                        : std::vector<double>{0.3, 0.6, 0.1},
                                0, "pair-" + std::to_string(i), "ja", "easy");
            tgt_run.items.push_back(ir);
        }

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.choice.n == 3, "compare_runs: n=3 paired");
        check(ml.choice.both_correct == 2, "compare_runs: both_correct=2");
        check(ml.choice.ref_only_correct == 1, "compare_runs: ref_only_correct=1");
        check(ml.choice.target_only_correct == 0, "compare_runs: target_only_correct=0");
        check(ml.choice.both_wrong == 0, "compare_runs: both_wrong=0");
        check_near(ml.choice.ref_accuracy,    1.0,    1e-9, "compare_runs: ref_accuracy=1.0");
        check_near(ml.choice.target_accuracy, 2.0/3.0, 1e-9, "compare_runs: target_accuracy=2/3");
    }

    // -----------------------------------------------------------------------
    // 9. compare_runs: semantic consistency
    // -----------------------------------------------------------------------
    {
        // Same semantic prediction → consistent
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";

        // pair-0: both select "A" → consistent
        ref_run.items.push_back(make_item("en-0","choice",true,0,{"A","B"},{0.8,0.2},0,"pair-0","en"));
        tgt_run.items.push_back(make_item("ja-0","choice",true,0,{"A","B"},{0.7,0.3},0,"pair-0","ja"));
        // pair-1: ref selects "A", tgt selects "B" → inconsistent
        ref_run.items.push_back(make_item("en-1","choice",true,0,{"A","B"},{0.8,0.2},0,"pair-1","en"));
        tgt_run.items.push_back(make_item("ja-1","choice",false,1,{"A","B"},{0.3,0.7},0,"pair-1","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.choice.n == 2, "consistency: n=2");
        check(ml.choice.n_consistent == 1, "consistency: n_consistent=1");
        check_near(ml.choice.semantic_consistency_rate, 0.5, 1e-9, "consistency: rate=0.5");
    }

    // -----------------------------------------------------------------------
    // 10. compare_runs: items without pair_id fall back to id matching
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        // No pair_id — match on id
        ref_run.items.push_back(make_item("item-a","noul",true,1,{"No","Yes"},{0.2,0.8},1,"","en"));
        tgt_run.items.push_back(make_item("item-a","noul",true,1,{"No","Yes"},{0.3,0.7},1,"","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.noul.n == 1, "id fallback: paired on id");
    }

    // -----------------------------------------------------------------------
    // 11. compare_runs: unpaired items excluded
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        ref_run.items.push_back(make_item("en-only","choice",true,0,{"A","B"},{0.8,0.2},0,"pair-en-only","en"));
        tgt_run.items.push_back(make_item("ja-other","choice",true,0,{"A","B"},{0.8,0.2},0,"pair-ja-only","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.choice.n == 0, "unpaired: no matches = n=0");
    }

    // -----------------------------------------------------------------------
    // 12. compare_runs: confidence shift
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        // ref confidence 0.8, tgt confidence 0.6 → shift -0.2
        ref_run.items.push_back(make_item("en-0","noul",true,1,{"No","Yes"},{0.2,0.8},1,"p0","en"));
        tgt_run.items.push_back(make_item("ja-0","noul",true,1,{"No","Yes"},{0.4,0.6},1,"p0","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        // conf_shift = target_conf - ref_conf = 0.6 - 0.8 = -0.2
        check_near(ml.noul.mean_conf_shift, -0.2, 1e-9, "confidence shift: -0.2");
    }

    // -----------------------------------------------------------------------
    // 13. compare_runs: language fields in result
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja", "model.gguf");
        check(ml.reference_language == "en", "result: reference_language=en");
        check(ml.target_language == "ja",    "result: target_language=ja");
        check(ml.model_path == "model.gguf", "result: model_path set");
    }

    // -----------------------------------------------------------------------
    // 14. compare_runs: token and latency stats
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        // EN: 10+2=12 tokens, JA: 10+2=12 tokens (both use "en"/"ja" length = 2)
        ref_run.items.push_back(make_item("en-0","choice",true,0,{"A","B"},{0.8,0.2},0,"p0","en"));
        tgt_run.items.push_back(make_item("ja-0","choice",true,0,{"A","B"},{0.7,0.3},0,"p0","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.choice.ref_mean_tokens > 0.0,    "token stats: ref_mean_tokens > 0");
        check(ml.choice.target_mean_tokens > 0.0, "token stats: target_mean_tokens > 0");
        check(ml.choice.ref_mean_eval_ms >= 0.0,  "latency stats: ref_mean_eval_ms >= 0");
    }

    // -----------------------------------------------------------------------
    // 15. compare_runs: to_json round-trip contains expected keys
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        ref_run.items.push_back(make_item("en-0","choice",true,0,{"A","B"},{0.8,0.2},0,"p0","en","easy"));
        tgt_run.items.push_back(make_item("ja-0","choice",false,1,{"A","B"},{0.3,0.7},0,"p0","ja","easy"));
        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        nlohmann::json j = ml.to_json();
        check(j.contains("reference_language"), "to_json: reference_language present");
        check(j.contains("target_language"),    "to_json: target_language present");
        check(j.contains("choice"),             "to_json: choice present");
        check(j.contains("noul"),               "to_json: noul present");
        check(j.contains("score"),              "to_json: score present");
        check(j.contains("records"),            "to_json: records present");
        check(j["records"].is_array(),          "to_json: records is array");
        check(j["records"].size() == 1,         "to_json: 1 record");
        const auto& rec = j["records"][0];
        check(rec.contains("pair_id"),          "record: pair_id present");
        check(rec.contains("transition"),       "record: transition present");
        check(rec.contains("semantically_consistent"), "record: semantically_consistent present");
        check(rec["transition"] == "ref_only_correct", "record: transition = ref_only_correct");
        check(rec["difficulty"] == "easy",      "record: difficulty = easy");
    }

    // -----------------------------------------------------------------------
    // 16. PairedPrimitiveStats to_json has expected structure
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        ref_run.items.push_back(make_item("en-0","noul",true,1,{"No","Yes"},{0.2,0.8},1,"p0","en"));
        tgt_run.items.push_back(make_item("ja-0","noul",true,1,{"No","Yes"},{0.3,0.7},1,"p0","ja"));
        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        nlohmann::json j = ml.noul.to_json();
        check(j.contains("ref_accuracy"),              "stats json: ref_accuracy");
        check(j.contains("target_accuracy"),           "stats json: target_accuracy");
        check(j.contains("semantic_consistency_rate"), "stats json: semantic_consistency_rate");
        check(j.contains("mean_conf_shift"),           "stats json: mean_conf_shift");
        check(j.contains("delta"),                     "stats json: delta");
        check(j.contains("tokens"),                    "stats json: tokens");
        check(j.contains("eval_ms"),                   "stats json: eval_ms");
    }

    // -----------------------------------------------------------------------
    // 17. Mixed primitive types handled separately
    // -----------------------------------------------------------------------
    {
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        ref_run.items.push_back(make_item("en-c","choice",true,0,{"A","B"},{0.8,0.2},0,"p-c","en"));
        tgt_run.items.push_back(make_item("ja-c","choice",false,1,{"A","B"},{0.3,0.7},0,"p-c","ja"));
        ref_run.items.push_back(make_item("en-n","noul",true,1,{"No","Yes"},{0.2,0.8},1,"p-n","en"));
        tgt_run.items.push_back(make_item("ja-n","noul",true,1,{"No","Yes"},{0.3,0.7},1,"p-n","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check(ml.choice.n == 1, "mixed: choice.n=1");
        check(ml.noul.n == 1,   "mixed: noul.n=1");
        check(ml.score.n == 0,  "mixed: score.n=0");
    }

    // -----------------------------------------------------------------------
    // 18. Language degradation direction (lower-is-better interpretation)
    // -----------------------------------------------------------------------
    {
        // accuracy degrades: EN=1.0, JA=0.5 → delta = 0.5 - 1.0 = -0.5 (negative = worse)
        RunResult ref_run, tgt_run;
        ref_run.config.name = "ref";
        tgt_run.config.name = "tgt";
        ref_run.items.push_back(make_item("en-0","choice",true,0,{"A","B"},{0.8,0.2},0,"p0","en"));
        ref_run.items.push_back(make_item("en-1","choice",true,0,{"A","B"},{0.8,0.2},0,"p1","en"));
        tgt_run.items.push_back(make_item("ja-0","choice",true,0,{"A","B"},{0.8,0.2},0,"p0","ja"));
        tgt_run.items.push_back(make_item("ja-1","choice",false,1,{"A","B"},{0.3,0.7},0,"p1","ja"));

        MultilingualResult ml = compare_runs(ref_run, tgt_run, "en", "ja");
        check_near(ml.choice.ref_accuracy,    1.0, 1e-9, "degradation: ref_accuracy=1.0");
        check_near(ml.choice.target_accuracy, 0.5, 1e-9, "degradation: target_accuracy=0.5");
        // delta in to_json: target - reference
        nlohmann::json j = ml.choice.to_json();
        double delta_acc = j["delta"]["accuracy"].get<double>();
        check(delta_acc < 0.0, "degradation: delta accuracy is negative");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
