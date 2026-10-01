#include "experiment/dataset.h"
#include "experiment/runner.h"
#include "mock_backend.h"
#include <cstdio>
#include <cstring>
#include <cassert>
#include <fstream>
#include <cstdlib>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else { printf("pass: %s\n", what); }
}

// Write a string to a temp file; returns path.
static std::string write_temp(const std::string& content) {
    // Use a fixed temp name in current dir (portable for tests)
    const char* path = "test_exp_tmp.jsonl";
    std::ofstream f(path);
    f << content;
    return path;
}

int main() {
    // -----------------------------------------------------------------------
    // Test 1: load_dataset from JSONL file
    // -----------------------------------------------------------------------
    {
        std::string jsonl =
            "{\"id\":\"r1\",\"type\":\"choice\",\"state\":\"S\",\"question\":\"Q\","
            "\"choices\":{\"a\":\"A\",\"b\":\"B\"},\"expected\":\"a\"}\n"
            "{\"id\":\"r2\",\"type\":\"noul\",\"state\":\"S\",\"question\":\"Q\","
            "\"expected\":\"true\"}\n";
        std::string path = write_temp(jsonl);
        std::vector<DatasetRow> rows;
        std::string err;
        bool ok = load_dataset(path, rows, err);
        check(ok, "load_dataset JSONL: ok");
        check(rows.size() == 2, "load_dataset JSONL: 2 rows");
        check(rows[0].id == "r1", "load_dataset: row[0] id");
        check(rows[1].id == "r2", "load_dataset: row[1] id");
    }

    // -----------------------------------------------------------------------
    // Test 2: parse_row — simple choice format
    // -----------------------------------------------------------------------
    {
        json j = json::parse(
            "{\"id\":\"c1\",\"type\":\"choice\",\"state\":\"customer complained\","
            "\"question\":\"Which queue?\","
            "\"choices\":{\"billing\":\"Billing dept\",\"tech\":\"Tech support\"},"
            "\"expected\":\"billing\"}");
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "parse_row choice: no error");
        check(row.id == "c1", "parse_row choice: id");
        check(row.input.type == "choice", "parse_row choice: type");
        check(row.input.options.size() == 2, "parse_row choice: 2 options");
        check(row.input.options[0].first == "billing", "parse_row choice: key0");
        check(row.expected.is_string() && row.expected.get<std::string>() == "billing",
              "parse_row choice: expected");
    }

    // -----------------------------------------------------------------------
    // Test 3: parse_row — JevBench choice format
    // -----------------------------------------------------------------------
    {
        json j = json::parse(
            "{\"id\":\"jb1\",\"state\":\"S\","
            "\"labels\":[\"a\",\"b\"],"
            "\"question\":{\"type\":\"choice\",\"instructions\":\"Pick one.\","
            "\"criteria\":{\"a\":\"Option A\",\"b\":\"Option B\"}},"
            "\"expected\":\"a\"}");
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "parse_row JevBench choice: no error");
        check(row.input.type == "choice", "parse_row JevBench choice: type");
        check(row.input.question == "Pick one.", "parse_row JevBench choice: question");
        check(row.input.options.size() == 2, "parse_row JevBench choice: 2 options");
        check(row.input.options[0].first == "a", "parse_row JevBench choice: key0");
        check(row.input.options[0].second == "Option A", "parse_row JevBench choice: desc0");
    }

    // -----------------------------------------------------------------------
    // Test 4: parse_row — noul format
    // -----------------------------------------------------------------------
    {
        json j = json::parse(
            "{\"id\":\"n1\",\"type\":\"noul\","
            "\"state\":\"User wants refund.\","
            "\"question\":\"Refund requested?\","
            "\"expected\":\"true\"}");
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "parse_row noul: no error");
        check(row.input.type == "noul", "parse_row noul: type");
        check(row.input.options.size() == 2, "parse_row noul: 2 options");
        check(row.input.options[0].first == "false", "parse_row noul: key0=false");
        check(row.input.options[1].first == "true", "parse_row noul: key1=true");
        check(row.expected.is_string() && row.expected.get<std::string>() == "true",
              "parse_row noul: expected=true");
    }

    // -----------------------------------------------------------------------
    // Test 5: parse_row — score format
    // -----------------------------------------------------------------------
    {
        json j = json::parse(
            "{\"id\":\"s1\",\"type\":\"score\","
            "\"state\":\"I am very angry.\","
            "\"question\":\"Frustration level?\","
            "\"levels\":[\"Calm\",\"Mild\",\"Angry\",\"Furious\"],"
            "\"expected\":2}");
        DatasetRow row;
        std::string err = parse_row(j, row);
        check(err.empty(), "parse_row score: no error");
        check(row.input.type == "score", "parse_row score: type");
        check(row.input.options.size() == 4, "parse_row score: 4 options");
        check(row.input.options[2].second == "Angry", "parse_row score: level2 desc");
        check(row.expected.is_number_integer() && row.expected.get<int>() == 2,
              "parse_row score: expected=2");
    }

    // -----------------------------------------------------------------------
    // Test 6: run_experiment — choice with LETTERS scheme, A wins → correct
    // -----------------------------------------------------------------------
    {
        MockBackend mock;
        mock.set_winner(65); // "A" wins

        std::vector<DatasetRow> rows;
        DatasetRow row;
        row.id = "t1";
        row.input.type     = "choice";
        row.input.state    = "Billing issue";
        row.input.question = "Which queue?";
        row.input.options  = {{"billing","Billing"},{"tech","Technical"}};
        row.expected       = json("billing"); // A = billing
        rows.push_back(row);

        ExperimentConfig cfg;
        cfg.name   = "test-letters";
        cfg.scheme = Scheme::LETTERS;
        RunResult result = run_experiment(mock, rows, cfg);
        check(result.metrics.n_total == 1, "run_experiment LETTERS: n_total=1");
        check(result.metrics.n_errors == 0, "run_experiment LETTERS: n_errors=0");
        check(result.metrics.choice.correct == 1, "run_experiment LETTERS: A wins => correct");
        check(result.items.size() == 1, "run_experiment LETTERS: 1 item");
        check(result.items[0].ok, "run_experiment LETTERS: item ok");
        check(result.items[0].correct, "run_experiment LETTERS: item correct");
    }

    // -----------------------------------------------------------------------
    // Test 7: run_experiment — option ordering REVERSED changes order
    // -----------------------------------------------------------------------
    {
        MockBackend mock;
        mock.set_winner(65); // "A" always wins

        std::vector<DatasetRow> rows;
        DatasetRow row;
        row.id = "t2";
        row.input.type     = "choice";
        row.input.state    = "S";
        row.input.question = "Q";
        row.input.options  = {{"billing","Billing"},{"tech","Technical"}};
        row.expected       = json("billing");
        rows.push_back(row);

        // Original: A=billing (index 0), A wins => billing selected => correct
        ExperimentConfig cfg_orig;
        cfg_orig.name         = "orig";
        cfg_orig.scheme       = Scheme::LETTERS;
        cfg_orig.option_order = OptionOrder::ORIGINAL;
        RunResult r_orig = run_experiment(mock, rows, cfg_orig);
        check(r_orig.items[0].stable_key == "billing", "ordering ORIGINAL: A wins => billing");

        // Reversed: A=tech (index 0 after reverse), A wins => tech selected => wrong
        ExperimentConfig cfg_rev;
        cfg_rev.name         = "rev";
        cfg_rev.scheme       = Scheme::LETTERS;
        cfg_rev.option_order = OptionOrder::REVERSED;
        RunResult r_rev = run_experiment(mock, rows, cfg_rev);
        check(r_rev.items[0].stable_key == "tech", "ordering REVERSED: A wins => tech");
        check(!r_rev.items[0].correct, "ordering REVERSED: tech != billing => not correct");
    }

    // -----------------------------------------------------------------------
    // Test 8: exp_candidate_binding returns 2 runs
    // -----------------------------------------------------------------------
    {
        MockBackend mock;
        mock.set_winner(65); // A wins

        std::vector<DatasetRow> rows;
        DatasetRow row;
        row.id = "cb1";
        row.input.type     = "choice";
        row.input.state    = "S";
        row.input.question = "Q";
        row.input.options  = {{"a","Alpha"},{"b","Beta"}};
        row.expected       = json("a");
        rows.push_back(row);

        CompareResult cr = exp_candidate_binding(mock, rows, "test_model.gguf");
        check(cr.runs.size() == 2, "exp_candidate_binding: 2 runs");
        check(cr.experiment_name == "candidate-binding", "exp_candidate_binding: name");
        check(cr.runs[0].config.scheme == Scheme::NATURAL, "exp_candidate_binding: run0 natural");
        check(cr.runs[1].config.scheme == Scheme::LETTERS, "exp_candidate_binding: run1 letters");
    }

    // -----------------------------------------------------------------------
    // Test 9: Config serialization
    // -----------------------------------------------------------------------
    {
        ExperimentConfig cfg;
        cfg.layout       = Layout::QUESTION_FIRST;
        cfg.scheme       = Scheme::LETTERS;
        cfg.option_order = OptionOrder::REVERSED;
        check(cfg.layout_str() == "question-first", "layout_str question-first");
        check(cfg.scheme_str() == "letters", "scheme_str letters");
        check(cfg.order_str() == "reversed", "order_str reversed");

        ExperimentConfig cfg2;
        check(cfg2.layout_str() == "auto", "layout_str auto");
        check(cfg2.scheme_str() == "natural", "scheme_str natural");
        check(cfg2.order_str() == "original", "order_str original");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
