#include "exemplar_record.h"
#include "embedding_text.h"
#include "exemplar_database.h"
#include "embedding/embedding.h"
#include "inference/llama_instance.h"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

#include "sqlite/sqlite3.h"

static int fails = 0;
static void check(bool cond, const char* what) {
    if(!cond) { printf("FAIL: %s\n", what); ++fails; }
    else      { printf("pass: %s\n", what); }
}

using namespace pjev;
using namespace pjev::filldb;
using json = nlohmann::ordered_json;

static const char* MODEL_PATH = "models/nomic-embed-text-v1.5.gguf";

static json make_score_ex() {
    return json::parse(R"({
        "id": "integ-score-1",
        "group_id": "integ-grp-1",
        "kind": "score",
        "question": "How many requirements are satisfied?",
        "options": ["none", "one", "two", "three", "four"],
        "target": [0.0, 0.0, 0.0, 0.0, 1.0],
        "split": "train",
        "source": "integ",
        "state": {
            "policy_name": "Example policy",
            "evidence_policy": "Facts must be explicitly stated.",
            "numbered_requirements": ["R1 holds", "R2 holds", "R3 holds", "R4 holds"]
        },
        "metadata": {
            "language": "en",
            "domain": "support",
            "family": "policy"
        }
    })");
}

static json make_choice_ex() {
    return json::parse(R"({
        "id": "integ-choice-1",
        "group_id": "integ-grp-2",
        "kind": "choice",
        "question": "Which candidate satisfies all requirements?",
        "options": ["Alice", "Bob", "Carol"],
        "target": [0.0, 1.0, 0.0],
        "split": "train",
        "source": "integ",
        "state": {"policy_name": "Candidate policy"},
        "metadata": {"language": "en", "domain": "hr", "family": "hiring"}
    })");
}

extern void register_sqlite_vec_autoext();

int main()
{
    if(!std::ifstream(MODEL_PATH).good()) {
        printf("SKIP: model not found at %s — skipping integration test\n", MODEL_PATH);
        return 0;
    }

    printf("Running integration test with real embeddings\n");
    printf("Model: %s\n\n", MODEL_PATH);

    const char* db_path = "test_filldb_integ.db";
    std::remove(db_path);

    LlamaInstance llama;

    EmbeddingConfig cfg;
    cfg.model_path = MODEL_PATH;
    cfg.verbose    = false;

    Embedding embedding(cfg);
    int emb_dim = embedding.dimension();
    printf("Embedding dimension: %d\n", emb_dim);
    check(emb_dim > 0, "integ: embedding dimension > 0");

    {
        ExemplarDatabase db(db_path);
        check(db.open(false),                                  "integ: db open");
        check(db.create_schema(emb_dim, MODEL_PATH, "test-fp"),"integ: create schema");

        auto embed_and_insert = [&](const json& j) -> bool {
            ExemplarRecord rec;
            std::string err = parse_exemplar(j, rec);
            if(!err.empty()) { printf("FAIL: parse: %s\n", err.c_str()); return false; }

            rec.embedding_text = build_document_embedding_text(
                rec.kind, rec.domain, rec.family, rec.question, rec.state, rec.options);

            std::vector<float> emb_vec;
            if(!embedding.encode(emb_vec, rec.embedding_text)) {
                printf("FAIL: embedding failed for %s\n", rec.id.c_str());
                return false;
            }
            if((int)emb_vec.size() != emb_dim) {
                printf("FAIL: unexpected dim %d for %s\n", (int)emb_vec.size(), rec.id.c_str());
                return false;
            }

            // Verify L2 norm ≈ 1
            double norm2 = 0;
            for(float f : emb_vec) norm2 += (double)f * f;
            if(std::fabs(norm2 - 1.0) >= 0.01)
                printf("FAIL: embedding not unit-norm for %s (norm2=%f)\n", rec.id.c_str(), norm2);

            db.begin_transaction();
            db.insert_group(rec);
            int64_t rowid = 0;
            bool ok = db.insert_example(rec, rowid) &&
                      db.insert_vector(rowid, emb_vec, rec.kind, rec.number_of_levels);
            if(ok) db.commit_transaction();
            else   db.rollback_transaction();
            return ok;
        };

        check(embed_and_insert(make_score_ex()),  "integ: insert score");
        check(embed_and_insert(make_choice_ex()), "integ: insert choice");
        check(db.example_count() == 2,            "integ: 2 examples");
    }

    // KNN smoke test
    {
        register_sqlite_vec_autoext();
        sqlite3* raw = nullptr;
        sqlite3_open(db_path, &raw);

        std::string query_text = "search_query: How many requirements pass for a candidate?";
        std::vector<float> q_emb;
        bool ok = embedding.encode(q_emb, query_text);
        check(ok, "integ: query embedding succeeded");

        if(ok && !q_emb.empty()) {
            // KNN on score kind
            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(raw,
                "SELECT rowid, distance FROM exemplar_vectors "
                "WHERE kind = 'score' AND embedding MATCH ? "
                "ORDER BY distance LIMIT 5;",
                -1, &stmt, nullptr);
            sqlite3_bind_blob(stmt, 1, q_emb.data(),
                              (int)(q_emb.size() * sizeof(float)), SQLITE_STATIC);

            int row_count = 0;
            bool distances_finite = true;
            while(sqlite3_step(stmt) == SQLITE_ROW) {
                ++row_count;
                double d = sqlite3_column_double(stmt, 1);
                if(!std::isfinite(d)) distances_finite = false;
            }
            sqlite3_finalize(stmt);
            check(row_count == 1,       "integ: KNN score returns 1 result");
            check(distances_finite,     "integ: KNN distances are finite");

            // KNN on choice kind should not return score
            sqlite3_prepare_v2(raw,
                "SELECT rowid FROM exemplar_vectors "
                "WHERE kind = 'choice' AND embedding MATCH ? "
                "ORDER BY distance LIMIT 5;",
                -1, &stmt, nullptr);
            sqlite3_bind_blob(stmt, 1, q_emb.data(),
                              (int)(q_emb.size() * sizeof(float)), SQLITE_STATIC);
            int choice_rows = 0;
            while(sqlite3_step(stmt) == SQLITE_ROW) ++choice_rows;
            sqlite3_finalize(stmt);
            check(choice_rows == 1, "integ: KNN choice returns 1 result (no cross-kind)");
        }

        sqlite3_close(raw);
    }

    std::remove(db_path);

    if(fails > 0) {
        printf("\n%d test(s) FAILED\n", fails);
        return 1;
    }
    printf("\nIntegration tests passed.\n");
    return 0;
}
