#pragma once

#include <string>
#include <vector>

#include "r0_retrieval.h"

namespace pjev
{
namespace r0
{

struct R0RunConfig {
    std::string db_path;
    std::string input_path;
    std::string model_path;
    std::string output_dir;
    std::string kind_filter;          // "" = all
    std::string split_filter;         // "" = all
    int         top_k                = 5;
    int         limit                = -1;
    bool        exclude_same_group   = false;
    bool        skip_db_records      = false;
    int         db_example_count     = 0;
    int         query_count          = 0;
    std::string db_embedding_model;
    int         db_embedding_dim     = 0;
    std::string db_distance_metric;
    std::string db_normalization;
    std::string run_timestamp;
    std::string note;
};

bool write_config(const R0RunConfig& cfg);

bool write_retrieval_jsonl(const std::string& out_dir,
                            const std::vector<QueryResult>& results);

bool write_summary_md(const std::string& out_dir,
                       const R0RunConfig& cfg,
                       const std::vector<QueryResult>& results);

bool write_examples_md(const std::string& out_dir,
                        const std::vector<QueryResult>& results);

bool write_histogram_csv(const std::string& out_dir,
                          const std::vector<QueryResult>& results);

} // namespace r0
} // namespace pjev
