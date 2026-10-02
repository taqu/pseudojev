#include "dataset.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace pjev
{
namespace
{
    std::string str_or(const nlohmann::json& j, const char* k, const std::string& def = "")
    {
        return (j.contains(k) && j[k].is_string()) ? j[k].get<std::string>() : def;
    }
} // namespace

std::string parse_row(const nlohmann::json& j, DatasetRow& row)
{
    using json = nlohmann::json;
    row.id = j.contains("id") ? (j["id"].is_string() ? j["id"].get<std::string>() : j["id"].dump()) : "";
    row.input.state = str_or(j, "state");
    row.expected = j.contains("expected") ? j["expected"] : json();

    const bool jevbench = j.contains("question") && j["question"].is_object();
    if(jevbench) {
        const json& q = j["question"];
        row.input.type = str_or(q, "type");
        row.input.question = str_or(q, "instructions");
        const json& c = q.contains("criteria") ? q["criteria"] : json::object();
        if(row.input.type == "noul") {
            row.input.options = {{"false", str_or(c, "false")}, {"true", str_or(c, "true")}};
            if(row.expected.is_string()) {
                std::string s = row.expected.get<std::string>();
                row.expected = (s == "yes") ? "true" : "false";
            }
        } else if(row.input.type == "choice") {
            if(j.contains("labels") && j["labels"].is_array()) {
                for(const auto& lab: j["labels"]) {
                    const std::string k2 = lab.get<std::string>();
                    row.input.options.push_back({k2, c.contains(k2) && c[k2].is_string() ? c[k2].get<std::string>() : k2});
                }
            } else {
                for(const auto& [k2, v]: c.items()) {
                    row.input.options.push_back({k2, v.is_string() ? v.get<std::string>() : ""});
                }
            }
        } else if(row.input.type == "score") {
            for(size_t i = 0; i < c.size(); i++) {
                row.input.options.push_back({std::to_string(i), c[i].is_string() ? c[i].get<std::string>() : ""});
            }
        }
    } else {
        row.input.type = str_or(j, "type");
        row.input.question = str_or(j, "question");
        if(row.input.type == "noul") {
            row.input.options = {{"false", ""}, {"true", ""}};
        } else if(row.input.type == "choice" && j.contains("choices") && j["choices"].is_object()) {
            for(const auto& [k2, v]: j["choices"].items()) {
                row.input.options.push_back({k2, v.is_string() ? v.get<std::string>() : ""});
            }
        } else if(row.input.type == "score" && j.contains("levels") && j["levels"].is_array()) {
            for(size_t i = 0; i < j["levels"].size(); i++) {
                const auto& d = j["levels"][i];
                row.input.options.push_back({std::to_string(i), d.is_string() ? d.get<std::string>() : ""});
            }
        }
    }

    if(row.input.type != "noul" && row.input.type != "choice" && row.input.type != "score") {
        return "unknown type: " + row.input.type;
    }
    if(row.input.options.size() < 2) {
        return "need at least 2 candidates";
    }
    return "";
}

bool load_dataset(const std::string& path, std::vector<DatasetRow>& rows, std::string& err)
{
    using json = nlohmann::json;
    std::stringstream ss;
    if(path == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream f(path);
        if(!f) {
            err = "cannot open: " + path;
            return false;
        }
        ss << f.rdbuf();
    }
    const std::string text = ss.str();
    // Try whole-document parse first (single object or array)
    try {
        json doc = json::parse(text);
        if(doc.is_array()) {
            for(auto& e: doc) {
                DatasetRow row;
                std::string e2 = parse_row(e, row);
                if(!e2.empty()) {
                    fprintf(stderr, "skip %s: %s\n", row.id.c_str(), e2.c_str());
                    continue;
                }
                rows.push_back(std::move(row));
            }
        } else {
            DatasetRow row;
            std::string e2 = parse_row(doc, row);
            if(!e2.empty()) {
                err = e2;
                return false;
            }
            rows.push_back(std::move(row));
        }
        return true;
    } catch(...) {
    }
    // JSONL fallback
    std::istringstream lines(text);
    std::string line;
    int ln = 0;
    while(std::getline(lines, line)) {
        ln++;
        if(line.find_first_not_of(" \t\r") == std::string::npos)
            continue;
        try {
            json j = json::parse(line);
            DatasetRow row;
            std::string e2 = parse_row(j, row);
            if(!e2.empty()) {
                fprintf(stderr, "%s:%d: skip %s: %s\n", path.c_str(), ln, row.id.c_str(), e2.c_str());
                continue;
            }
            rows.push_back(std::move(row));
        } catch(const std::exception& e) {
            fprintf(stderr, "%s:%d: JSON error: %s\n", path.c_str(), ln, e.what());
        }
    }
    return true;
}
} // namespace pjev
