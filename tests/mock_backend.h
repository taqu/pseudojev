#pragma once
#include "inference/backend.h"
#include <vector>
#include <string>

// Mock backend for testing without a real model.
// Token assignment:
//   A-J  -> 65-74 (matches ASCII)
//   No   -> 200
//   Yes  -> 201
//   0-9  -> 202-211
//   Other multi-char strings -> encoded as multiple tokens (not single)
class MockBackend : public ILlamaBackend {
public:
    static int single_token_id(const std::string& text) {
        if (text.size() == 1) {
            char c = text[0];
            if (c >= 'A' && c <= 'J') return (unsigned char)c;
        }
        if (text == "No")  return 200;
        if (text == "Yes") return 201;
        for (int i = 0; i < 10; i++) {
            if (text == std::to_string(i)) return 202 + i;
        }
        return -1; // not a single token
    }

    std::vector<int> tokenize(const std::string& text, bool, bool) const override {
        int id = single_token_id(text);
        if (id >= 0) return {id};
        // Multi-char fallback: encode each byte as a separate token
        std::vector<int> toks;
        for (unsigned char c : text) toks.push_back(1000 + c);
        return toks;
    }

    int vocab_size() const override { return 600; }

    std::string token_to_piece(int token_id) const override {
        if (token_id >= 65 && token_id <= 74) return std::string(1, (char)token_id);
        if (token_id == 200) return "No";
        if (token_id == 201) return "Yes";
        for (int i = 0; i < 10; i++) {
            if (token_id == 202 + i) return std::to_string(i);
        }
        return "?";
    }

    void set_winner(int token_id) {
        winner_id_ = token_id;
    }

    const float* eval_tokens(const std::vector<int>&) override {
        logits_.assign(600, 0.0f);
        if (winner_id_ >= 0 && winner_id_ < 600) {
            logits_[winner_id_] = 100.0f;
        }
        return logits_.data();
    }

private:
    int   winner_id_ = 65; // default "A" wins
    mutable std::vector<float> logits_;
};
