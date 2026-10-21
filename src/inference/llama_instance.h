#pragma once

namespace pjev
{

// RAII guard for the global llama backend (llama_backend_init/free).
// Construct exactly one instance before creating any Embedding or LlamaBackend.
class LlamaInstance
{
public:
    LlamaInstance();
    ~LlamaInstance();

    LlamaInstance(const LlamaInstance&)            = delete;
    LlamaInstance& operator=(const LlamaInstance&) = delete;
};

} // namespace pjev
