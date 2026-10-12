#include "llama_instance.h"

#include "llama.h"

namespace pjev
{

LlamaInstance::LlamaInstance()
{
    llama_backend_init();
}

LlamaInstance::~LlamaInstance()
{
    llama_backend_free();
}

} // namespace pjev
