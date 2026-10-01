# Builds the Phase 0 experiment against a static, CPU-only llama.cpp build.
# First:  make llama   (configures and builds ../llama.cpp/build)
LLAMA   := ../llama.cpp
LBUILD  := $(LLAMA)/build
CXXFLAGS := -O2 -std=c++17 -Wall -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/vendor
LIBS    := $(LBUILD)/src/libllama.a $(LBUILD)/ggml/src/libggml.a $(LBUILD)/ggml/src/libggml-cpu.a $(LBUILD)/ggml/src/libggml-base.a
LDLIBS  := -fopenmp -lpthread -ldl -lm

phase0: phase0.cpp $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ phase0.cpp $(LIBS) $(LDLIBS)

llama:
	cmake -S $(LLAMA) -B $(LBUILD) -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
	      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF
	cmake --build $(LBUILD) -j --target llama

clean:
	rm -f phase0

.PHONY: llama clean
