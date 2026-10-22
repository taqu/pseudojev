cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# pjev_add_test(NAME LIBS...): one executable per test source, run from the repository root.
function(pjev_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ${ARGN})
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endfunction()

# Unit tests (no model files needed)
pjev_add_test(test_prompt_strategy  pjev-core)  # prompt layout and candidate labels
pjev_add_test(test_special_tokens   pjev-core)  # user content stays in untrusted segments
pjev_add_test(test_decision_mock    pjev-core)  # decision engine scoring with a mock backend
pjev_add_test(test_ensemble         pjev-core)  # decision engine option-order ensemble
pjev_add_test(test_perf_opt         pjev-core)  # decision engine KV prefix reuse
pjev_add_test(test_routing          pjev-core)  # task-aware routing: dispatch, kNN, mapping, errors
pjev_add_test(test_calibration      pjev-core)  # calibration loading, fitting, metrics, partitioning, output
pjev_add_test(test_jev_parsing      pjev-core)  # Jev request parsing / response serialization
pjev_add_test(test_server           pjev-core)  # API hardening: structured errors, limits
pjev_add_test(test_cli              pjev-core)  # CLI output formatting
pjev_add_test(test_worker           pjev-core)  # worker IPC framing, protocol, runtime dir
pjev_add_test(test_distribution     pjev-core)  # pjev.json and runtime path resolution
pjev_add_test(test_model_identity   pjev-core)  # SHA-256, model identity, candidate probing

if (TARGET pjev-filldb-lib)
    pjev_add_test(test_exemplar_store pjev-filldb-lib)  # read-only exemplar store, allowlist, LZ4, corruption
    pjev_add_test(test_filldb         pjev-filldb-lib)  # exemplar records, embedding text, database writer
    pjev_add_test(test_calibrate      pjev-filldb-lib)  # calibration generation on a fixture database
    # End to end with the real embedding model (skips if the model is absent)
    pjev_add_test(test_filldb_integration pjev-filldb-lib pjev-runtime)
endif()

# Golden regression against the frozen R4 policy (needs the runtime resources; skipped without them)
pjev_add_test(test_golden_regression pjev-runtime)
set_tests_properties(test_golden_regression PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 LABELS integration)
