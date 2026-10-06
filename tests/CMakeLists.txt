cmake_minimum_required(VERSION 3.14)

set(PJEV_TEST_INCS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_SOURCE_DIR}/spdlog/include
)

if (MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# test_prompt_strategy: tests PromptStrategy (no backend needed)
add_executable(test_prompt_strategy test_prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp)
target_include_directories(test_prompt_strategy PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_prompt_strategy COMMAND test_prompt_strategy)

# test_special_tokens: tests that user content ends up in untrusted segments
add_executable(test_special_tokens test_special_tokens.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp)
target_include_directories(test_special_tokens PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_special_tokens COMMAND test_special_tokens)

# test_decision_mock: tests DecisionEngine logic with mock backend
add_executable(test_decision_mock test_decision_mock.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp)
target_include_directories(test_decision_mock PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_decision_mock COMMAND test_decision_mock)

# test_jev_parsing: tests Jev request parsing and response serialization
# Uses a mock backend to avoid needing a real model
add_executable(test_jev_parsing test_jev_parsing.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/server/jev_api.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp)
target_include_directories(test_jev_parsing PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_jev_parsing COMMAND test_jev_parsing)

# test_experiment: tests experiment framework (dataset parsing, runner, named experiments)
add_executable(test_experiment test_experiment.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/dataset.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/runner.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp)
target_include_directories(test_experiment PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_experiment COMMAND test_experiment)

# test_metrics: tests metrics.h (accuracy, MAE, QWK, stability)
add_executable(test_metrics test_metrics.cpp)
target_include_directories(test_metrics PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_metrics COMMAND test_metrics)

# test_calibration: tests calibration temperature scaling, metrics, fitting, artifacts, splits
add_executable(test_calibration test_calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_fit.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/dataset.cpp)
target_include_directories(test_calibration PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_calibration COMMAND test_calibration)

# test_multilingual: tests multilingual pairing, stats, validate_pairs
add_executable(test_multilingual test_multilingual.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_fit.cpp
    ${CMAKE_SOURCE_DIR}/src/multilingual/multilingual.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/dataset.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/runner.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp)
target_include_directories(test_multilingual PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_multilingual COMMAND test_multilingual)

# test_server: tests Phase 6 hardening (structured errors, request IDs, field limits, etc.)
add_executable(test_server test_server.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/server/jev_api.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp)
target_include_directories(test_server PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_server COMMAND test_server)

# test_model_gate: tests model identity, SHA-256, candidate compat, value profile, Phase5Decision
add_executable(test_model_gate test_model_gate.cpp
    ${CMAKE_SOURCE_DIR}/src/model/model_identity.cpp
    ${CMAKE_SOURCE_DIR}/src/model/value_profile.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/multilingual/multilingual.cpp)
target_include_directories(test_model_gate PRIVATE ${PJEV_TEST_INCS})
if (WIN32)
    target_link_libraries(test_model_gate PRIVATE psapi)
endif()
add_test(NAME test_model_gate COMMAND test_model_gate)

# test_perf_opt: tests Phase 7 performance optimization (KV prefix reuse, batch inference)
add_executable(test_perf_opt test_perf_opt.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp)
target_include_directories(test_perf_opt PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_perf_opt COMMAND test_perf_opt)

# test_distribution: tests Phase 8 distribution (ReleaseConfig, path resolution, get_executable_dir)
add_executable(test_distribution test_distribution.cpp
    ${CMAKE_SOURCE_DIR}/src/distribution/release_config.cpp
    ${CMAKE_SOURCE_DIR}/src/model/model_identity.cpp)
target_include_directories(test_distribution PRIVATE ${PJEV_TEST_INCS})
if (WIN32)
    target_link_libraries(test_distribution PRIVATE psapi)
endif()
add_test(NAME test_distribution COMMAND test_distribution)

# test_cli: tests Phase 9 CLI formatting, option parsing, and DecisionEngine integration
add_executable(test_cli test_cli.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cli.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp)
target_include_directories(test_cli PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_cli COMMAND test_cli)

# test_ensemble: tests E1 binary option-order ensemble for noul
add_executable(test_ensemble test_ensemble.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp)
target_include_directories(test_ensemble PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_ensemble COMMAND test_ensemble)

# test_worker: tests Phase 10 IPC framing, protocol serialization, runtime dir, config hash, WorkerInfo JSON
add_executable(test_worker test_worker.cpp
    ${CMAKE_SOURCE_DIR}/src/worker/ipc_protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/worker/runtime_dir.cpp
    ${CMAKE_SOURCE_DIR}/src/worker/local_transport.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp)
target_include_directories(test_worker PRIVATE ${PJEV_TEST_INCS})
if (WIN32)
    target_link_libraries(test_worker PRIVATE psapi)
endif()
add_test(NAME test_worker COMMAND test_worker)
