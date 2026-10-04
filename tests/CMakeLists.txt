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
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp)
target_include_directories(test_decision_mock PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_decision_mock COMMAND test_decision_mock)

# test_jev_parsing: tests Jev request parsing and response serialization
# Uses a mock backend to avoid needing a real model
add_executable(test_jev_parsing test_jev_parsing.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/server/jev_api.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp)
target_include_directories(test_jev_parsing PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_jev_parsing COMMAND test_jev_parsing)

# test_experiment: tests experiment framework (dataset parsing, runner, named experiments)
add_executable(test_experiment test_experiment.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/dataset.cpp
    ${CMAKE_SOURCE_DIR}/src/experiment/runner.cpp)
target_include_directories(test_experiment PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_experiment COMMAND test_experiment)

# test_metrics: tests metrics.h (accuracy, MAE, QWK, stability)
add_executable(test_metrics test_metrics.cpp)
target_include_directories(test_metrics PRIVATE ${PJEV_TEST_INCS})
add_test(NAME test_metrics COMMAND test_metrics)

# test_calibration: tests calibration temperature scaling, metrics, fitting, artifacts
add_executable(test_calibration test_calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_fit.cpp)
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
