cmake_minimum_required(VERSION 3.14)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${PROJECT_ROOT}/bin")

set(PJEV_R2_INCS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
    ${CMAKE_SOURCE_DIR}/spdlog/include
    ${CMAKE_SOURCE_DIR}/tools/pjevfilldb
    ${CMAKE_SOURCE_DIR}/tools/pjev-r0
    ${CMAKE_SOURCE_DIR}/tools/pjev-r1
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(pjev-r2
    main.cpp
    r2_offline.cpp
    # The R1 evaluation loop, compiled without its main(): online R2 runs exactly this harness.
    ${CMAKE_SOURCE_DIR}/tools/pjev-r1/main.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjev-r1/r1_task.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjev-r1/r1_metrics.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjev-r1/r1_output.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjev-r0/r0_retrieval.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjevfilldb/exemplar_record.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjevfilldb/embedding_text.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjevfilldb/lz4_utils.cpp
    ${CMAKE_SOURCE_DIR}/tools/pjevfilldb/vec_autoext.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/prompt_strategy.cpp
    ${CMAKE_SOURCE_DIR}/src/decision/decision_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration/calibration_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/inference/llama_backend.cpp
    ${CMAKE_SOURCE_DIR}/src/server/jev_api.cpp
    ${CMAKE_SOURCE_DIR}/src/util/timer.cpp
    ${CMAKE_SOURCE_DIR}/src/embedding/embedding.cpp
    ${CMAKE_SOURCE_DIR}/src/inference/llama_instance.cpp
    ${CMAKE_SOURCE_DIR}/src/util/platform.cpp
    ${CMAKE_SOURCE_DIR}/src/model/model_identity.cpp
    ${CMAKE_SOURCE_DIR}/src/sqlite/sqlite3.c
    ${CMAKE_SOURCE_DIR}/src/sqlite/sqlite-vec.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4hc.c
    ${CMAKE_SOURCE_DIR}/src/lz4/lz4frame.c
    ${CMAKE_SOURCE_DIR}/src/lz4/xxhash.c
)
target_include_directories(pjev-r2 PRIVATE ${PJEV_R2_INCS})
target_compile_definitions(pjev-r2 PRIVATE PJEV_R1_NO_MAIN)
target_link_libraries(pjev-r2 PRIVATE llama pjev-httplib)
if(WIN32)
    target_link_libraries(pjev-r2 PRIVATE psapi)
endif()
set_target_properties(pjev-r2 PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")
target_compile_features(pjev-r2 PRIVATE cxx_std_20)
