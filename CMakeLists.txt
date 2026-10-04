cmake_minimum_required(VERSION 4.0)
project(pseudojev C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(pjev-httplib STATIC
    llama.cpp/vendor/cpp-httplib/httplib.cpp
)
target_include_directories(pjev-httplib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
)
find_package(Threads REQUIRED)
target_link_libraries(pjev-httplib PUBLIC Threads::Threads)
target_compile_features(pjev-httplib PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(pjev-httplib PRIVATE /w)
else()
    target_compile_options(pjev-httplib PRIVATE -w)
endif()
if (WIN32)
    target_link_libraries(pjev-httplib PUBLIC ws2_32)
endif()

set(PJEV_INCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(PJEV_HEADER
    src/util/timer.h
    src/inference/backend.h
    src/inference/llama_backend.h
    src/calibration/calibration.h
    src/calibration/calibration_metrics.h
    src/calibration/calibration_fit.h
    src/decision/prompt_strategy.h
    src/decision/decision_engine.h
    src/server/jev_api.h
    src/server/http_server.h
    src/experiment/config.h
    src/experiment/metrics.h
    src/experiment/permute.h
    src/experiment/dataset.h
    src/experiment/runner.h
)
source_group(include FILES ${PJEV_HEADER})

set(PJEV_SOURCE
    src/main.cpp
    src/util/timer.cpp
    src/inference/llama_backend.cpp
    src/calibration/calibration.cpp
    src/calibration/calibration_metrics.cpp
    src/calibration/calibration_fit.cpp
    src/decision/prompt_strategy.cpp
    src/decision/decision_engine.cpp
    src/server/jev_api.cpp
    src/server/http_server.cpp
    src/experiment/dataset.cpp
    src/experiment/runner.cpp
)
source_group(src FILES ${PJEV_SOURCE})

add_executable(pjev
    ${PJEV_SOURCE}
    ${PJEV_HEADER}
)
if (MSVC)
    target_compile_options(pjev PRIVATE /utf-8)
else()
endif()

target_include_directories(pjev PRIVATE ${PJEV_INCS})
target_link_libraries(pjev PRIVATE llama pjev-httplib)
set_target_properties(pjev PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

enable_testing()
add_subdirectory(tests)
