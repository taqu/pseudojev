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
target_compile_features(pjev-httplib PRIVATE cxx_std_20)
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
    src/util/platform.h
    src/inference/backend.h
    src/inference/llama_backend.h
    src/calibration/calibration.h
    src/calibration/calibration_metrics.h
    src/calibration/calibration_fit.h
    src/decision/prompt_strategy.h
    src/decision/decision_engine.h
    src/decision/rotation.h
    src/server/jev_api.h
    src/server/server_config.h
    src/server/http_server.h
    src/experiment/config.h
    src/experiment/metrics.h
    src/experiment/permute.h
    src/experiment/dataset.h
    src/experiment/runner.h
    src/experiment/noul_metrics.h
    src/experiment/choice_metrics.h
    src/experiment/score_metrics.h
    src/multilingual/multilingual.h
    src/model/model_identity.h
    src/model/value_profile.h
    src/distribution/release_config.h
    src/cli/cli.h
    src/worker/ipc_protocol.h
    src/worker/runtime_dir.h
    src/worker/local_transport.h
    src/worker/worker_server.h
    src/worker/worker_client.h
    src/murmur3/murmur3.h
)
source_group(include FILES ${PJEV_HEADER})

set(PJEV_SOURCE
    src/main.cpp
    src/util/timer.cpp
    src/util/platform.cpp
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
    src/experiment/noul_metrics.cpp
    src/experiment/choice_metrics.cpp
    src/experiment/score_metrics.cpp
    src/multilingual/multilingual.cpp
    src/model/model_identity.cpp
    src/model/value_profile.cpp
    src/distribution/release_config.cpp
    src/cli/cli.cpp
    src/worker/ipc_protocol.cpp
    src/worker/runtime_dir.cpp
    src/worker/local_transport.cpp
    src/worker/worker_server.cpp
    src/worker/worker_client.cpp
    src/murmur3/murmur3.c
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
target_compile_features(pjev PRIVATE cxx_std_20)

# Embed git commit and llama.cpp revision as compile-time strings
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE PJEV_GIT_COMMIT_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (NOT PJEV_GIT_COMMIT_VAL)
    set(PJEV_GIT_COMMIT_VAL "unknown")
endif()
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
    OUTPUT_VARIABLE PJEV_LLAMA_REV_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (NOT PJEV_LLAMA_REV_VAL)
    set(PJEV_LLAMA_REV_VAL "unknown")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PJEV_VERSION_RAW)
string(STRIP "${PJEV_VERSION_RAW}" PJEV_VERSION)

# Platform/arch detection (must be before target_compile_definitions)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PJEV_PLATFORM "windows")
    set(PJEV_ARCH "x86_64")
    set(CPACK_GENERATOR "ZIP")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PJEV_PLATFORM "macos")
    execute_process(COMMAND uname -m OUTPUT_VARIABLE PJEV_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CPACK_GENERATOR "TGZ")
else()
    set(PJEV_PLATFORM "linux")
    execute_process(COMMAND uname -m OUTPUT_VARIABLE PJEV_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CPACK_GENERATOR "TGZ")
endif()

target_compile_definitions(pjev PRIVATE
    PJEV_VERSION="${PJEV_VERSION}"
    PJEV_GIT_COMMIT="${PJEV_GIT_COMMIT_VAL}"
    PJEV_LLAMA_REVISION="${PJEV_LLAMA_REV_VAL}"
    PJEV_PLATFORM="${PJEV_PLATFORM}"
    PJEV_ARCH="${PJEV_ARCH}"
)

target_include_directories(pjev PRIVATE ${PJEV_INCS})
target_link_libraries(pjev PRIVATE llama pjev-httplib)
if (WIN32)
    target_link_libraries(pjev PRIVATE psapi)
endif()
set_target_properties(pjev PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

install(TARGETS pjev RUNTIME DESTINATION .)

# CPack — release archive generation
set(CPACK_PACKAGE_NAME "pjev")
set(CPACK_PACKAGE_VERSION "${PJEV_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "pjev-${PJEV_VERSION}-${PJEV_PLATFORM}-${PJEV_ARCH}")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME};ALL;/")
include(CPack)

enable_testing()
add_subdirectory(tests)
